//! v15: v13's instruction count without v13's control flow.
//!
//! **The premise.** v13 took ~11 range-check instructions and four stack reloads out of the row
//! and came back 15 ms slower, and the README has been reading that as proof that instruction
//! count is not the budget. It proves something narrower. v13 removed those instructions by
//! changing the loop's *shape*: `first_chunk()` returns an `Option`, `probe_words` returns
//! `false` and sends the row back through `step`. Both are new exits from the hot loop, and a
//! loop with more exits has to materialize its carried state at each one. The instructions came
//! out; the register pressure went up. Meanwhile the one instruction-count win the project does
//! have — `+cssc`, 15 ms — changed no control flow at all.
//!
//! So this version removes exactly v13's *first* half — the three bounds-checked loads — and
//! nothing else. Same `step`, same cold fallback, same two streams, same table call. If the
//! shape was the cost, this wins what v13 lost. If it loses too, the README is right and the
//! loop really is not instruction bound.
//!
//! **The change.** [`step`] fetches its 16-byte name window and 8-byte value window through
//! [`window`] rather than by slicing, and hands them to `scan_flat_keyed_win` and
//! `parse_temp_branchless_win` — the two functions v13 already added, unchanged, along with the
//! two tests that hold them equal to the slice forms.
//!
//! The safety argument is the buffer sizing, which was already true and merely unexploited: the
//! worker allocates `chunk + OVERLAP + TAIL_GUARD` bytes, reads at most `chunk + OVERLAP` into
//! them, and every `pos` handed to `step` is below the chunk's line-aligned `end`, which is at
//! most what was read. So `pos + TAIL_GUARD < buf.len()` on every call, and [`TAIL_GUARD`] is
//! 123 against the 24 bytes the widest row touches. `step` states it as a `debug_assert`, so
//! every test and every `edge.sh` case checks it.
//!
//! **The safe version first.** Slicing each stream to `&buf[..end + TAIL_GUARD]` and looping on
//! `pos + TAIL_GUARD < lane.len()` is the same claim written so LLVM can verify it, and LLVM
//! does not: the implication has to cross the loop's back edge and a PHI, and `isImpliedCondition`
//! does not follow it. That version compiled to 212 instructions per pair against v12's 214 —
//! the two saved were a rematerialized constant, and all 17 `cmp`, 9 `b.hi` and 4 `cmn` survived.
//! Hence the raw pointer.
//!
//! **What happened.** It lost, and the README is right.
//!
//! ```text
//!                     median     min     max     n
//!   v12 t18            0.369   0.368   0.381     5
//!   v15 t18            0.381   0.380   0.382     5
//!   v12 t18 (dup)      0.375   0.371   0.377     5
//!   v15 t18 (dup)      0.382   0.379   0.387     5
//!   v12 t8             0.668   0.663   0.681     5
//!   v15 t8             0.681   0.678   0.690     5
//!   v12 t8 (dup)       0.668   0.664   0.676     5
//!   v15 t8 (dup)       0.682   0.674   0.694     5
//! ```
//!
//! The duplicate arms put the batch's noise floor at 6 ms (t18) and under 1 ms (t8). Against
//! that, v15 is 9.5 ms slower at 18 threads and 13.5 ms slower at 8 — four arms, one direction,
//! and the same sign and rough size as v13's 15 ms. The min/max ranges do brush at both thread
//! counts, so no single pair is decisive; the consistency across four is.
//!
//! So the loop-shape hypothesis is dead. v13 did not lose because `Option` and the re-run through
//! `step` added exits. v15 added no exit, removed 11 instructions per pair that disassembly
//! confirms are gone, and lost anyway. Removing instructions from this loop does not pay, and
//! `+cssc` — which won 15 ms by removing instructions — must have won for some other reason
//! than the count.
//!
//! The register file says what probably happened. Of the 28 instructions the checks took with
//! them, the allocator immediately put 19 back as rematerialized constants: `movk` 3 -> 10,
//! `mov` 11 -> 16, plus `lsr`, `ldp`, `cset`, `orr`. The SEED constant is now rebuilt with four
//! `mov`/`movk` inside the loop and SEMI with a `mov`/`orr` pair. Net is 5.5 instructions per row
//! against a row that was 107, and whatever those 5.5 were worth, the rematerialization chains
//! cost more. A loop this close to spilling does not have a spare instruction slot to give back;
//! it has a spare *register* problem, and taking instructions out does not fix that.
//!
//! Kept, unmerged, as the second and cleaner datapoint for a claim that previously rested on one.
//!
//! Usage: v15_bounds [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use obrc::chunk::{local_bounds, num_chunks, split_streams, CHUNK_SIZE};
use obrc::inline_table::{InlineTable, KEY_SLACK};
use obrc::output::{write_results, Stats};
use obrc::parse::{parse_temp_branchless, parse_temp_branchless_win};
use obrc::swar::{find_semi_and_hash_flat_raw, scan_flat_keyed_win, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

const DEFAULT_TABLE_BITS: u32 = 14;
const STREAMS: usize = 2;
const OVERLAP: usize = MAX_LINE_LEN;

/// Readable slack every row start is guaranteed, and now the invariant [`step`] rests on
/// rather than re-proving. The widest row touches 24 bytes from `pos`: a 15-byte name, the
/// `;`, and the parser's 8-byte window. This is 123, because the *cold* paths need a whole
/// 107-byte line and a 32-byte probe key.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// The `N` bytes at `pos`, with the range check dropped.
///
/// # Safety
///
/// `pos + N <= buf.len()`. Every caller here is covered by `step`'s precondition, which is
/// `pos + TAIL_GUARD < buf.len()` with `TAIL_GUARD` at 123 and `N` at most 24 past `pos`.
#[inline(always)]
unsafe fn window<const N: usize>(buf: &[u8], pos: usize) -> &[u8; N] {
    &*buf.as_ptr().add(pos).cast::<[u8; N]>()
}

/// Consumes the line at `pos` and returns the start of the next one.
///
/// Requires `pos + TAIL_GUARD < buf.len()`, which the worker's buffer sizing guarantees for
/// every row start below the chunk's `end`.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    debug_assert!(pos + TAIL_GUARD < buf.len());
    // SAFETY: the precondition gives 123 readable bytes from `pos`; this reads 16.
    let w = unsafe { window::<16>(buf, pos) };
    let Some((len, hash, klo, khi)) = scan_flat_keyed_win(w) else {
        return step_long(buf, table, pos);
    };
    let semi = pos + len;
    // SAFETY: `len < 16` on this path, so this reads bytes `pos + 17 ..= pos + 24`.
    let (value, advance) = parse_temp_branchless_win(unsafe { window::<8>(buf, semi + 1) });
    table.upsert_words(buf, pos, len, hash, value, klo, khi);
    semi + 1 + advance
}

/// A name of 16 bytes or more. Left on the slice forms: it is 2.9% of the official list and
/// none of this dataset, and keeping it identical to v12's means the two differ in one thing.
#[cold]
#[inline(never)]
fn step_long(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let (semi, hash) = find_semi_and_hash_flat_raw(buf, pos);
    let (value, next) = parse_temp_branchless(buf, semi + 1);
    table.upsert(buf, pos, semi - pos, hash, value);
    next
}

#[derive(Default, Clone, Copy)]
struct Spent {
    read: Duration,
    parse: Duration,
    total: Duration,
}

fn worker(
    file: &File,
    len: usize,
    chunk: usize,
    bits: u32,
    cursor: &AtomicUsize,
    total: usize,
) -> (InlineTable, Spent) {
    let t_start = Instant::now();
    set_thread_qos_user_interactive();
    let mut table = InlineTable::new(bits);
    // `step`'s precondition lives here: TAIL_GUARD bytes past everything `avail` can reach.
    let mut buf = vec![0u8; chunk + OVERLAP + TAIL_GUARD];
    let mut spent = Spent::default();

    loop {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        let base = i * chunk;
        if base >= len {
            continue;
        }
        let avail = (chunk + OVERLAP).min(len - base);
        let t0 = Instant::now();
        file.read_exact_at(&mut buf[..avail], base as u64).expect("pread");
        let t1 = Instant::now();
        spent.read += t1 - t0;
        let view = &buf[..avail];

        let Some((start, end)) = local_bounds(view, base, chunk) else {
            continue;
        };

        let mut s = split_streams::<STREAMS>(view, start, end);

        loop {
            let mut all = true;
            for k in 0..STREAMS {
                all &= s[k].0 < s[k].1;
            }
            if !all {
                break;
            }
            for k in 0..STREAMS {
                s[k].0 = step(&buf, &mut table, s[k].0);
            }
        }

        for k in 0..STREAMS {
            while s[k].0 < s[k].1 {
                s[k].0 = step(&buf, &mut table, s[k].0);
            }
        }
        spent.parse += t1.elapsed();
    }
    spent.total = t_start.elapsed();
    (table, spent)
}

fn main() {
    let t_main = Instant::now();
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);

    let chunk = env::var("OBRC_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > MAX_LINE_LEN)
        .unwrap_or(CHUNK_SIZE);

    let bits = env::var("OBRC_TABLE_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&b| (1..=24).contains(&b))
        .unwrap_or(DEFAULT_TABLE_BITS);

    let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let len = file.metadata().expect("stat").len() as usize;
    let total = num_chunks(len, chunk);
    let cursor = AtomicUsize::new(0);

    let t_setup = t_main.elapsed();

    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| scope.spawn(|| worker(&file, len, chunk, bits, &cursor, total)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    let t_workers = t_main.elapsed();

    let mut merged: BTreeMap<&[u8], Stats> = BTreeMap::new();
    for (t, _) in &results {
        for (name, stats) in t.iter() {
            merged.entry(name).and_modify(|s| s.merge(&stats)).or_insert(stats);
        }
    }

    let entries: Vec<(Vec<u8>, Stats)> = merged.into_iter().map(|(k, v)| (k.to_vec(), v)).collect();
    let t_merge = t_main.elapsed();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write_results(&mut out, &entries).unwrap();
    out.flush().unwrap();
    let t_out = t_main.elapsed();

    if env::var_os("OBRC_PHASES").is_some() {
        report_phases(&results, t_setup, t_workers, t_merge, t_out);
    }
    std::process::exit(0);
}

fn report_phases(
    results: &[(InlineTable, Spent)],
    t_setup: Duration,
    t_workers: Duration,
    t_merge: Duration,
    t_out: Duration,
) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut spent: Vec<Spent> = results.iter().map(|(_, s)| *s).collect();

    eprintln!("  setup (open+stat)   {:7.1} ms", ms(t_setup));
    eprintln!("  workers             {:7.1} ms", ms(t_workers - t_setup));
    eprintln!("  merge               {:7.1} ms", ms(t_merge - t_workers));
    eprintln!("  output              {:7.1} ms", ms(t_out - t_merge));
    eprintln!("  --- from main entry {:7.1} ms", ms(t_out));

    let sum = |f: fn(&Spent) -> Duration| spent.iter().map(|s| ms(f(s))).sum::<f64>();
    let n = spent.len() as f64;
    eprintln!(
        "  per thread, mean of {}: read {:.1} ms, parse {:.1} ms  ({:.0}% read)",
        spent.len(),
        sum(|s| s.read) / n,
        sum(|s| s.parse) / n,
        100.0 * sum(|s| s.read) / (sum(|s| s.read) + sum(|s| s.parse)),
    );

    spent.sort_by_key(|s| s.total);
    let (lo, hi) = (ms(spent[0].total), ms(spent[spent.len() - 1].total));
    eprintln!("  thread lifetime     {lo:7.1} .. {hi:.1} ms  (straggler costs {:.1} ms)", hi - lo);
}
