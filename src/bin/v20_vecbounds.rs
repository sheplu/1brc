//! v20: v19's vector scan and v15's unchecked windows, which only compose one way.
//!
//! **Why these two together.** v15 removed the row's range checks — 28 instructions a pair,
//! disassembly confirmed gone — and came back 9.5 ms slower. The allocator put 19 of the 28
//! straight back as rematerialised constants, and its doc comment names them: `SEED` rebuilt
//! with four `mov`/`movk` inside the loop, and `SEMI` with a `mov`/`orr` pair. That is the
//! project's clearest single picture of the register-pressure fixed point v18 later measured a
//! rate for.
//!
//! Half of that specific failure is a general-purpose register holding `0x3B3B..`, and
//! [`v19_vecscan`](../v19_vecscan/index.html) does not have one: the compare against `;` happens
//! in a vector register, so `SEMI` becomes a `movi.16b` immediate and `LOW` stops existing. v19
//! measured that alone as −2.0 instructions a row, which is under the noise floor and not worth
//! a batch by itself. Its value is that it changes what v15's removal costs.
//!
//! So the composition is the experiment, and it is a real one rather than two changes bundled
//! for convenience: v15 is a measured loss, v19 is a measured near-nothing, and the claim is
//! that together they are neither. If the fixed-point model is right, the 28 checks come out
//! and fewer of them come back, because there is one less constant competing for the file. If
//! v20 lands where v15 did, then the pressure is somewhere other than the constants and the
//! model needs the next term.
//!
//! **What changed from v15.** One call. [`step`] fetches the same 16-byte window through the
//! same [`window`] helper under the same invariant, and hands it to `scan_flat_keyed_neon_win`
//! instead of `scan_flat_keyed_win`. Same safety argument, same `debug_assert`, same cold
//! fallback, same two streams, same table call, same parser.
//!
//! **The disassembly said the composition worked.** Counted with `scripts/loopcount.py` before
//! the batch:
//!
//! ```text
//!                  insns/row   const-build/pair   bounds/row   branches/row
//!   v12                107.0                 15          3.5           13.5
//!   v15                101.5                 30          1.0            8.0
//!   v19                105.0                 16          4.0           13.0
//!   v20                 95.5                 22          1.0            8.0
//! ```
//!
//! Exactly the predicted shape. Apart, the two changes are worth −5.5 and −2.0 instructions a
//! row; together they are worth −11.5, because `SEMI` and `LOW` stop being rebuilt at all —
//! v15's eight rematerialisation instructions a pair for those two constants are simply absent
//! from v20. It is the smallest this loop has ever been, and the only version under 100.
//!
//! **Measured** (`scripts/v20-vecbounds.tsv`, REPS=7):
//!
//! ```text
//!                      t18            t8
//!   v12           383  384 ms    704  703 ms
//!   v15           387            —
//!   v19           360            672
//!   v20           376  377       702  702
//! ```
//!
//! **v20 is 7 ms faster than v12 and 16 ms slower than v19**, on a floor of 1 ms — and at 8
//! threads it gives the whole thing back, 702 against v12's 704 where v19 reads 672. Being 9.5
//! instructions a row smaller than v19 bought nothing and cost most of what v19 won.
//!
//! **So the checks are not overhead.** That now has two independent measurements in two register
//! regimes: v15 removed them and lost 9.5 ms with the constants under maximum pressure, v20
//! removed them and lost 16 ms with `SEMI` and `LOW` out of the file entirely. The second is the
//! cleaner datapoint, because v19 is its only control — one call differs, and every
//! rematerialisation v15's loss was blamed on is gone from it. Whatever the range checks cost,
//! taking them out costs more, and the register file is not the reason.
//!
//! v19's writeup has the account this fits: the loop is bound on integer ALU throughput, not on
//! instruction issue. The range checks are `cmn`/`cmp`/`b.hi` — three of the loop's cheapest and
//! most predictable slots — and the code that replaces them still has to compute the same
//! addresses. Removing them frees issue slots the loop was not short of, and then the allocator
//! spends the room: v20 rebuilds `SEED` with four `mov`/`movk` a row where v19 does not, so the
//! *integer* count only falls from 73.0 to 70.5 while the instruction count falls by 9.5.
//!
//! Stage 2 of the plan this project has been working from — "make the bounds checks provably
//! dead" — is closed by this rather than by v15. That plan read v13 as saying the checks were
//! ~13 instructions a row of pure overhead and the only question was how to remove them without
//! adding control flow. v20 removes them without adding control flow and loses anyway.
//!
//! Kept, unmerged, as the arm that makes v19's result mean something narrower and more useful
//! than "vectors are fast".
//!
//! Usage: v20_vecbounds [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

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
use obrc::swar::{find_semi_and_hash_flat_raw, scan_flat_keyed_neon_win, FLAT_SLACK};
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

/// v15's row, with the vector delimiter search substituted.
///
/// Requires `pos + TAIL_GUARD < buf.len()`, which the worker's buffer sizing guarantees for
/// every row start below the chunk's `end`.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    debug_assert!(pos + TAIL_GUARD < buf.len());
    // SAFETY: the precondition gives 123 readable bytes from `pos`; this reads 16.
    let w = unsafe { window::<16>(buf, pos) };
    let Some((len, hash, klo, khi)) = scan_flat_keyed_neon_win(w) else {
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
