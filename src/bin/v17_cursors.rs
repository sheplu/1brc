//! v17: v16 with the stream cursors in named locals instead of an array.
//!
//! **Why this exists.** v16 grew `step` by about twenty instructions, and at that size LLVM
//! stopped keeping the two streams in registers. It tail-merged the pair loop's two rows with the
//! two drain loops into one shared body and reached them through a rotating pointer:
//!
//! ```text
//!   ldp x3, x23, [x19]      ; pos, end  <- &s[k]
//!   ...
//!   str x0, [x23]           ; pos'      -> &s[k]
//!   tbnz w24, #0, ...       ; k ^= 1
//! ```
//!
//! `s` is `[(usize, usize); 2]` on the stack. Once the merged body addresses it with a runtime
//! pointer, SROA cannot promote it, and every row's `pos` takes a store-to-load round trip on the
//! way to the next one. That is four to six cycles added to the recurrence — the same recurrence
//! v16 exists to shorten by six. v12 has the identical source shape and does not do this: its two
//! cursors live in `x23` and `x24` for the whole chunk.
//!
//! **The change.** Name the four values and let the array die at the split:
//!
//! ```text
//!   let ((mut p0, e0), (mut p1, e1)) = (s[0], s[1]);
//! ```
//!
//! There is then no object to take a pointer to, so the merge has nothing to merge through. The
//! cost is that the stream count stops being a `const` this binary can be swept over; `v12_pairs`
//! and `v11_stream` keep the generic form for that.
//!
//! This is the fix for a codegen accident, not a new idea about the data. If v16 and v17 measure
//! the same, the spill was free and the tail-merge was LLVM being right.
//!
//! **Measured** (`scripts/v16-newline.tsv`, REPS=5, duplicate arms 1-5 ms apart):
//!
//! ```text
//!                      t18            t8
//!   v12_pairs      376  376 ms   680  679 ms
//!   v16_newline    413  415 ms   716  720 ms
//!   v17_cursors    418  419 ms   737  740 ms
//! ```
//!
//! **The spill was not the problem, and removing it costs 5 ms at t18 and 20 ms at t8.** The fix
//! works as designed — the pair loop's frame stores go to zero and the disassembly matches v12's
//! register profile — and it is slower than leaving the round trip in.
//!
//! That is the useful result, and it is worth more than v16's. A store-to-load forward on Apple
//! silicon is four to six cycles, it was sitting on the loop-carried `pos`, and taking it off made
//! the loop slower. There is no reading of that under which the recurrence is the binding
//! constraint. LLVM's tail-merge was the better call: one shared body at 143 instructions a row
//! costs less than two copies of it, and four cursors held live across a body that size cost more
//! in register pressure than the round trip did in latency.
//!
//! The scaling says the same thing twice. At t8, where the work lands on the six wide cores and the
//! two narrow clusters are idle, the penalty is 20 ms rather than 5 — the version with *more*
//! parallelism to exploit is hurt *more* by using registers instead of memory. Latency pressure
//! does not behave that way; footprint and issue pressure do.
//!
//! Read with [`v16_newline`](../v16_newline/index.html): between them, one version shortened the
//! recurrence and lost, and one took four more cycles off it and lost harder. The pair loop is
//! throughput-bound. `+cssc`, the project's one instruction-count win, is the shape that pays —
//! fewer µops for the same work, no new loads, no new live values.
//!
//! Everything else is v16: newline-first `step`, two streams, `pread` into a recycled per-thread
//! buffer, the fixed 16-byte `;` window, no final avalanche, the keyed probe.
//!
//! Usage: v17_cursors [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

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
use obrc::parse::parse_temp_branchless;
use obrc::swar::{find_semi_and_hash_flat_raw, line_len_flat, scan_flat_keyed, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

/// See v10: 2^13..2^16 all measure the same, and this is the smallest of them.
const DEFAULT_TABLE_BITS: u32 = 14;

/// Read past the chunk so the line straddling its end can be finished locally.
const OVERLAP: usize = MAX_LINE_LEN;

/// Slack past the read that the wide path may touch: the flat scan reads a 16-byte window
/// from the name start, the parser loads 8 bytes, and the cold path's inline probe reads 32.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// v16's row, unchanged: the newline scan first, because it is the only thing the next iteration
/// waits on, then the delimiter, the value and the table hanging off it.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let next = pos + line_len_flat(buf, pos) + 1;
    let Some((semi, hash, klo, khi)) = scan_flat_keyed(buf, pos) else {
        return step_long(buf, table, pos);
    };
    let (value, _) = parse_temp_branchless(buf, semi + 1);
    table.upsert_words(buf, pos, semi - pos, hash, value, klo, khi);
    next
}

/// A name of 16 bytes or more: v10's row, out of line so it shares no registers with the hot
/// path. 2.9% of the official station list and none of this dataset's common names.
#[cold]
#[inline(never)]
fn step_long(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let (semi, hash) = find_semi_and_hash_flat_raw(buf, pos);
    let (value, next) = parse_temp_branchless(buf, semi + 1);
    table.upsert(buf, pos, semi - pos, hash, value);
    next
}

/// What one worker spent its life on.
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

        let s = split_streams::<2>(view, start, end);
        let ((mut p0, e0), (mut p1, e1)) = (s[0], s[1]);

        while p0 < e0 && p1 < e1 {
            p0 = step(&buf, &mut table, p0);
            p1 = step(&buf, &mut table, p1);
        }
        while p0 < e0 {
            p0 = step(&buf, &mut table, p0);
        }
        while p1 < e1 {
            p1 = step(&buf, &mut table, p1);
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
    // Deliberate: 18 tables, each with a key arena, would otherwise be walked and freed here
    // for no purpose. That teardown is part of the wall clock a benchmark sees.
    std::process::exit(0);
}

/// Prints where the run went. Everything is measured from the top of `main`.
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
