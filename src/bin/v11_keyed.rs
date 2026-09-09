//! v11: the table's probe key is taken from the scan instead of re-read from memory.
//!
//! **The observation.** Disassembling v10's inner loop turns up ~135 instructions per row,
//! against a measured ~17 cycles per row per thread — so at Apple's issue width the loop is
//! *instruction-throughput* bound, not latency bound. Roughly half those instructions are not
//! the algorithm. The largest single block is the probe key: `probe_key` loads the name's 32
//! bytes as two `u128`s, indexes a mask table twice, and `and`s — four loads, two `adrp`/`add`
//! address computations, two mask loads and four masks — and then, because a 32-byte key plus
//! three interleaved row states will not fit in the register file, LLVM spills the key to the
//! stack and reloads it four instructions later for the compare.
//!
//! **The change.** All of that recomputes bytes the scan is already holding.
//! [`scan_flat_keyed`] loads the name's first 16 bytes and masks off everything at and past the
//! `;` — it has to, or the trailing garbage would reach the hash. For any name that fits its
//! window, that masked pair *is* the probe key: `probe_key` keeps the low `len` bytes of the
//! same 16 and zeroes the rest, and its high `u128` is `MASK[0]`, which is zero. So the scan
//! hands the two words back and
//! [`upsert_words`](InlineTable::upsert_words) starts its compare with both operands already
//! in registers. One select replaces four loads, two mask lookups, four `and`s and a spill
//! pair.
//!
//! Knowing the name is under 16 bytes then collapses the probe a second time. That length
//! guarantees a zero byte inside the first 16, which no stored name of 16 bytes or more has
//! and no empty slot lacks — so the key's *low* half alone separates the query from every
//! entry the table can hold. The high half, the length check and the long-name comparison all
//! drop out, and the probe reads one `ldp` off the entry's line instead of two.
//!
//! Names of 16 bytes and over do not fit the window, so their length is not known there and
//! neither is their key. They take [`step_long`], which is v10's row verbatim; the branch is
//! the same ~33:1 one v9 already relied on, and it predicts.
//!
//! Everything else is v10: `pread` into a recycled per-thread buffer, the fixed 16-byte `;`
//! window, no final avalanche, three interleaved line streams, a table that grows on demand.
//!
//! **What it bought.** The hot row goes from ~148 instructions to ~96 and no longer spills. One
//! interleaved batch, 1e9 rows: 535 ms → **463 ms** at 18 threads (−13.5%), 946 → **784** at 8
//! (−17.1%). In `hot_floor` under `pread`, `raw3` → `keyed3` is 511 → 440 ms — 71 ms, which is 26%
//! of all compute above the 237 ms read floor and the largest single win in the project measured
//! that way.
//!
//! Usage: v11_keyed [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

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
use obrc::swar::{find_semi_and_hash_flat_raw, scan_flat_keyed, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

/// See v10: 2^13..2^16 all measure the same, and this is the smallest of them.
const DEFAULT_TABLE_BITS: u32 = 14;

/// Same as v7 — see the note there on why 4 and above lose.
const STREAMS: usize = 3;

/// Read past the chunk so the line straddling its end can be finished locally.
const OVERLAP: usize = MAX_LINE_LEN;

/// Slack past the read that the wide path may touch: the flat scan reads a 16-byte window
/// from the name start, the parser loads 8 bytes, and the cold path's inline probe reads 32.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// Consumes the line at `pos` and returns the start of the next one.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let Some((semi, hash, klo, khi)) = scan_flat_keyed(buf, pos) else {
        return step_long(buf, table, pos);
    };
    let (value, next) = parse_temp_branchless(buf, semi + 1);
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

/// What one worker spent its life on. Three `Instant::now()` per 2 MiB chunk is ~370 chunks a
/// thread and under 30 µs all told, so this is measured unconditionally and only *reported*
/// under `OBRC_PHASES`. Reading and parsing are strictly serialized within a thread, so the two
/// do add up — the interesting quantity is the ratio, and how far the slowest thread runs past
/// the fastest.
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
    // Deliberate: 18 tables, each with a key arena, would otherwise be walked and freed here
    // for no purpose. That teardown is part of the wall clock a benchmark sees.
    std::process::exit(0);
}

/// Prints where the run went. Everything is measured from the top of `main`, so the gap between
/// `t_out` and the wall clock a benchmark reports is `exec` plus dynamic linking plus teardown —
/// real cost that no in-process timer can see.
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

    // Per-thread, so these are core-milliseconds and sum to more than the wall clock.
    let sum = |f: fn(&Spent) -> Duration| spent.iter().map(|s| ms(f(s))).sum::<f64>();
    let n = spent.len() as f64;
    eprintln!(
        "  per thread, mean of {}: read {:.1} ms, parse {:.1} ms  ({:.0}% read)",
        spent.len(),
        sum(|s| s.read) / n,
        sum(|s| s.parse) / n,
        100.0 * sum(|s| s.read) / (sum(|s| s.read) + sum(|s| s.parse)),
    );

    // A thread that finishes early has idled while another still held a chunk. The spread is
    // the load imbalance, and it is wall clock nobody can use.
    spent.sort_by_key(|s| s.total);
    let (lo, hi) = (ms(spent[0].total), ms(spent[spent.len() - 1].total));
    eprintln!("  thread lifetime     {lo:7.1} .. {hi:.1} ms  (straggler costs {:.1} ms)", hi - lo);
}
