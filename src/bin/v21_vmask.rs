//! v21: the key masking follows the compare onto the vector unit.
//!
//! **The finding this acts on.** v19 moved one operation — the `;` compare — off the integer
//! ALUs and won 23 ms at 18 threads against an instruction-count prediction of 1.5. v20 is its
//! control: 9.5 instructions a row *smaller* than v19 and 16 ms slower. Together they say the
//! pair loop is bound on integer ALU throughput specifically, not on instruction issue, and that
//! the vector unit is sitting idle beside it. Instruction count is a bad proxy in both
//! directions, and the useful question about any remaining operation is not how many there are
//! but **which unit runs them**.
//!
//! **What is left in the scan that is secretly byte-lane work.** `keep = (1 << (tz & 0x38)) - 1`
//! and `w & keep` are a byte-lane select written in scalar arithmetic: keep the bytes before the
//! `;`, drop the rest. NEON does that with one compare against a constant `[0, 1, .. 15]` and one
//! `and.16b`. And because the vector mask covers all sixteen lanes at once, `klo` and `khi` fall
//! straight out of it — the two `csel` that chose *which* word had been masked are not replaced,
//! they stop being needed. Whichever half the `;` is in, the other half is already right: kept
//! whole below it, zeroed whole above it.
//!
//! The delimiter index comes off a `shrn` rather than two extracts. Narrowing eight 16-bit lanes
//! by four leaves a nibble per input byte, so one 64-bit extract carries all sixteen compare
//! results and `trailing_zeros() >> 2` is the length outright. v19 needed two extracts, an `orr`,
//! a `cbz` and a `csel`, because a 64-bit lane only holds eight of the answers.
//!
//! Three clusters, one direction. `obrc::swar::scan_flat_keyed_vmask` has the line-by-line
//! account.
//!
//! **What deliberately does not move.** The hash. `short` still means `len < 8`, and the two
//! `csel` that pick `prev` and `last` stay exactly where v12 put them. They are on the multiply's
//! dependency chain rather than on the masking, and folding them away would mean giving short
//! names a two-round hash — a different hash, which is a different experiment and would break the
//! differential test that holds every scan in this project equal to `scan_flat_keyed`. Nothing
//! downstream of the scan changes at all: same parser, same `upsert_words`, same table, same two
//! streams, same chunking.
//!
//! Also not the mechanism: this has one range check where v19 has two, because there is one load
//! where v19 has three. v15 and v20 both removed range checks deliberately and both lost, so this
//! is recorded as a side effect and not claimed as a cause.
//!
//! **The disassembly, before the batch.** `scripts/loopcount.py`, integer = total minus loads,
//! stores, branches, vector ops and register-file transfers:
//!
//! ```text
//!                  insns/row   integer/row   vector+xfer/row   measured vs v12
//!   v12                107.0          79.5                 0                 —
//!   v19                105.0          73.0               3.0            −23 ms
//!   v20                 95.5          70.5               3.0             −7 ms
//!   v21                 91.5          56.5               9.0                 ?
//! ```
//!
//! 91.5 is the smallest this loop has ever compiled to, and 56.5 integer operations a row is a
//! different order of change from anything before it — v19 won 23 ms for 6.5, and this is 16.5.
//! Straight-lining that rate gives −58 ms, which the batch header refused to predict: the loop
//! still issues thirteen loads a row and still waits on a dependent table load, so something has
//! to become the bound before the integer side runs out. The band written down in advance was
//! **−20 to −55 ms**.
//!
//! **Measured** (`scripts/v21-vmask.tsv`, REPS=7):
//!
//! ```text
//!                      t18            t8
//!   v12           390 ms         708 ms
//!   v19           369  365       676
//!   v21           358  360       662
//! ```
//!
//! **−8 ms against v19 at 18 threads and −14 at 8**, on duplicate spreads of 4 and 2 ms. Real,
//! and the fastest version the project has: 359 ms, 31 ms below v12 in the same batch. It is also
//! between a third and a seventh of what was predicted, and the prediction was already hedged
//! wide. (This batch reads v12 at 390 where the two before it read 383 and 384 — about 1.5% slow
//! throughout, which is why only the within-batch comparison is quoted.)
//!
//! **So integer operations are not a rate either.** That is the finding, and it is the third cost
//! model this loop has broken:
//!
//! ```text
//!         integer ops removed   crossings added   measured
//!   v19                   6.5                 0     −23 ms
//!   v21                  16.5                 2      −8 ms
//! ```
//!
//! A "crossing" is a register-file transfer that did not exist before. v19 added none: the `cmeq`
//! ran on bytes already in a vector register, and the two extracts it needed produced values the
//! row wanted in general-purpose registers anyway. v21 has to send `len` *back* the other way —
//! `ctz` computes it in a GPR, `dup.16b` returns it to the vector unit to build the lane mask,
//! and the masked words come out again. The count shows it plainly: transfers go from 1.0 a row
//! to 3.0.
//!
//! And they are in series. The row's chain is now
//! `ldr q → cmeq → shrn → fmov → ctz → dup → cmhi → and.16b → fmov → hash`, where v19's was
//! `ldr q → cmeq → umov → ctz → lsl → sub → and → hash`. Four of the added links are register-file
//! crossings at roughly four cycles each, so v21 bought sixteen integer ALU slots by inserting
//! something like a fifteen-cycle detour on the chain those slots were sitting on. It came out
//! ahead. It came out barely ahead.
//!
//! The unit of account is therefore neither instructions (v20) nor integer operations (this) but
//! **integer operations removed at no additional crossing**. v19 is the only change so far that
//! managed that, which is why it is still the largest single win.
//!
//! **What this names next.** The detour is avoidable. The prefix mask does not have to be built
//! from `len`: a prefix-OR across the `cmeq` result — four `ext`/`orr` pairs, then one `bic` —
//! marks every lane at or after the first `;` without a scalar ever being involved, so `klo` and
//! `khi` stop depending on `len` and the `dup` disappears. Two chains in parallel instead of one
//! long one, and the crossing count goes back to v19's. If the account above is right that should
//! recover a real part of the missing prediction; if it lands at −8 again, crossings are not the
//! term either and this loop is bound on something nobody here has named yet.
//!
//! Usage: v21_vmask [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

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
use obrc::swar::{find_semi_and_hash_flat_raw, scan_flat_keyed_vmask, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

/// See v10: 2^13..2^16 all measure the same, and this is the smallest of them.
const DEFAULT_TABLE_BITS: u32 = 14;

/// v12's pair loop, unchanged.
const STREAMS: usize = 2;

/// Read past the chunk so the line straddling its end can be finished locally.
const OVERLAP: usize = MAX_LINE_LEN;

/// Slack past the read that the wide path may touch: the flat scan reads a 16-byte window
/// from the name start, the parser loads 8 bytes, and the cold path's inline probe reads 32.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// v19's row with the masking scan substituted. One call differs.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let Some((semi, hash, klo, khi)) = scan_flat_keyed_vmask(buf, pos) else {
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

/// What one worker spent its life on. Three `Instant::now()` per chunk is ~730 chunks a
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
