//! v19: the delimiter search moves to a vector register, to free two general-purpose ones.
//!
//! **What v18 established.** The pair loop is throughput-bound at roughly 0.6 to 1.0 ms per
//! instruction-per-row at 18 threads, and it is simultaneously at a register-pressure fixed
//! point: 107 instructions a row, 26 of the 29 usable GPRs live. Three versions have now had a
//! source-level removal converted straight back into rematerialised constants — v13 into control
//! flow, v15 into 19 rebuilt constants, v18 into 13 — while v16's *additions* stuck, because
//! nothing had to make room for them. The conclusion v18 closes on is that the next thing worth
//! trying is not a cheaper computation but **one fewer live value**.
//!
//! **The five constants.** `SEMI` (`0x3B3B..`), `-LOW` (`0x0101..`), the hash seed, the parser's
//! `0x10101000` and its `#28`. None is an ARM64 logical immediate, which is the only reason any
//! of them occupies a register — `HIGH` is `0x8080..`, is encodable, and costs nothing. All five
//! are splat constants, and all 32 vector registers are untouched.
//!
//! **The change.** [`scan_flat_keyed_neon`] builds the two delimiter masks with one `cmeq`
//! against a splat `;` and two extracts, instead of `eor`/`sub`/`bic`/`and` twice. That returns
//! `SEMI` to a vector register and deletes `LOW` outright: two of the five, in one change, and
//! two instructions a row cheaper into the bargain.
//!
//! Nothing else moves. Same 16-byte window, same two words feeding the hash and the key, same
//! select, same downstream arithmetic — the extracts give `0xFF` per matching byte where the
//! SWAR form gave `0x80`, and every consumer reads those masks through `trailing_zeros`, whose
//! answer is identical. No new branch, no new live value, no change to the recurrence.
//!
//! **This is not the NEON bitmap the project rejected.** That design compared 64-byte blocks,
//! built a bitmap, and drove a *separate* drain loop off the bits: the masks cost 10 ms and the
//! drain cost 194, and the row work that followed had to load and mask the name itself because
//! there was no scan left to do it — forfeiting exactly the trick that makes v11 fast. Here the
//! row loop is unchanged and the masked key still falls out of the scan. The rejected result says
//! a bitmap plus a drain loses; it says nothing about a compare in place.
//!
//! **The disassembly, before the batch.** Counted with `scripts/loopcount.py`, which reproduces
//! v12's 107.0 and v18's 114.5 exactly:
//!
//! ```text
//!                  insns/row   const-build/pair   loads/row   vector+fmov/row
//!   v12                107.0                 15        12.0               0
//!   v19                105.0                 16        14.0             3.0
//! ```
//!
//! Both halves of the design showed up and both were smaller than hoped. `SEMI` and `LOW` are
//! gone from the general-purpose file — v12's `mov x2, #0x3838383838383838` / `orr #0x3333..`
//! and `mov x16, #-0x101010101010102` / `movk #0xfeff` are not in v19's loop at all, replaced by
//! a `movi.16b v1, #0x3b` that costs one instruction and no register. But LLVM spent the freed
//! registers immediately: `#28` moves out of `x30` and gets rebuilt twice a pair, and the `ldp`
//! that fetched both words becomes two separate `ldr` because the vector load now shares the
//! addressing. Net **−2.0 instructions a row**, which at v18's rate is −1 to −2 ms: under the
//! floor, and this arm went into the batch as an attribution control rather than a candidate.
//!
//! **Measured** (`scripts/v20-vecbounds.tsv` REPS=7, then `scripts/v19-confirm.tsv` REPS=9,
//! because one arm 10× its own prediction is not a result):
//!
//! ```text
//!                     t18                    t8
//!   v12          383  384 ms   |  384  387    704  703 ms  |  711  708
//!   v19          360           |  363  359    672          |  678  678
//! ```
//!
//! **−23 ms at 18 threads and −32 at 8, and it reproduced.** Six arms across two batches, one
//! direction, on duplicate spreads of 0–4 ms. This is the project's fastest version and the
//! largest single win since v11.
//!
//! **It is 10 to 20 times what the instruction count predicted, and that is the finding.** Every
//! rate this project has is from changes that kept the work on the same execution units: v16
//! added 36 integer instructions a row and cost 38 ms, v18 added 7.5 and cost 4.5, v15 removed
//! integer instructions and *lost* because the allocator rebuilt constants. Fit against those,
//! v19's 2.0 is worth about 1.5 ms. It returned 23.
//!
//! So the throughput the loop is bound by is not instruction issue in general. It is the integer
//! side specifically. Stripping loads, stores, branches and vector work out of the counts:
//!
//! ```text
//!                  integer ops/row     measured vs v12
//!   v12                       79.5                   —
//!   v15                       80.5                  +4
//!   v19                       73.0                 −23
//! ```
//!
//! The eight `eor`/`sub`/`bic`/`and` that v12 spends per row building two delimiter masks were
//! never eight instructions' worth of cost — they were eight *integer ALU slots* on a loop that
//! has no spare ones, and one `cmeq` on an entirely idle vector unit does the same work for
//! free. The register-pressure story from v13/v15/v18 is still true and still visible in the
//! disassembly above; it was simply never the largest term.
//!
//! [`v20_vecbounds`](../v20_vecbounds/index.html) is the other half of this batch and cuts the
//! other way: it is 9.5 instructions a row *smaller* than v19 and 16 ms slower.
//!
//! **What later qualified this.** [`v21_vmask`](../v21_vmask/index.html) applies the same argument
//! to the key mask, removes 16.5 integer operations a row where v19 removed 7.5, and returns 8 ms
//! where v19 returned 23. So the integer column is not a rate either.
//!
//! What is special about v19 is the *direction* of the crossings it added, not their number. It
//! added two — `fmov x8, d0` and `mov.d x9, v0[1]`, the two extracts of the compare — and both are
//! outbound, producing the delimiter masks in the general-purpose registers where
//! `trailing_zeros` needs them anyway. v21 has to send `len` back the other way through a `dup` to
//! build its lane mask, and [`v22_prefix`](../v22_prefix/index.html) later measured that one
//! inbound round trip at 6 ms by deleting it and changing nothing else.
//!
//! The "added no crossings" claim this section used to make was a counting error, not a judgement
//! call: `scripts/loopcount.py` classified by mnemonic and filed `mov.d x9, v0[1]` under `const`.
//! The corrected integer counts are 79.5 for v12 and 72.0 here, so the removal is 7.5 rather than
//! 6.5. Neither correction changes the finding — the vector unit was idle and using it was free —
//! and both are recorded because the crossing table in the README is built out of them.
//!
//! Everything else is v12: two interleaved streams, `pread` into a recycled per-thread buffer,
//! the fixed 16-byte `;` window, no final avalanche, the keyed probe, a table that grows on
//! demand.
//!
//! Usage: v19_vecscan [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

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
use obrc::swar::{find_semi_and_hash_flat_raw, scan_flat_keyed_neon, FLAT_SLACK};
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

/// v12's row with the vector delimiter search substituted. Same order, same shape.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let Some((semi, hash, klo, khi)) = scan_flat_keyed_neon(buf, pos) else {
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
