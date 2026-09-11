//! v18: two instructions-only removals, on the branch the old cost model said could not matter.
//!
//! **What v16 and v17 changed about the plan.** For four versions this project has been designing
//! against a latency model: the pair loop was held to be bound by the row-to-row recurrence, so the
//! only changes worth making were ones that shortened it, and everything hanging off `len` — the
//! hash, the value, the table — was written off as free. v16 shortened the recurrence and lost
//! 38 ms. v17 shortened it further, by taking a store-to-load round trip off the carried `pos`, and
//! lost 42. Both doc comments carry the numbers.
//!
//! What that leaves is a throughput-bound loop and a rate: v16 added 36 instructions a row and cost
//! 38 ms, or **about 1 ms per instruction-per-row at 18 threads**. Under that rate the items the
//! old model dismissed are worth roughly what they cost, and two of them have been sitting in the
//! source unmeasured since v5.
//!
//! **The two changes.** Both are pure removals. Neither adds a branch, a load, or a live value;
//! neither touches the recurrence or the loop's shape. This is deliberately the `+cssc` shape —
//! strictly fewer instructions for identical work — because that is the only shape this project has
//! ever measured a win from, and v13 and v15 are what happens when a removal comes bundled with a
//! control-flow change or a rematerialisation.
//!
//! [`parse_temp_branchless_mul`] folds the three digits into one multiply instead of three field
//! extracts and two chained multiply-accumulates: seven instructions to four, two multiplies to
//! one, and `#10` and `#100` stop being rebuilt inside the loop.
//!
//! [`scan_flat_keyed_tz`] builds the key mask from the `;`'s bit position rather than from the
//! length derived from it. LLVM already folds `8 * (len & 7)` into `and #0x38` on the w0 side; the
//! `+ 8` on the w1 side hides it there, costing `lsr`/`add`/`ubfiz` where one `and` would do.
//!
//! Together they should be about six instructions a row.
//!
//! **They are not, and the disassembly said so before the batch ran.** Both changes were built and
//! counted separately against v12's 214-instruction pair loop:
//!
//! ```text
//!                        insns/row   ubfx/ubfiz   mul   mov/movk/orr   loads
//!   v12                      107.0            8     8             23      24
//!   + tz mask                108.0            6     8             24      24
//!   + magic multiply         114.5            2     6             36      26
//! ```
//!
//! The arithmetic came out exactly as designed — the digit extracts go 8 → 2, the multiplies 8 → 6,
//! the `ubfiz` on the w1 side is gone — and the loop got **bigger by 7.5 instructions a row**. The
//! whole saving, and more, comes back as `mov`/`movk`/`orr`: 23 → 36. The two constants the multiply
//! needs, `0x0000000F000F0F00` and `0x640A0001`, do not fit, so the allocator pays for them by
//! rematerialising others inside the body.
//!
//! That is the third time. v13 removed instructions and got control flow back; v15 removed 28 check
//! instructions a pair and got 19 back as rebuilt constants; v18 removes six a row and gets thirteen
//! back. This loop touches 26 of the 29 usable GPRs, and it behaves like a register-pressure fixed
//! point: **a source-level removal that does not also remove a live value gets converted into
//! rematerialisation at 1:1 or worse.** v16 is the other half of the same observation — additions
//! stick, because nothing has to make room for them.
//!
//! **So the bet, restated.** Not −6 ms. The instruction count says +7.5 a row, which at v16's rate
//! is about +7 ms, and this arm now exists to test the rate as much as the change. Three outcomes
//! and what each would mean: near +7 ms and the rate is linear, so this is simply a loss and the
//! target has to become live values rather than operations; near zero and the rate is wrong, which
//! would mean v16's 38 ms was its extra *load* and not its extra instructions; below v12 and
//! rematerialised constants are cheaper than the `umaddl` chain they replaced, which would make
//! static instruction count a bad proxy in both directions.
//!
//! **Measured** (`scripts/v18-thrift.tsv`, REPS=9, duplicate arms 0-1 ms apart):
//!
//! ```text
//!                      t18            t8
//!   v12_pairs      378  378 ms   672  673 ms
//!   v18_thrift     383  382 ms   675  675 ms
//! ```
//!
//! **+4.5 ms at 18 threads, +2.5 at 8.** The first outcome, near enough: the loop is slower by
//! about what the instruction count said it would be. Against a floor of 1 ms this is a result, not
//! a coin flip — and v12 reading 378 here against 376 in the v16 batch is the two batches agreeing.
//!
//! The rate is real and it is roughly linear. v16 added 36 instructions a row for 38 ms; v18 adds
//! 7.5 for 4.5. Call it **0.6 to 1.0 ms per instruction-per-row at 18 threads**, sublinear at the
//! small end, which is what a partly-saturated machine should look like. Either way the sign is
//! settled in both directions now, and "instruction count does not matter on this loop" — carried
//! in the README since v13 — is retired.
//!
//! **What it costs the plan.** The two changes here were the last cheap arithmetic on the row, and
//! they were supposed to be the easy 6 ms. They are not available, and neither is anything shaped
//! like them, because the loop does not accept removals. Three versions have now been converted
//! into rematerialisation: v13 into control flow, v15 into 19 rebuilt constants, v18 into 13. The
//! loop is at a register-pressure fixed point at 107 instructions a row and 26 of 29 usable GPRs,
//! and LLVM will spend any register you hand it.
//!
//! So the next thing to try is not a cheaper computation. **It is one fewer live value.** The
//! register-resident loop invariants are `SEMI`, `-LOW`, the hash seed, `0x10101000` and `#28` —
//! none of them a logical immediate, all of them splat constants a NEON register could hold — plus
//! the buffer base, four table-header fields reloaded from the frame every row, and two cursors
//! with two bounds. Note that this also re-reads v13: hoisting the table header out of the frame
//! *adds* four live values to a loop that has none to spare, which is a better account of why it
//! lost than the control-flow story was, and a reason not to retry it in the form the plan
//! proposed.
//!
//! Retained rather than reverted, and `parse_temp_branchless_mul` and `scan_flat_keyed_tz` stay in
//! the library with their tests, because the arithmetic is correct and strictly better and the only
//! reason it loses is a register file. On a build with less pressure in this loop — fewer streams,
//! or the constants moved off the GPRs — they should be tried again first.
//!
//! Everything else is v12: two interleaved streams, `pread` into a recycled per-thread buffer, the
//! fixed 16-byte `;` window, no final avalanche, the keyed probe, a table that grows on demand.
//!
//! Usage: v18_thrift [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

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
use obrc::parse::parse_temp_branchless_mul;
use obrc::swar::{find_semi_and_hash_flat_raw, scan_flat_keyed_tz, FLAT_SLACK};
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

/// v12's row with the two cheaper primitives substituted. Same order, same shape, same
/// recurrence — `next` still comes out of the value's dot search.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let Some((semi, hash, klo, khi)) = scan_flat_keyed_tz(buf, pos) else {
        return step_long(buf, table, pos);
    };
    let (value, next) = parse_temp_branchless_mul(buf, semi + 1);
    table.upsert_words(buf, pos, semi - pos, hash, value, klo, khi);
    next
}

/// A name of 16 bytes or more: v10's row, out of line so it shares no registers with the hot
/// path. 2.9% of the official station list and none of this dataset's common names.
#[cold]
#[inline(never)]
fn step_long(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let (semi, hash) = find_semi_and_hash_flat_raw(buf, pos);
    let (value, next) = parse_temp_branchless_mul(buf, semi + 1);
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
