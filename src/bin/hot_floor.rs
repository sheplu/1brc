//! Diagnostic: attributes the per-row cost of the v5 hot loop to its parts.
//!
//! `io_floor` bounds the loop from below (what it costs merely to read the bytes). This
//! bounds it from above by removing one stage at a time, so the difference between two
//! adjacent modes is the marginal cost of the stage between them.
//!
//! Modes, cumulative:
//!   touch   XOR-reduce the bytes                     (== io_floor mmap)
//!   parse   + locate ';' with SWAR, branchless parse
//!   hash    + fold the name into a hash (fused scan)
//!   table   + upsert into the per-thread table       (== v5)
//!
//! Suffix `parse` or `table` with 2/4/8 to walk that many independent line streams, which
//! isolates how much of the cost is the loop-carried dependency and how much is the table's
//! load/store aliasing.
//!
//! Usage: hot_floor <mode> [path]   (OBRC_THREADS, default 8)

use std::env;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use obrc::chunk::{chunk_bounds, num_chunks, split_streams, CHUNK_SIZE};
use obrc::parse::parse_temp_branchless;
use obrc::swar::{find_semi_and_hash, SLACK};
use obrc::sys::{set_thread_qos_user_interactive, Mapping};
use obrc::table::Table;
use obrc::MAX_LINE_LEN;

const TABLE_BITS: u32 = 16;
const TAIL_GUARD: usize = MAX_LINE_LEN + SLACK;

const SEMI: u64 = 0x3B3B_3B3B_3B3B_3B3B;
const LOW: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;

/// The `;` scan from `swar`, with the hashing removed.
#[inline(always)]
fn find_semi(data: &[u8], pos: usize) -> usize {
    let mut p = pos;
    loop {
        let word = u64::from_le_bytes(data[p..p + 8].try_into().unwrap());
        let x = word ^ SEMI;
        let m = x.wrapping_sub(LOW) & !x & HIGH;
        if m != 0 {
            return p + ((m.trailing_zeros() >> 3) as usize);
        }
        p += 8;
    }
}

/// Scan + parse, no table. Returns the value and the next line.
#[inline(always)]
fn parse_only(data: &[u8], pos: usize) -> (u64, usize) {
    let semi = find_semi(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    (v as u64, next)
}

/// Scan + hash + parse + upsert.
#[inline(always)]
fn full(data: &[u8], table: &mut Table, pos: usize) -> (u64, usize) {
    let (semi, h) = find_semi_and_hash(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    table.upsert_wide_cmp(data, pos, semi - pos, h, v);
    (h, next)
}

/// As `streamed`, but each stream upserts into its own table, so no two streams ever write
/// the same address. Tests whether the shared table's load/store ordering is what stops the
/// streams from overlapping.
#[inline(always)]
fn streamed_split<const N: usize>(
    data: &[u8],
    start: usize,
    end: usize,
    acc: &mut u64,
    n: &mut u64,
    tables: &mut [Table],
) {
    let mut s = split_streams::<N>(data, start, end);
    loop {
        let mut all = true;
        for k in 0..N {
            all &= s[k].0 < s[k].1;
        }
        if !all {
            break;
        }
        for k in 0..N {
            let (h, next) = full(data, &mut tables[k], s[k].0);
            *acc ^= h;
            s[k].0 = next;
        }
        *n += N as u64;
    }
    for k in 0..N {
        while s[k].0 < s[k].1 {
            let (h, next) = full(data, &mut tables[k], s[k].0);
            *acc ^= h;
            s[k].0 = next;
            *n += 1;
        }
    }
}

/// Runs `step` over `N` independent line streams covering `[start, end)`.
#[inline(always)]
fn streamed<const N: usize>(
    data: &[u8],
    start: usize,
    end: usize,
    acc: &mut u64,
    n: &mut u64,
    mut step: impl FnMut(&[u8], usize) -> (u64, usize),
) {
    let mut s = split_streams::<N>(data, start, end);
    loop {
        let mut all = true;
        for k in 0..N {
            all &= s[k].0 < s[k].1;
        }
        if !all {
            break;
        }
        for k in 0..N {
            let (v, next) = step(data, s[k].0);
            *acc ^= v;
            s[k].0 = next;
        }
        *n += N as u64;
    }
    for k in 0..N {
        while s[k].0 < s[k].1 {
            let (v, next) = step(data, s[k].0);
            *acc ^= v;
            s[k].0 = next;
            *n += 1;
        }
    }
}

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "table".to_string());
    let path = env::args().nth(2).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);

    let mapping = Mapping::open(&path).unwrap_or_else(|e| panic!("cannot map {path}: {e}"));
    let data = mapping.as_slice();
    let total = num_chunks(data.len(), CHUNK_SIZE);
    let cursor = AtomicUsize::new(0);
    let sink = AtomicU64::new(0);
    let rows = AtomicU64::new(0);

    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                set_thread_qos_user_interactive();
                let mut table = Table::new(TABLE_BITS);
                let split_n = match mode.as_str() {
                    "split2" => 2,
                    "split4" => 4,
                    "split8" => 8,
                    _ => 0,
                };
                let mut split: Vec<Table> = (0..split_n).map(|_| Table::new(TABLE_BITS)).collect();
                let mut acc = 0u64;
                let mut n = 0u64;
                let wide_limit = data.len().saturating_sub(TAIL_GUARD);

                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= total {
                        break;
                    }
                    let Some((chunk_start, chunk_end)) = chunk_bounds(data, i, CHUNK_SIZE) else {
                        continue;
                    };
                    let end = chunk_end.min(wide_limit).max(chunk_start);
                    let mut pos = chunk_start;

                    match mode.as_str() {
                        "touch" => {
                            for c in data[chunk_start..end].chunks_exact(8) {
                                acc ^= u64::from_le_bytes(c.try_into().unwrap());
                            }
                        }
                        "parse" => {
                            while pos < end {
                                let (v, next) = parse_only(data, pos);
                                acc ^= v;
                                pos = next;
                                n += 1;
                            }
                        }
                        "hash" => {
                            while pos < end {
                                let (semi, h) = find_semi_and_hash(data, pos);
                                let (v, next) = parse_temp_branchless(data, semi + 1);
                                acc ^= h ^ v as u64;
                                pos = next;
                                n += 1;
                            }
                        }
                        "table" => {
                            while pos < end {
                                let (h, next) = full(data, &mut table, pos);
                                acc ^= h;
                                pos = next;
                                n += 1;
                            }
                        }
                        "parse2" => streamed::<2>(data, pos, end, &mut acc, &mut n, parse_only),
                        "parse4" => streamed::<4>(data, pos, end, &mut acc, &mut n, parse_only),
                        "parse8" => streamed::<8>(data, pos, end, &mut acc, &mut n, parse_only),
                        "table2" => streamed::<2>(data, pos, end, &mut acc, &mut n, |d, p| {
                            full(d, &mut table, p)
                        }),
                        "table4" => streamed::<4>(data, pos, end, &mut acc, &mut n, |d, p| {
                            full(d, &mut table, p)
                        }),
                        "table8" => streamed::<8>(data, pos, end, &mut acc, &mut n, |d, p| {
                            full(d, &mut table, p)
                        }),
                        "nocmp" => {
                            while pos < end {
                                let (semi, h) = find_semi_and_hash(data, pos);
                                let (v, next) = parse_temp_branchless(data, semi + 1);
                                table.upsert_no_key_compare(data, pos, semi - pos, h, v);
                                acc ^= h;
                                pos = next;
                                n += 1;
                            }
                        }
                        "split2" => {
                            streamed_split::<2>(data, pos, end, &mut acc, &mut n, &mut split)
                        }
                        "split4" => {
                            streamed_split::<4>(data, pos, end, &mut acc, &mut n, &mut split)
                        }
                        "split8" => {
                            streamed_split::<8>(data, pos, end, &mut acc, &mut n, &mut split)
                        }
                        other => panic!("unknown mode {other:?}"),
                    }
                }
                acc ^= table.len() as u64;
                sink.fetch_xor(acc, Ordering::Relaxed);
                rows.fetch_add(n, Ordering::Relaxed);
            });
        }
    });

    let secs = start.elapsed().as_secs_f64();
    let n = rows.load(Ordering::Relaxed).max(1);
    eprintln!(
        "{mode:>6}  {secs:.3} s  {:.1} GB/s  {:.2} ns/row  ({} rows, sink {:016x})",
        data.len() as f64 / secs / 1e9,
        secs * 1e9 / n as f64 * threads as f64,
        n,
        sink.load(Ordering::Relaxed)
    );
    std::process::exit(0);
}
