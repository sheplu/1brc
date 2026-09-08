//! Diagnostic: attributes the per-row cost of the hot loop to its parts.
//!
//! `io_floor` bounds the loop from below (what it costs merely to read the bytes). This
//! bounds it from above by removing one stage at a time, so the difference between two
//! adjacent modes is the marginal cost of the stage between them.
//!
//! Modes, cumulative:
//!   touch   XOR-reduce the bytes                     (== io_floor)
//!   parse   + locate ';' with SWAR, branchless parse
//!   hash    + fold the name into a hash (fused scan)
//!   table   + upsert into the offset-keyed table     (== v5)
//!   itable  + upsert into the inline-key table       (== v6..v8)
//!
//! Suffix `parse`, `table` or `itable` with 2/3/4/8 to walk that many independent line
//! streams, which isolates how much of the cost is the loop-carried dependency and how much
//! is the table's load/store aliasing.
//!
//! `OBRC_READER` picks how the bytes arrive: `pread` (default, what v8 ships) or `mmap`.
//! The reader is not a detail of the harness — v8 exists because swapping it was worth
//! 0.2 s, and the two put the parse loop under different memory behaviour. A mode is only
//! comparable to another mode measured with the same reader.
//!
//! `itable3` is v8's engine exactly, so `itable3 - io_floor` is the compute budget any new
//! technique has to attack.
//!
//! Usage: hot_floor <mode> [path]   (OBRC_THREADS, default 8; OBRC_READER, default pread)

use std::env;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use obrc::chunk::{chunk_bounds, local_bounds, num_chunks, split_streams, CHUNK_SIZE};
use obrc::inline_table::{InlineTable, KEY_SLACK};
use obrc::parse::parse_temp_branchless;
use obrc::swar::{
    find_semi_and_hash, find_semi_and_hash_flat, find_semi_and_hash_flat_raw, FLAT_SLACK,
};
use obrc::sys::{set_thread_qos_user_interactive, Mapping};
use obrc::table::Table;
use obrc::MAX_LINE_LEN;

/// Slots per table, as a power of two. Overridable with `OBRC_TABLE_BITS` because the size
/// is itself a thing under test: 16 is 4 MiB per thread, which holds 413 live entries
/// scattered over more 16 KB pages than the L1 dTLB can map. [`InlineTable`] grows on
/// demand, so a small value here is a cache-behaviour choice and not a capacity limit.
const DEFAULT_TABLE_BITS: u32 = 16;

/// Read past the chunk so the line straddling its end can be finished locally, as in v8.
const OVERLAP: usize = MAX_LINE_LEN;

/// Slack past a row that the wide paths may touch: the fused scan overshoots the `;`, the
/// parser loads 8 bytes, and the inline probe reads 32 from the name start.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

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

/// Flat scan + hash + parse, no table.
#[inline(always)]
fn fhash_only(data: &[u8], pos: usize) -> (u64, usize) {
    let (semi, h) = find_semi_and_hash_flat(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    (h ^ v as u64, next)
}

/// Scan + hash + parse + upsert into the offset-keyed table.
#[inline(always)]
fn full(data: &[u8], table: &mut Table, pos: usize) -> (u64, usize) {
    let (semi, h) = find_semi_and_hash(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    table.upsert_wide_cmp(data, pos, semi - pos, h, v);
    (h, next)
}

/// [`full`] against the inline-key table — the body of v8's `step`.
#[inline(always)]
fn full_inline(data: &[u8], table: &mut InlineTable, pos: usize) -> (u64, usize) {
    let (semi, h) = find_semi_and_hash(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    table.upsert(data, pos, semi - pos, h, v);
    (h, next)
}

/// [`full_inline`] with the loop scan swapped for the fixed 16-byte one. The hash is
/// bit-identical, so this must print the same sink as `itable`.
#[inline(always)]
fn full_flat(data: &[u8], table: &mut InlineTable, pos: usize) -> (u64, usize) {
    let (semi, h) = find_semi_and_hash_flat(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    table.upsert(data, pos, semi - pos, h, v);
    (h, next)
}

/// [`full_flat`] with the hash's final avalanche dropped. Same rows, different hash, so the
/// sink differs from `flat` by design — the entry count is what has to match.
#[inline(always)]
fn full_raw(data: &[u8], table: &mut InlineTable, pos: usize) -> (u64, usize) {
    let (semi, h) = find_semi_and_hash_flat_raw(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    table.upsert(data, pos, semi - pos, h, v);
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

/// Per-thread scratch. Only the table the mode actually uses is allocated at full size;
/// zeroing 4 MiB the mode never touches would land inside the measured window.
struct State {
    table: Table,
    itable: InlineTable,
    split: Vec<Table>,
    acc: u64,
    n: u64,
}

impl State {
    fn new(mode: &str, bits: u32) -> Self {
        let offset_bits = match mode {
            m if m.starts_with("itable") => 0,
            m if m.starts_with("table") || m.starts_with("split") || m == "nocmp" => bits,
            _ => 0,
        };
        let inline_bits =
            if ["itable", "flat", "raw"].iter().any(|p| mode.starts_with(p)) { bits } else { 0 };
        let split_n = match mode {
            "split2" => 2,
            "split4" => 4,
            "split8" => 8,
            _ => 0,
        };
        State {
            table: Table::new(offset_bits),
            itable: InlineTable::new(inline_bits),
            split: (0..split_n).map(|_| Table::new(bits)).collect(),
            acc: 0,
            n: 0,
        }
    }
}

/// Runs one line-aligned range. `data` must have [`TAIL_GUARD`] readable bytes past `end`.
fn run_chunk(mode: &str, data: &[u8], start: usize, end: usize, st: &mut State) {
    let State { table, itable, split, acc, n } = st;
    let mut pos = start;

    match mode {
        "touch" => {
            for c in data[start..end].chunks_exact(8) {
                *acc ^= u64::from_le_bytes(c.try_into().unwrap());
            }
        }
        "parse" => {
            while pos < end {
                let (v, next) = parse_only(data, pos);
                *acc ^= v;
                pos = next;
                *n += 1;
            }
        }
        "hash" => {
            while pos < end {
                let (semi, h) = find_semi_and_hash(data, pos);
                let (v, next) = parse_temp_branchless(data, semi + 1);
                *acc ^= h ^ v as u64;
                pos = next;
                *n += 1;
            }
        }
        "table" => {
            while pos < end {
                let (h, next) = full(data, table, pos);
                *acc ^= h;
                pos = next;
                *n += 1;
            }
        }
        "itable" => {
            while pos < end {
                let (h, next) = full_inline(data, itable, pos);
                *acc ^= h;
                pos = next;
                *n += 1;
            }
        }
        "fhash" => {
            while pos < end {
                let (semi, h) = find_semi_and_hash_flat(data, pos);
                let (v, next) = parse_temp_branchless(data, semi + 1);
                *acc ^= h ^ v as u64;
                pos = next;
                *n += 1;
            }
        }
        "flat" => {
            while pos < end {
                let (h, next) = full_flat(data, itable, pos);
                *acc ^= h;
                pos = next;
                *n += 1;
            }
        }
        "raw" => {
            while pos < end {
                let (h, next) = full_raw(data, itable, pos);
                *acc ^= h;
                pos = next;
                *n += 1;
            }
        }
        "parse2" => streamed::<2>(data, pos, end, acc, n, parse_only),
        "parse3" => streamed::<3>(data, pos, end, acc, n, parse_only),
        "parse4" => streamed::<4>(data, pos, end, acc, n, parse_only),
        "parse8" => streamed::<8>(data, pos, end, acc, n, parse_only),
        "table2" => streamed::<2>(data, pos, end, acc, n, |d, p| full(d, table, p)),
        "table3" => streamed::<3>(data, pos, end, acc, n, |d, p| full(d, table, p)),
        "table4" => streamed::<4>(data, pos, end, acc, n, |d, p| full(d, table, p)),
        "table8" => streamed::<8>(data, pos, end, acc, n, |d, p| full(d, table, p)),
        "itable2" => streamed::<2>(data, pos, end, acc, n, |d, p| full_inline(d, itable, p)),
        "itable3" => streamed::<3>(data, pos, end, acc, n, |d, p| full_inline(d, itable, p)),
        "itable4" => streamed::<4>(data, pos, end, acc, n, |d, p| full_inline(d, itable, p)),
        "itable8" => streamed::<8>(data, pos, end, acc, n, |d, p| full_inline(d, itable, p)),
        "fhash2" => streamed::<2>(data, pos, end, acc, n, fhash_only),
        "fhash4" => streamed::<4>(data, pos, end, acc, n, fhash_only),
        "flat2" => streamed::<2>(data, pos, end, acc, n, |d, p| full_flat(d, itable, p)),
        "flat3" => streamed::<3>(data, pos, end, acc, n, |d, p| full_flat(d, itable, p)),
        "flat4" => streamed::<4>(data, pos, end, acc, n, |d, p| full_flat(d, itable, p)),
        "raw2" => streamed::<2>(data, pos, end, acc, n, |d, p| full_raw(d, itable, p)),
        "raw3" => streamed::<3>(data, pos, end, acc, n, |d, p| full_raw(d, itable, p)),
        "raw4" => streamed::<4>(data, pos, end, acc, n, |d, p| full_raw(d, itable, p)),
        "nocmp" => {
            while pos < end {
                let (semi, h) = find_semi_and_hash(data, pos);
                let (v, next) = parse_temp_branchless(data, semi + 1);
                table.upsert_no_key_compare(data, pos, semi - pos, h, v);
                *acc ^= h;
                pos = next;
                *n += 1;
            }
        }
        "split2" => streamed_split::<2>(data, pos, end, acc, n, split),
        "split4" => streamed_split::<4>(data, pos, end, acc, n, split),
        "split8" => streamed_split::<8>(data, pos, end, acc, n, split),
        other => panic!("unknown mode {other:?}"),
    }
}

/// Walks the mapping directly. The last [`TAIL_GUARD`] bytes are skipped, because the wide
/// paths would read off the end of it.
fn worker_mmap(mode: &str, bits: u32, data: &[u8], total: usize, cursor: &AtomicUsize) -> State {
    set_thread_qos_user_interactive();
    let mut st = State::new(mode, bits);
    let wide_limit = data.len().saturating_sub(TAIL_GUARD);

    loop {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        let Some((start, chunk_end)) = chunk_bounds(data, i, CHUNK_SIZE) else {
            continue;
        };
        let end = chunk_end.min(wide_limit).max(start);
        run_chunk(mode, data, start, end, &mut st);
    }
    st
}

/// v8's reader: one recycled buffer per thread, `pread` per chunk. The buffer's slack past
/// EOF is what lets the wide paths cover the whole file with no scalar tail.
fn worker_pread(
    mode: &str,
    bits: u32,
    file: &File,
    len: usize,
    total: usize,
    cursor: &AtomicUsize,
) -> State {
    set_thread_qos_user_interactive();
    let mut st = State::new(mode, bits);
    let mut buf = vec![0u8; CHUNK_SIZE + OVERLAP + TAIL_GUARD];

    loop {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        let base = i * CHUNK_SIZE;
        if base >= len {
            continue;
        }
        let avail = (CHUNK_SIZE + OVERLAP).min(len - base);
        file.read_exact_at(&mut buf[..avail], base as u64).expect("pread");

        let Some((start, end)) = local_bounds(&buf[..avail], base, CHUNK_SIZE) else {
            continue;
        };
        run_chunk(mode, &buf, start, end, &mut st);
    }
    st
}

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "itable3".to_string());
    let path = env::args().nth(2).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);
    let reader = env::var("OBRC_READER").unwrap_or_else(|_| "pread".to_string());
    let bits = env::var("OBRC_TABLE_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&b| b <= 24)
        .unwrap_or(DEFAULT_TABLE_BITS);

    // An `Entry` in the offset-keyed table points into `data`, which under `pread` is a
    // buffer recycled every chunk. The previous chunk's keys become garbage, so every
    // compare misses and each thread re-inserts all 413 stations per chunk — measurably,
    // 27k entries instead of 3.3k on the 10M file. The mode still runs and still prints a
    // plausible number, which is worse than failing, so refuse the pair outright. This is
    // the reason v6 moved the key inline and v8 could then switch reader.
    let offset_keyed = mode.starts_with("table") || mode.starts_with("split") || mode == "nocmp";
    if reader == "pread" && offset_keyed {
        panic!(
            "mode {mode:?} keys the table by offset into the input, which a recycled pread \
             buffer invalidates every chunk; run it with OBRC_READER=mmap"
        );
    }

    let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let len = file.metadata().expect("stat").len() as usize;
    let total = num_chunks(len, CHUNK_SIZE);
    let cursor = AtomicUsize::new(0);

    let start = Instant::now();
    let states: Vec<State> = match reader.as_str() {
        "mmap" => {
            let mapping = Mapping::open(&path).unwrap_or_else(|e| panic!("cannot map {path}: {e}"));
            let data = mapping.as_slice();
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| scope.spawn(|| worker_mmap(&mode, bits, data, total, &cursor)))
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        }
        "pread" => std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| scope.spawn(|| worker_pread(&mode, bits, &file, len, total, &cursor)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        }),
        other => panic!("unknown reader {other:?}, expected mmap or pread"),
    };
    let secs = start.elapsed().as_secs_f64();

    // `acc` is a XOR over every row, so it is independent of how chunks happened to be
    // handed out: two modes that agree on the rows must print the same sink. The table
    // sizes are *not* stable that way (a station counts once per thread that saw it, and
    // the assignment is dynamic), so they are reported apart and never folded in.
    let mut sink = 0u64;
    let mut rows = 0u64;
    let mut distinct = 0usize;
    for st in &states {
        sink ^= st.acc;
        rows += st.n;
        distinct += st.table.len() + st.itable.len();
    }

    eprintln!(
        "{mode:>8} {reader:>5} b{bits:<2} {threads} threads  {secs:.3} s  {:.1} GB/s  \
         {:.2} ns/row  ({rows} rows, sink {sink:016x}, {distinct} entries)",
        len as f64 / secs / 1e9,
        secs * 1e9 / rows.max(1) as f64 * threads as f64,
    );
    std::process::exit(0);
}
