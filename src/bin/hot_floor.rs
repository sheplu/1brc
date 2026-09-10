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
//! `OBRC_READER` picks how the bytes arrive: `pread` (default, what v8 ships), `mmap`, or
//! `mmap_warm`. The reader is not a detail of the harness — v8 exists because swapping it was
//! worth 0.2 s, and the two put the parse loop under different memory behaviour. A mode is only
//! comparable to another mode measured with the same reader.
//!
//! `mmap_warm` is `mmap` with every page faulted in *before* the clock starts. No real version
//! could do that — the faults are work — so it is a decomposition and never a result: it prices
//! the parse loop running over resident pages, which is the floor of any design that parses the
//! mapping in place instead of copying it into a buffer.
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

use obrc::block::{block_delims, block_newlines, block_semis, BLOCK};
use obrc::chunk::{chunk_bounds, local_bounds, num_chunks, split_streams, CHUNK_SIZE};
use obrc::inline_table::{probe_words, Entry, InlineTable, KEY_SLACK};
use obrc::parse::{parse_temp_branchless, parse_temp_branchless_win, parse_temp_len};
use obrc::swar::{
    find_semi_and_hash, find_semi_and_hash_flat, find_semi_and_hash_flat_raw, hash_len_raw,
    scan_flat_keyed, scan_flat_keyed_win, FLAT_SLACK,
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
const _: () = assert!(TAIL_GUARD >= BLOCK);

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

/// Loop scan + hash + parse, no table.
#[inline(always)]
fn hash_only(data: &[u8], pos: usize) -> (u64, usize) {
    let (semi, h) = find_semi_and_hash(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    (h ^ v as u64, next)
}

/// [`full`] with the key comparison removed, to price it.
#[inline(always)]
fn nocmp(data: &[u8], table: &mut Table, pos: usize) -> (u64, usize) {
    let (semi, h) = find_semi_and_hash(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    table.upsert_no_key_compare(data, pos, semi - pos, h, v);
    (h, next)
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

/// [`full_raw`] with the probe key taken from the scan rather than rebuilt from memory, and the
/// probe collapsed to the key's low half: v11's row. Names of 16 bytes and over fall out of the
/// scan's window, so they take `full_raw` itself, out of line. Same hash as `raw`, so the sink
/// must match it exactly.
#[inline(always)]
fn full_keyed(data: &[u8], table: &mut InlineTable, pos: usize) -> (u64, usize) {
    let Some((semi, h, klo, khi)) = scan_flat_keyed(data, pos) else {
        return full_keyed_long(data, table, pos);
    };
    let (v, next) = parse_temp_branchless(data, semi + 1);
    table.upsert_words(data, pos, semi - pos, h, v, klo, khi);
    (h, next)
}

#[cold]
#[inline(never)]
fn full_keyed_long(data: &[u8], table: &mut InlineTable, pos: usize) -> (u64, usize) {
    full_raw(data, table, pos)
}

/// [`full_keyed`] with its two fixed-size loads fetched as windows whose length the type
/// system carries, so neither the scan nor the parse checks anything. Same table path, so the
/// difference against `keyed` is exactly the bounds checks.
///
/// `data` is the whole buffer, which runs [`TAIL_GUARD`] past the last row, so a window that
/// does not fit means the caller is off the end rather than that this row is unusual. Sending
/// that to the same cold path the scan already uses for long names keeps the hot body to one
/// exit.
#[inline(always)]
fn full_keyed_win(data: &[u8], table: &mut InlineTable, pos: usize) -> (u64, usize) {
    let Some(row) = keyed_win(data, pos) else {
        return full_keyed_long(data, table, pos);
    };
    let (len, h, klo, khi, v, next) = row;
    table.upsert_words(data, pos, len, h, v, klo, khi);
    (h, next)
}

/// The window half of [`full_keyed_win`], shared with [`step_hot`]: scan and parse, no table.
///
/// Returns the name length rather than the `;` position, which is what both callers want.
#[inline(always)]
fn keyed_win(data: &[u8], pos: usize) -> Option<(usize, u64, u64, u64, i16, usize)> {
    let (len, h, klo, khi) = scan_flat_keyed_win(data.get(pos..)?.first_chunk()?)?;
    let val = pos + len + 1;
    let (v, adv) = parse_temp_branchless_win(data.get(val..)?.first_chunk()?);
    Some((len, h, klo, khi, v, val + adv))
}

/// One row against slots borrowed out of the table, so the header stays in registers.
///
/// `None` means the row could not be finished here — a name too long for the window, or a
/// probe that reached an empty slot — and in both cases **nothing has been written**. The
/// caller drops the borrow and repeats the row through [`full_keyed`], which owns the table.
#[inline(always)]
fn step_hot<const WIN: bool>(
    data: &[u8],
    slots: &mut [Entry],
    shift: u32,
    pos: usize,
) -> Option<(u64, usize)> {
    let (len, h, klo, khi, v, next) = if WIN {
        keyed_win(data, pos)?
    } else {
        let (semi, h, klo, khi) = scan_flat_keyed(data, pos)?;
        let (v, next) = parse_temp_branchless(data, semi + 1);
        (semi - pos, h, klo, khi, v, next)
    };

    debug_assert!(len < 16);
    debug_assert_eq!(
        obrc::inline_table::key_from_words(klo, khi),
        obrc::inline_table::probe_key(data, pos, len)
    );

    let key_lo = (klo as u128) | ((khi as u128) << 64);
    probe_words(slots, shift, h, key_lo, v).then_some((h, next))
}

/// [`streamed`] with the table's slots and shift held in registers across a run of rows.
///
/// The borrow is what buys the codegen. `upsert_words` keeps `&mut InlineTable` live across an
/// `insert` that may reallocate `entries`, so LLVM reloads `shift`, `entries.ptr`,
/// `entries.len` and `mask` from the frame on every row. Handing out the slice proves that
/// cannot happen, and all four hoist out of the loop.
///
/// The price is that a row which has to insert cannot run under the borrow. [`step_hot`]
/// returns `None` having written nothing, the run ends, the borrow drops, and that one row is
/// redone through `full_keyed`. Once per distinct station — 413 times per thread over the
/// whole file — so the repeated scan is not a cost. Streams that already advanced in the
/// interrupted batch simply continue from where they are.
#[inline(always)]
fn streamed_hot<const N: usize, const WIN: bool>(
    data: &[u8],
    start: usize,
    end: usize,
    acc: &mut u64,
    n: &mut u64,
    table: &mut InlineTable,
) {
    let (mut a, mut rows) = (*acc, *n);
    let mut s = split_streams::<N>(data, start, end);

    loop {
        // `N` when the interleaved loop ran out of rows, otherwise the stream that missed.
        let cold = {
            let (slots, shift) = table.hot_slots();
            'run: loop {
                let mut all = true;
                for k in 0..N {
                    all &= s[k].0 < s[k].1;
                }
                if !all {
                    break 'run N;
                }
                for k in 0..N {
                    let Some((v, next)) = step_hot::<WIN>(data, slots, shift, s[k].0) else {
                        break 'run k;
                    };
                    a ^= v;
                    s[k].0 = next;
                    rows += 1;
                }
            }
        };
        if cold == N {
            break;
        }
        let (v, next) = full_keyed(data, table, s[cold].0);
        a ^= v;
        s[cold].0 = next;
        rows += 1;
    }

    // The few rows past the point where the first stream ran dry, as in `streamed`.
    for k in 0..N {
        while s[k].0 < s[k].1 {
            let (v, next) = full_keyed(data, table, s[k].0);
            a ^= v;
            s[k].0 = next;
            rows += 1;
        }
    }
    (*acc, *n) = (a, rows);
}

/// One row's work, as a trait rather than a closure or a fn item.
///
/// Both of those reach the driver through a compiler-generated `Fn::call` shim, and nothing can
/// mark that shim `#[inline(always)]`. LLVM then inlines it on cost — and the cost includes the
/// always-inline body it wraps, so the shim lands over threshold for exactly the modes that do
/// the most work, which are the ones the harness exists to measure. `raw2` shipped a call per
/// row while `raw3` did not. A trait method is a direct call to a function written here, so the
/// attribute goes where it is needed and the outcome stops depending on a threshold.
trait Step {
    /// Returns the value to fold into the sink and the start of the next row.
    fn step(&mut self, data: &[u8], pos: usize) -> (u64, usize);
}

/// [`Step`] for a driver that already knows where the row ends.
trait Row {
    fn row(&mut self, data: &[u8], pos: usize, nl: usize) -> u64;
}

/// [`Row`] for a driver that also knows where the `;` is.
trait Pair {
    fn pair(&mut self, data: &[u8], pos: usize, semi: usize, nl: usize) -> u64;
}

macro_rules! step {
    ($name:ident, $body:expr) => {
        struct $name;
        impl Step for $name {
            #[inline(always)]
            fn step(&mut self, data: &[u8], pos: usize) -> (u64, usize) {
                $body(data, pos)
            }
        }
    };
    ($name:ident, $ctx:ty, $body:expr) => {
        struct $name<'a>(&'a mut $ctx);
        impl Step for $name<'_> {
            #[inline(always)]
            fn step(&mut self, data: &[u8], pos: usize) -> (u64, usize) {
                $body(data, &mut *self.0, pos)
            }
        }
    };
}

step!(ParseOnly, parse_only);
step!(HashOnly, hash_only);
step!(FhashOnly, fhash_only);
step!(Full, Table, full);
step!(Nocmp, Table, nocmp);
step!(FullInline, InlineTable, full_inline);
step!(FullFlat, InlineTable, full_flat);
step!(FullRaw, InlineTable, full_raw);
step!(FullKeyed, InlineTable, full_keyed);
step!(FullKeyedWin, InlineTable, full_keyed_win);

struct RowFlat;
impl Row for RowFlat {
    #[inline(always)]
    fn row(&mut self, data: &[u8], pos: usize, nl: usize) -> u64 {
        row_flat(data, pos, nl)
    }
}

struct RowLen;
impl Row for RowLen {
    #[inline(always)]
    fn row(&mut self, data: &[u8], pos: usize, nl: usize) -> u64 {
        row_len(data, pos, nl)
    }
}

struct RowTable<'a>(&'a mut InlineTable);
impl Row for RowTable<'_> {
    #[inline(always)]
    fn row(&mut self, data: &[u8], pos: usize, nl: usize) -> u64 {
        row_table(data, self.0, pos, nl)
    }
}

/// Wraps one mode's whole loop in a function of its own.
///
/// `run_chunk` dispatches forty modes, and inlined into it the drivers were competing for a
/// single budget that LLVM spent in source order: early modes ended up with an out-of-line
/// `bl` to `InlineTable::upsert`, later ones had it inlined. A mode was therefore faster or
/// slower partly by where it sat in the `match` — exactly the artefact this harness exists to
/// rule out. Being generic over `f`, this gets one instantiation per call site, so every mode
/// starts with a full budget; the extra call costs one branch per chunk.
///
/// The drivers themselves stay `#[inline(always)]`, because the row body has to fuse *into*
/// the loop. Marking the drivers `#[inline(never)]` instead looks equivalent and is not: the
/// row body then fails to inline into the driver and the call lands back on every row.
///
/// The row bodies reach the drivers as [`Step`]/[`Row`]/[`Pair`] impls rather than closures,
/// for the same reason. A closure passed as `impl Fn` is one opaque callee that LLVM may or
/// may not inline depending on remaining budget; a zero-sized struct with an
/// `#[inline(always)]` trait method is not optional.
#[inline(never)]
fn once<R>(f: impl FnOnce() -> R) -> R {
    f()
}

// The drivers below keep their accumulators in locals for the same reason: passed as `&mut`
// across a call the compiler cannot see through, both were reloaded and stored back per row.

/// As [`streamed`], but each stream upserts into its own table, so no two streams ever write
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
    let (mut a, mut rows) = (*acc, *n);
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
            a ^= h;
            s[k].0 = next;
        }
        rows += N as u64;
    }
    for k in 0..N {
        while s[k].0 < s[k].1 {
            let (h, next) = full(data, &mut tables[k], s[k].0);
            a ^= h;
            s[k].0 = next;
            rows += 1;
        }
    }
    (*acc, *n) = (a, rows);
}

/// Row loop driven by the newline bitmap instead of by the previous row's parse.
///
/// Every version so far learns where row `i+1` begins only after row `i`'s temperature has
/// been parsed. [`split_streams`] hides that by interleaving three chains; this removes it.
/// One `block_newlines` call yields every row boundary in 64 bytes up front, so the rows in a
/// block are independent by construction and the out-of-order window can overlap them.
///
/// Blocks are aligned down from `start` and the bits below it masked off, so the mask always
/// covers whole 64-byte lines of the input. The final block is truncated at `end`, whose
/// newline the next chunk must not see twice. A file with no trailing newline leaves a row
/// with no bit at all, which the tail check after the loop picks up.
#[inline(always)]
fn drain_newlines(
    data: &[u8],
    start: usize,
    end: usize,
    acc: &mut u64,
    n: &mut u64,
    mut r: impl Row,
) {
    let (mut a, mut rows) = (*acc, *n);
    let mut base = start & !(BLOCK - 1);
    let mut pos = start;
    // Newlines before `start` closed rows the previous chunk owns.
    let mut keep = !0u64 << (start - base);

    while base < end {
        let mut m = unsafe { block_newlines(data.as_ptr().add(base)) } & keep;
        keep = !0;
        if base + BLOCK > end {
            // 1 ..= 63 here, so the shift is in range.
            m &= (1u64 << (end - base)) - 1;
        }
        while m != 0 {
            let nl = base + m.trailing_zeros() as usize;
            m &= m - 1;
            a ^= r.row(data, pos, nl);
            pos = nl + 1;
            rows += 1;
        }
        base += BLOCK;
    }

    // Only fires on a final chunk whose file has no trailing newline.
    if pos < end {
        a ^= r.row(data, pos, end);
        rows += 1;
    }
    (*acc, *n) = (a, rows);
}

/// Scan + hash + parse for a row whose end is already known. The parser still finds the line
/// end itself, so the assert is a free cross-check that the mask agrees with it.
#[inline(always)]
fn row_flat(data: &[u8], pos: usize, nl: usize) -> u64 {
    let (semi, h) = find_semi_and_hash_flat_raw(data, pos);
    let (v, next) = parse_temp_branchless(data, semi + 1);
    debug_assert_eq!(next, nl + 1, "bitmap and parser disagree on where the row ends");
    h ^ v as u64
}

/// [`row_flat`] with the parser's dot search replaced by the length the mask already gave us.
#[inline(always)]
fn row_len(data: &[u8], pos: usize, nl: usize) -> u64 {
    let (semi, h) = find_semi_and_hash_flat_raw(data, pos);
    let v = parse_temp_len(data, semi + 1, nl - semi - 1);
    h ^ v as u64
}

/// [`row_len`] plus the inline-key upsert — v10's engine with the bitmap driving the rows.
#[inline(always)]
fn row_table(data: &[u8], table: &mut InlineTable, pos: usize, nl: usize) -> u64 {
    let (semi, h) = find_semi_and_hash_flat_raw(data, pos);
    let v = parse_temp_len(data, semi + 1, nl - semi - 1);
    table.upsert(data, pos, semi - pos, h, v);
    h
}

/// Both delimiters come from the bitmap, so nothing scans the row: the hash is driven by the
/// name length and the temperature by the value length.
#[inline(always)]
fn pair_rows(data: &[u8], pos: usize, semi: usize, nl: usize) -> u64 {
    let h = hash_len_raw(data, pos, semi - pos);
    let v = parse_temp_len(data, semi + 1, nl - semi - 1);
    h ^ v as u64
}

/// [`pair_rows`] plus the inline-key upsert.
#[inline(always)]
fn pair_table(data: &[u8], table: &mut InlineTable, pos: usize, semi: usize, nl: usize) -> u64 {
    let h = hash_len_raw(data, pos, semi - pos);
    let v = parse_temp_len(data, semi + 1, nl - semi - 1);
    table.upsert(data, pos, semi - pos, h, v);
    h
}

/// [`pair_table`] with the key dropped, to price it.
#[inline(always)]
fn pair_nokey(data: &[u8], table: &mut InlineTable, pos: usize, semi: usize, nl: usize) -> u64 {
    let h = hash_len_raw(data, pos, semi - pos);
    let v = parse_temp_len(data, semi + 1, nl - semi - 1);
    table.upsert_no_key(data, pos, semi - pos, h, v);
    h
}

/// [`pair_table`] with the accumulate dropped, to price it.
#[inline(always)]
fn pair_nostats(data: &[u8], table: &mut InlineTable, pos: usize, semi: usize, nl: usize) -> u64 {
    let h = hash_len_raw(data, pos, semi - pos);
    let v = parse_temp_len(data, semi + 1, nl - semi - 1);
    table.upsert_no_stats(data, pos, semi - pos, h, v);
    h
}

struct PairRows;
impl Pair for PairRows {
    #[inline(always)]
    fn pair(&mut self, data: &[u8], pos: usize, semi: usize, nl: usize) -> u64 {
        pair_rows(data, pos, semi, nl)
    }
}

struct PairTable<'a>(&'a mut InlineTable);
impl Pair for PairTable<'_> {
    #[inline(always)]
    fn pair(&mut self, data: &[u8], pos: usize, semi: usize, nl: usize) -> u64 {
        pair_table(data, self.0, pos, semi, nl)
    }
}

struct PairNokey<'a>(&'a mut InlineTable);
impl Pair for PairNokey<'_> {
    #[inline(always)]
    fn pair(&mut self, data: &[u8], pos: usize, semi: usize, nl: usize) -> u64 {
        pair_nokey(data, self.0, pos, semi, nl)
    }
}

struct PairNostats<'a>(&'a mut InlineTable);
impl Pair for PairNostats<'_> {
    #[inline(always)]
    fn pair(&mut self, data: &[u8], pos: usize, semi: usize, nl: usize) -> u64 {
        pair_nostats(data, self.0, pos, semi, nl)
    }
}

/// [`drain_newlines`] driven by both bitmaps, so no row is scanned at all.
///
/// Names hold no `;` and temperatures hold no `\n`, so the two delimiters strictly alternate
/// and the k-th `;` in a chunk belongs to the k-th `\n`. Pairing is therefore just popping one
/// bit from each mask, with one exception: a row's `;` can land in one block and its `\n` in
/// the next, which leaves a `;` bit with no partner. At most one can ever be outstanding —
/// the `\n` follows its `;` within six bytes, and the next `;` follows that `\n` — so a single
/// carried position covers it.
#[inline(always)]
fn drain_delims(
    data: &[u8],
    start: usize,
    end: usize,
    acc: &mut u64,
    n: &mut u64,
    mut r: impl Pair,
) {
    let (mut a, mut rows) = (*acc, *n);
    let mut base = start & !(BLOCK - 1);
    let mut pos = start;
    let mut keep = !0u64 << (start - base);
    // One past the carried `;`, or 0 for none. Offsetting by one keeps position 0 — a chunk
    // starting on an empty name — from reading as "nothing carried".
    let mut carry = 0usize;

    while base < end {
        let (mut sm, mut nm) = unsafe { block_delims(data.as_ptr().add(base)) };
        sm &= keep;
        nm &= keep;
        keep = !0;
        if base + BLOCK > end {
            let last = (1u64 << (end - base)) - 1;
            sm &= last;
            nm &= last;
        }

        while nm != 0 {
            let nl = base + nm.trailing_zeros() as usize;
            nm &= nm - 1;

            // Selected without branching: the carry fires on about one row in six, often
            // enough to cost mispredictions and too rarely for any history to predict.
            let low = sm & sm.wrapping_neg();
            let taken = 0u64.wrapping_sub((carry == 0) as u64);
            let semi = if carry != 0 { carry - 1 } else { base + low.trailing_zeros() as usize };
            sm ^= low & taken;
            carry = 0;

            a ^= r.pair(data, pos, semi, nl);
            pos = nl + 1;
            rows += 1;
        }

        debug_assert!(sm.count_ones() <= 1, "two unpaired `;` at a block boundary");
        if sm != 0 {
            carry = base + sm.trailing_zeros() as usize + 1;
        }
        base += BLOCK;
    }

    // Only fires on a final chunk whose file has no trailing newline.
    if pos < end {
        let semi = if carry != 0 { carry - 1 } else { find_semi(data, pos) };
        a ^= r.pair(data, pos, semi, end);
        rows += 1;
    }
    (*acc, *n) = (a, rows);
}

/// Runs `step` over `N` independent line streams covering `[start, end)`.
#[inline(always)]
fn streamed<const N: usize>(
    data: &[u8],
    start: usize,
    end: usize,
    acc: &mut u64,
    n: &mut u64,
    mut step: impl Step,
) {
    let (mut a, mut rows) = (*acc, *n);
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
            let (v, next) = step.step(data, s[k].0);
            a ^= v;
            s[k].0 = next;
        }
        rows += N as u64;
    }
    for k in 0..N {
        while s[k].0 < s[k].1 {
            let (v, next) = step.step(data, s[k].0);
            a ^= v;
            s[k].0 = next;
            rows += 1;
        }
    }
    (*acc, *n) = (a, rows);
}

/// [`streamed`] with one stream — the serial dependency chain every version up to v10 walks.
#[inline(always)]
fn serial(data: &[u8], start: usize, end: usize, acc: &mut u64, n: &mut u64, mut step: impl Step) {
    let (mut a, mut rows) = (*acc, *n);
    let mut pos = start;
    while pos < end {
        let (v, next) = step.step(data, pos);
        a ^= v;
        pos = next;
        rows += 1;
    }
    (*acc, *n) = (a, rows);
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
            if ["itable", "flat", "raw", "keyed", "win", "hdr", "hot", "nltable", "dltable",
                "dlnokey", "dlnostats"]
                .iter()
                .any(|p| mode.starts_with(p))
            {
                bits
            } else {
                0
            };
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
        "parse" => once(|| serial(data, pos, end, acc, n, ParseOnly)),
        "hash" => once(|| serial(data, pos, end, acc, n, HashOnly)),
        "table" => once(|| serial(data, pos, end, acc, n, Full(table))),
        "itable" => once(|| serial(data, pos, end, acc, n, FullInline(itable))),
        "fhash" => once(|| serial(data, pos, end, acc, n, FhashOnly)),
        "flat" => once(|| serial(data, pos, end, acc, n, FullFlat(itable))),
        "raw" => once(|| serial(data, pos, end, acc, n, FullRaw(itable))),
        "keyed" => once(|| serial(data, pos, end, acc, n, FullKeyed(itable))),
        "nocmp" => once(|| serial(data, pos, end, acc, n, Nocmp(table))),
        // Delimiter bitmaps only, no rows drained. Everything a block-at-a-time design would
        // do is additive on top of this, so if `mask` alone is not far below `flat3` the
        // approach is dead before any of it is written.
        "mask" => {
            while pos < end {
                let (s, nl) = unsafe { block_delims(data.as_ptr().add(pos)) };
                *acc ^= s ^ nl.rotate_left(1);
                pos += BLOCK;
            }
        }
        "mask_semi" => {
            while pos < end {
                *acc ^= unsafe { block_semis(data.as_ptr().add(pos)) };
                pos += BLOCK;
            }
        }
        "mask_nl" => {
            while pos < end {
                *acc ^= unsafe { block_newlines(data.as_ptr().add(pos)) };
                pos += BLOCK;
            }
        }
        // Rows driven by the newline bitmap. `nlrows` keeps v10's per-row work exactly, so
        // against `raw` it prices the loop shape alone; `nlparse` then spends the newline the
        // mask already produced, and `nltable` adds the upsert back to make it v10's engine.
        "nlrows" => once(|| drain_newlines(data, pos, end, acc, n, RowFlat)),
        "nlparse" => once(|| drain_newlines(data, pos, end, acc, n, RowLen)),
        "nltable" => once(|| drain_newlines(data, pos, end, acc, n, RowTable(itable))),
        // Both delimiters from the bitmap: `dlrows` drops the flat scan that `nlrows` still
        // runs, and `dltable` is the whole of v10's per-row work with no scanning left in it.
        "dlrows" => once(|| drain_delims(data, pos, end, acc, n, PairRows)),
        "dltable" => once(|| drain_delims(data, pos, end, acc, n, PairTable(itable))),
        // The upsert's two halves, priced against `dltable` on the same driver.
        "dlnokey" => once(|| drain_delims(data, pos, end, acc, n, PairNokey(itable))),
        "dlnostats" => once(|| drain_delims(data, pos, end, acc, n, PairNostats(itable))),
        "parse2" => once(|| streamed::<2>(data, pos, end, acc, n, ParseOnly)),
        "parse3" => once(|| streamed::<3>(data, pos, end, acc, n, ParseOnly)),
        "parse4" => once(|| streamed::<4>(data, pos, end, acc, n, ParseOnly)),
        "parse8" => once(|| streamed::<8>(data, pos, end, acc, n, ParseOnly)),
        "table2" => once(|| streamed::<2>(data, pos, end, acc, n, Full(table))),
        "table3" => once(|| streamed::<3>(data, pos, end, acc, n, Full(table))),
        "table4" => once(|| streamed::<4>(data, pos, end, acc, n, Full(table))),
        "table8" => once(|| streamed::<8>(data, pos, end, acc, n, Full(table))),
        "itable2" => once(|| streamed::<2>(data, pos, end, acc, n, FullInline(itable))),
        "itable3" => once(|| streamed::<3>(data, pos, end, acc, n, FullInline(itable))),
        "itable4" => once(|| streamed::<4>(data, pos, end, acc, n, FullInline(itable))),
        "itable8" => once(|| streamed::<8>(data, pos, end, acc, n, FullInline(itable))),
        "fhash2" => once(|| streamed::<2>(data, pos, end, acc, n, FhashOnly)),
        "fhash4" => once(|| streamed::<4>(data, pos, end, acc, n, FhashOnly)),
        "flat2" => once(|| streamed::<2>(data, pos, end, acc, n, FullFlat(itable))),
        "flat3" => once(|| streamed::<3>(data, pos, end, acc, n, FullFlat(itable))),
        "flat4" => once(|| streamed::<4>(data, pos, end, acc, n, FullFlat(itable))),
        "raw2" => once(|| streamed::<2>(data, pos, end, acc, n, FullRaw(itable))),
        "raw3" => once(|| streamed::<3>(data, pos, end, acc, n, FullRaw(itable))),
        "raw4" => once(|| streamed::<4>(data, pos, end, acc, n, FullRaw(itable))),
        // v11 freed the registers the 32-byte probe key used to spill, so the stream count
        // that lost for v7 is worth asking again.
        "keyed2" => once(|| streamed::<2>(data, pos, end, acc, n, FullKeyed(itable))),
        "keyed3" => once(|| streamed::<3>(data, pos, end, acc, n, FullKeyed(itable))),
        "keyed4" => once(|| streamed::<4>(data, pos, end, acc, n, FullKeyed(itable))),
        "keyed5" => once(|| streamed::<5>(data, pos, end, acc, n, FullKeyed(itable))),
        "keyed6" => once(|| streamed::<6>(data, pos, end, acc, n, FullKeyed(itable))),
        "keyed8" => once(|| streamed::<8>(data, pos, end, acc, n, FullKeyed(itable))),
        // The two halves of v13, and both together. Every one of these must leave `acc` and
        // `n` equal to `keyed2`'s — they change how the bytes are reached, not what is read.
        "win2" => once(|| streamed::<2>(data, pos, end, acc, n, FullKeyedWin(itable))),
        "win3" => once(|| streamed::<3>(data, pos, end, acc, n, FullKeyedWin(itable))),
        "hdr2" => once(|| streamed_hot::<2, false>(data, pos, end, acc, n, itable)),
        "hdr3" => once(|| streamed_hot::<3, false>(data, pos, end, acc, n, itable)),
        "hot1" => once(|| streamed_hot::<1, true>(data, pos, end, acc, n, itable)),
        "hot2" => once(|| streamed_hot::<2, true>(data, pos, end, acc, n, itable)),
        "hot3" => once(|| streamed_hot::<3, true>(data, pos, end, acc, n, itable)),
        "hot4" => once(|| streamed_hot::<4, true>(data, pos, end, acc, n, itable)),
        "split2" => once(|| streamed_split::<2>(data, pos, end, acc, n, split)),
        "split4" => once(|| streamed_split::<4>(data, pos, end, acc, n, split)),
        "split8" => once(|| streamed_split::<8>(data, pos, end, acc, n, split)),
        other => panic!("unknown mode {other:?}"),
    }
}

/// Walks the mapping directly. The last [`TAIL_GUARD`] bytes are skipped, because the wide
/// paths would read off the end of it.
fn worker_mmap(
    mode: &str,
    bits: u32,
    data: &[u8],
    chunk: usize,
    total: usize,
    cursor: &AtomicUsize,
) -> State {
    set_thread_qos_user_interactive();
    let mut st = State::new(mode, bits);
    let wide_limit = data.len().saturating_sub(TAIL_GUARD);

    loop {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        let Some((start, chunk_end)) = chunk_bounds(data, i, chunk) else {
            continue;
        };
        let end = chunk_end.min(wide_limit).max(start);
        run_chunk(mode, data, start, end, &mut st);
    }
    st
}

/// Faults every page of `data` in, on `threads` threads, and returns the seconds it took.
///
/// One read per 16 KB page installs the PTE; the byte is discarded, so the read has to be
/// `volatile` or it is not a read at all. Threads take contiguous spans rather than striding
/// through each other's pages, which is both what a real pre-faulter would do and what keeps
/// the kernel's own clustering working with us.
fn prefault(data: &[u8], threads: usize) -> f64 {
    const PAGE: usize = 16384;
    let t0 = Instant::now();
    let span = data.len().div_ceil(threads.max(1));
    std::thread::scope(|scope| {
        for t in 0..threads {
            let lo = (t * span).min(data.len());
            let part = &data[lo..(lo + span).min(data.len())];
            scope.spawn(move || {
                set_thread_qos_user_interactive();
                let mut p = 0;
                while p < part.len() {
                    unsafe { core::ptr::read_volatile(part.as_ptr().add(p)) };
                    p += PAGE;
                }
            });
        }
    });
    t0.elapsed().as_secs_f64()
}

/// v8's reader: one recycled buffer per thread, `pread` per chunk. The buffer's slack past
/// EOF is what lets the wide paths cover the whole file with no scalar tail.
fn worker_pread(
    mode: &str,
    bits: u32,
    file: &File,
    len: usize,
    chunk: usize,
    total: usize,
    cursor: &AtomicUsize,
) -> State {
    set_thread_qos_user_interactive();
    let mut st = State::new(mode, bits);
    let mut buf = vec![0u8; chunk + OVERLAP + TAIL_GUARD];

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
        file.read_exact_at(&mut buf[..avail], base as u64).expect("pread");

        let Some((start, end)) = local_bounds(&buf[..avail], base, chunk) else {
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
    let chunk = env::var("OBRC_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > MAX_LINE_LEN)
        .unwrap_or(CHUNK_SIZE);
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
    let total = num_chunks(len, chunk);
    let cursor = AtomicUsize::new(0);

    let map = || Mapping::open(&path).unwrap_or_else(|e| panic!("cannot map {path}: {e}"));

    // Deliberately ahead of the clock, and only for `mmap_warm`: the point of that reader is to
    // price the parse loop over pages that are already resident, so the faults have to happen
    // where they are not counted. The prefault cost is reported separately and is real work that
    // any shipped version would still owe.
    let warm = (reader == "mmap_warm").then(map);
    if let Some(m) = &warm {
        eprintln!("  prefault {:.3} s (not timed)", prefault(m.as_slice(), threads));
    }

    let start = Instant::now();
    let states: Vec<State> = match reader.as_str() {
        "mmap" | "mmap_warm" => {
            // `mmap` maps inside the timed region, as a real version would have to.
            let cold = warm.is_none().then(map);
            let data = warm.as_ref().or(cold.as_ref()).expect("mapped either side").as_slice();
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| scope.spawn(|| worker_mmap(&mode, bits, data, chunk, total, &cursor)))
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        }
        "pread" => std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| worker_pread(&mode, bits, &file, len, chunk, total, &cursor))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        }),
        other => panic!("unknown reader {other:?}, expected pread, mmap or mmap_warm"),
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
