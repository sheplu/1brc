//! Open-addressed table that stores the station name *inside* the entry.
//!
//! [`crate::table::Table`] keeps only an offset into the input and compares keys by reading
//! the name back out of the mapping. That read is a second, scattered memory access on
//! every row, and the ablation harness attributes essentially the entire cost of the table
//! to it: probing and updating a slot is free (within noise), while adding the comparison
//! costs ~0.27 s over a billion rows.
//!
//! Here the first 32 bytes of the name live in the entry itself, on the cache line the
//! probe already loaded, so the common case touches no other memory. 32 bytes covers every
//! name in the official list (longest is 26) but the spec allows 100, so longer names fall
//! back to comparing the remainder out of the input — correct, just slower, and never
//! exercised by this dataset.

use crate::output::Stats;

/// `MASK[n]` keeps the low `n` bytes. Indexed 0..=16; 272 bytes, permanently L1-resident.
static MASK: [u128; 17] = {
    let mut m = [0u128; 17];
    let mut n = 1;
    while n <= 16 {
        m[n] = u128::MAX >> (128 - 8 * n);
        n += 1;
    }
    m
};

/// Bytes of the name held inline. Must be a multiple of 16.
pub const INLINE_KEY: usize = 32;

/// Readable bytes the probe requires from the start of a name.
pub const KEY_SLACK: usize = INLINE_KEY;

/// 64 bytes, 64-byte aligned — *not* a cache line. This machine's line is 128 bytes, so two
/// entries always share one. That is harmless here and mildly useful: the table is per-thread,
/// so there is no false sharing, and a linear probe's next slot arrives with the current one.
#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct Entry {
    /// The name's first [`INLINE_KEY`] bytes, zero beyond `len`. All-zero means the slot is
    /// empty — see [`upsert`](InlineTable::upsert) for why no real name can look like that.
    key: [u128; 2],
    /// `sum`..`max` are the only fields a repeat row touches, so they are kept adjacent. The
    /// accumulate is still four loads and two stores: `sum` and `count` differ in width, so
    /// nothing pairs. Widening `count` to `u64` to make them pair was tried — LLVM answered
    /// with one `ldr q`/`str q` instead of an `ldp`/`stp`, halving the memory ops for the same
    /// instruction count, and the clock moved 6 ms of 402. Not worth truncating the stored
    /// hash to pay for it.
    sum: i64,
    count: u32,
    min: i16,
    max: i16,
    /// Where the owned copy of the name starts in the table's key arena.
    key_off: u32,
    len: u32,
    /// Only [`grow`](InlineTable::grow) reads this, to re-slot the entry without having to
    /// re-hash a name it would first have to reassemble. The other 56 bytes leave exactly
    /// this much padding, so carrying it is free.
    hash: u64,
}

const _: () = assert!(core::mem::size_of::<Entry>() == 64);

/// The name's first [`INLINE_KEY`] bytes, zeroed past `len`.
///
/// Reads [`KEY_SLACK`] bytes from `off` regardless of `len`, so callers must keep that much
/// slack in bounds. Masking rather than branching on the length keeps this off the critical
/// path: both halves are independent of the hash and of each other.
#[inline(always)]
pub fn probe_key(data: &[u8], off: usize, len: usize) -> [u128; 2] {
    let lo = u128::from_le_bytes(data[off..off + 16].try_into().unwrap());
    let hi = u128::from_le_bytes(data[off + 16..off + 32].try_into().unwrap());
    [lo & MASK[len.min(16)], hi & MASK[len.saturating_sub(16).min(16)]]
}

/// [`probe_key`] for a name whose 16-byte window is already masked, as two little-endian
/// words — the shape [`scan_flat_keyed`](crate::swar::scan_flat_keyed) hands back.
///
/// Only correct for `len <= 16`, which is exactly why the high half is zero: `probe_key`
/// would index `MASK[len - 16]`, and that is `MASK[0]`.
#[inline(always)]
pub fn key_from_words(lo: u64, hi: u64) -> [u128; 2] {
    [(lo as u128) | ((hi as u128) << 64), 0]
}

pub struct InlineTable {
    entries: Box<[Entry]>,
    /// Owned copy of every name inserted. The table therefore borrows nothing from the
    /// input, which the `pread` reader needs — it recycles one buffer per chunk, so an
    /// offset into it would dangle as soon as the next chunk was read.
    keys: Vec<u8>,
    /// Bits taken off the *top* of the hash to pick a slot. See [`slot`].
    shift: u32,
    mask: usize,
    used: usize,
}

/// Slot for `hash`, taken from its high bits.
///
/// The low bits of a multiply are barely mixed — bit 0 of a product is just the product of
/// the inputs' bit 0s — so masking the bottom of a multiplicative hash spreads badly unless
/// the hash first avalanches. The top bits do not have that problem, which is what lets
/// [`crate::swar::find_semi_and_hash_flat_raw`] skip its final avalanche entirely.
#[inline(always)]
fn slot(hash: u64, shift: u32) -> usize {
    (hash >> shift) as usize
}

/// Grow once the table is this full. Linear probing degrades sharply past a half-full
/// table, and a doubling is [`cold`](InlineTable::grow) and bounded — from 2^11 slots the
/// spec's 10,000 names cost four of them, and this dataset's 413 cost none.
const MAX_LOAD_NUM: usize = 1;
const MAX_LOAD_DEN: usize = 2;

impl InlineTable {
    /// Allocates `2^bits` slots and grows from there as needed, so `bits` is a floor chosen
    /// for cache behaviour rather than a capacity that has to cover the worst case.
    /// Allocate once per thread and reuse across chunks.
    pub fn new(bits: u32) -> Self {
        // At least two slots, because [`slot`] shifts right by `64 - bits` and a 64-bit
        // shift is undefined. Callers pass 0 to mean "this table is never used", and two
        // entries is a rounding error next to the one they do use.
        let bits = bits.max(1);
        let cap = 1usize << bits;
        InlineTable {
            entries: vec![Entry::default(); cap].into_boxed_slice(),
            keys: Vec::new(),
            shift: 64 - bits,
            mask: cap - 1,
            used: 0,
        }
    }

    /// Doubles the slot count and re-slots every live entry.
    ///
    /// The arena is untouched — `key_off` stays valid, so no name is copied or re-hashed.
    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        let cap = self.entries.len() * 2;
        let (shift, mask) = (self.shift - 1, cap - 1);
        let mut entries = vec![Entry::default(); cap].into_boxed_slice();
        for e in self.entries.iter().filter(|e| e.len != 0) {
            let mut idx = slot(e.hash, shift);
            while entries[idx].len != 0 {
                idx = (idx + 1) & mask;
            }
            entries[idx] = *e;
        }
        self.entries = entries;
        self.shift = shift;
        self.mask = mask;
    }

    /// Accumulates one measurement. `hash` must be derived from `data[off..off + len]`, and
    /// [`KEY_SLACK`] bytes from `off` must be readable.
    ///
    /// A hash match is never taken as a key match: the full name is always compared.
    ///
    /// **Why the length is not compared, and why the slot needs no occupancy flag.** Station
    /// names hold no NUL, so a name of `len < INLINE_KEY` zero-padded to [`INLINE_KEY`] bytes
    /// determines both the bytes and the length: it has a zero at index `len`, where a longer
    /// name has either a name byte or, at `len == INLINE_KEY` and above, no zero at all. The key
    /// alone therefore settles the match for every name this dataset contains — the longest is
    /// 26 — and the same argument makes an all-zero key impossible for a live entry, so `Default`
    /// marks an empty slot and the hot compare rejects it for free.
    ///
    /// The bound is strict, and that is the whole of it: a name of *exactly* [`INLINE_KEY`] bytes
    /// fills the key with non-zero and so is indistinguishable from the first 32 bytes of any
    /// longer name. It gets the length check and the tail compare like the longer names it can be
    /// confused with — `tail_eq` then compares an empty range, and `e.len` does the work.
    ///
    /// `#[inline(always)]`, with the insert outlined to keep this small enough to deserve it.
    /// Under a plain `#[inline]` the whole body sat right at LLVM's size threshold, so whether
    /// it inlined depended on how much budget the caller had already spent: v8, v9 and v10 all
    /// shipped an out-of-line `bl` here on *every row*, and in `hot_floor` some ablation modes
    /// inlined it and others did not purely by position in the `match`. A call in the middle of
    /// the row body also stops the scan for row `i+1` from overlapping the probe for row `i`,
    /// which is the entire point of `split_streams`.
    #[inline(always)]
    pub fn upsert(&mut self, data: &[u8], off: usize, len: usize, hash: u64, value: i16) {
        debug_assert!(len > 0 && len <= 100);
        let key = probe_key(data, off, len);
        let mut idx = slot(hash, self.shift);

        // Probe read-only so the arena stays borrowable for the long-name comparison.
        loop {
            let e = &self.entries[idx];
            if e.key == key {
                if len < INLINE_KEY
                    || (e.len as usize == len
                        && tail_eq(&self.keys, e.key_off as usize, data, off, len))
                {
                    let e = &mut self.entries[idx];
                    e.sum += value as i64;
                    e.count += 1;
                    // Extending the range is logarithmic in the row count — a few dozen times
                    // per station per thread against 134k rows — so this is a branch that
                    // predicts, and taking it out of line leaves the common row with a plain
                    // load and no store. Writing it as one sign test lets the two bounds be
                    // checked with a single branch.
                    if ((value as i32 - e.min as i32) | (e.max as i32 - value as i32)) < 0 {
                        widen(e, value);
                    }
                    return;
                }
            } else if e.key == [0; 2] {
                break;
            }
            idx = (idx + 1) & self.mask;
        }

        self.insert(data, off, len, hash, value, key, idx);
    }

    /// [`upsert`](Self::upsert) for a name shorter than 16 bytes whose zero-padded window the
    /// caller already holds, as the two words
    /// [`scan_flat_keyed`](crate::swar::scan_flat_keyed) returns.
    ///
    /// Under that length bound the probe collapses twice over. The key needs no rebuilding
    /// from memory — the scan masked those same bytes to keep them out of the hash, so
    /// [`key_from_words`] is a relabelling. And only its *low* half has to be compared: a
    /// name of fewer than 16 bytes has a zero at index `len`, every stored name of 16 or more
    /// has none in its first 16 bytes, and an empty slot is all zeros — so the low half alone
    /// separates the query from every entry the table can hold, live or empty, at any length.
    /// The high half, the length, and the long-name tail compare all drop out, and with them
    /// the second `ldp` off the entry's cache line.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_words(
        &mut self,
        data: &[u8],
        off: usize,
        len: usize,
        hash: u64,
        value: i16,
        klo: u64,
        khi: u64,
    ) {
        debug_assert!(len > 0 && len < 16);
        debug_assert_eq!(key_from_words(klo, khi), probe_key(data, off, len));
        let key_lo = (klo as u128) | ((khi as u128) << 64);
        let mut idx = slot(hash, self.shift);

        loop {
            let e = &self.entries[idx];
            if e.key[0] == key_lo {
                let e = &mut self.entries[idx];
                e.sum += value as i64;
                e.count += 1;
                if ((value as i32 - e.min as i32) | (e.max as i32 - value as i32)) < 0 {
                    widen(e, value);
                }
                return;
            }
            if e.key[0] == 0 {
                break;
            }
            idx = (idx + 1) & self.mask;
        }

        // Rebuilt rather than handed over: `insert` takes the key by value, so passing it
        // costs a 32-byte stack write that the hot path executed and then never read.
        self.insert(data, off, len, hash, value, [key_lo, 0], idx);
    }

    /// The slots and the shift, borrowed, for a caller that wants to keep them in registers.
    ///
    /// [`upsert_words`](Self::upsert_words) cannot: `&mut self` stays live across `insert`,
    /// which may [`grow`](Self::grow) and replace `entries`, so every row reloads `shift`,
    /// `entries.ptr`, `entries.len` and `mask` from the frame — four stack loads, one of them
    /// on the address-generation chain that ends at the entry's cache line.
    ///
    /// Handing out the slice moves that proof to the borrow checker. While the borrow is live
    /// nothing can reallocate, so the four values are loop-invariant and hoist out. The caller
    /// pays for it by having to drop the borrow to insert, which is what
    /// [`probe_words`] returning `false` is for.
    #[inline(always)]
    pub fn hot_slots(&mut self) -> (&mut [Entry], u32) {
        (&mut self.entries, self.shift)
    }

    /// [`upsert`](Self::upsert) with the key dropped entirely — no [`probe_key`], no compare,
    /// slots matched on length alone. Ablation only: distinct names of equal length merge, so
    /// the results are wrong. Prices everything the key costs, the two masked loads included.
    ///
    /// With no key there is no all-zero-key sentinel either, so occupancy falls back to `len`.
    /// The accumulate is `upsert`'s, `widen` included — otherwise the difference between the
    /// two would price the store-free common path as well as the key.
    #[inline(always)]
    pub fn upsert_no_key(&mut self, data: &[u8], off: usize, len: usize, hash: u64, value: i16) {
        let mut idx = slot(hash, self.shift);
        loop {
            let e = &mut self.entries[idx];
            if e.len == 0 {
                break;
            }
            if e.len as usize == len {
                e.sum += value as i64;
                e.count += 1;
                if ((value as i32 - e.min as i32) | (e.max as i32 - value as i32)) < 0 {
                    widen(e, value);
                }
                return;
            }
            idx = (idx + 1) & self.mask;
        }
        self.insert(data, off, len, hash, value, [0; 2], idx);
    }

    /// [`upsert`](Self::upsert) with the four-field accumulate dropped. Ablation only: it probes
    /// and compares exactly as the real one does, then throws the measurement away, so it prices
    /// the read-modify-write against a slot the probe has already pulled into L1.
    ///
    /// "Exactly as the real one does" is load-bearing and easy to lose: this body has to track
    /// `upsert`'s probe line for line, or the difference between them prices the probe too.
    #[inline(always)]
    pub fn upsert_no_stats(&mut self, data: &[u8], off: usize, len: usize, hash: u64, value: i16) {
        let key = probe_key(data, off, len);
        let mut idx = slot(hash, self.shift);
        loop {
            let e = &self.entries[idx];
            if e.key == key {
                if len < INLINE_KEY
                    || (e.len as usize == len
                        && tail_eq(&self.keys, e.key_off as usize, data, off, len))
                {
                    return;
                }
            } else if e.key == [0; 2] {
                break;
            }
            idx = (idx + 1) & self.mask;
        }
        self.insert(data, off, len, hash, value, key, idx);
    }

    /// Claims the empty slot the probe stopped on. At most once per distinct station, so it is
    /// outlined: its cost is irrelevant and its size is not.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn insert(
        &mut self,
        data: &[u8],
        off: usize,
        len: usize,
        hash: u64,
        value: i16,
        key: [u128; 2],
        idx: usize,
    ) {
        let key_off = self.keys.len() as u32;
        self.keys.extend_from_slice(&data[off..off + len]);
        self.entries[idx] = Entry {
            key,
            sum: value as i64,
            key_off,
            count: 1,
            len: len as u32,
            min: value,
            max: value,
            hash,
        };
        self.used += 1;
        // Growing here rather than before the probe leaves the table at most half full on
        // entry, so the probe loop always meets an empty slot and always terminates.
        if self.used * MAX_LOAD_DEN > self.entries.len() * MAX_LOAD_NUM {
            self.grow();
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[u8], Stats)> + '_ {
        self.entries.iter().filter(|e| e.len != 0).map(move |e| {
            let start = e.key_off as usize;
            (
                &self.keys[start..start + e.len as usize],
                Stats { min: e.min, max: e.max, sum: e.sum, count: e.count },
            )
        })
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }
}

/// [`upsert_words`](InlineTable::upsert_words) against slots borrowed out of the table with
/// [`hot_slots`](InlineTable::hot_slots).
///
/// `false` means the probe reached an empty slot, and **nothing was written** — the caller drops
/// the borrow and repeats the row through `upsert_words`, which owns the table and may insert
/// and grow. At most once per distinct station, so the repeat is not a cost.
///
/// The mask is derived from `slots.len()` rather than passed in, and that is the point: `x & m`
/// where `m == len - 1` is provably `< len`, so neither the initial slot nor the probe's
/// back-edge needs a bounds check. Two unrelated arguments would prove nothing, which is why the
/// `self.mask` field cannot do this. [`grow`](InlineTable::grow) already demonstrates the shape —
/// it probes with a local `mask = cap - 1` against a slice of length `cap` and compiles without a
/// check, while `upsert_words`, reading the same value from a field, gets one.
///
/// Masking the initial slot is free: [`slot`] shifts by `64 - bits` and the table holds
/// `1 << bits` slots, so the result is already in range and the `and` only makes that visible.
///
/// A free function rather than a method, so that adding it leaves the mangled names of every
/// existing method — and therefore the disassembly v1..v12 are compared against — untouched.
#[inline(always)]
pub fn probe_words(slots: &mut [Entry], shift: u32, hash: u64, key_lo: u128, value: i16) -> bool {
    debug_assert!(slots.len().is_power_of_two());
    let mask = slots.len() - 1;
    let mut idx = slot(hash, shift) & mask;

    loop {
        let e = &slots[idx];
        if e.key[0] == key_lo {
            let e = &mut slots[idx];
            e.sum += value as i64;
            e.count += 1;
            if ((value as i32 - e.min as i32) | (e.max as i32 - value as i32)) < 0 {
                widen(e, value);
            }
            return true;
        }
        if e.key[0] == 0 {
            return false;
        }
        idx = (idx + 1) & mask;
    }
}

/// Extends an entry's range. Out of line so the common row pays no store; see
/// [`upsert`](InlineTable::upsert).
#[cold]
#[inline(never)]
fn widen(e: &mut Entry, value: i16) {
    e.min = e.min.min(value);
    e.max = e.max.max(value);
}

/// Compares the part of a name that does not fit inline: the stored copy in `keys` against
/// the probe in `data`. Cold — no name in the official list is long enough to reach it.
///
/// Called at `len == INLINE_KEY` too, where the range is empty and it trivially agrees; there
/// the caller's `e.len` check is what separates the name from the longer ones it shares a key
/// with.
#[cold]
#[inline(never)]
fn tail_eq(keys: &[u8], stored: usize, data: &[u8], off: usize, len: usize) -> bool {
    keys[stored + INLINE_KEY..stored + len] == data[off + INLINE_KEY..off + len]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::fx_hash;
    use std::collections::BTreeMap;

    /// Lays out the records as input rows, with [`KEY_SLACK`] of readable slack past the last
    /// one, as the workers guarantee. Returns the bytes and one `(offset, len)` span per record.
    fn layout(records: &[(&str, i16)]) -> (Vec<u8>, Vec<(usize, usize)>) {
        let mut data = Vec::new();
        let mut spans = Vec::new();
        for (name, _) in records {
            spans.push((data.len(), name.len()));
            data.extend_from_slice(name.as_bytes());
            data.extend_from_slice(b";12.3\n");
        }
        data.resize(data.len() + KEY_SLACK, 0xAA);
        (data, spans)
    }

    fn run(records: &[(&str, i16)], bits: u32) -> InlineTable {
        let (data, spans) = layout(records);

        let mut t = InlineTable::new(bits);
        for (i, (name, value)) in records.iter().enumerate() {
            let (off, len) = spans[i];
            t.upsert(&data, off, len, fx_hash(name.as_bytes()), *value);
        }
        t
    }

    fn collect(t: &InlineTable) -> BTreeMap<String, (i16, i16, i64, u32)> {
        t.iter()
            .map(|(k, s)| (String::from_utf8(k.to_vec()).unwrap(), (s.min, s.max, s.sum, s.count)))
            .collect()
    }

    #[test]
    fn accumulates_per_key() {
        let t = run(&[("Abha", 10), ("Oslo", -50), ("Abha", 30), ("Abha", -5)], 8);
        let got = collect(&t);
        assert_eq!(got["Abha"], (-5, 30, 35, 3));
        assert_eq!(got["Oslo"], (-50, -50, -50, 1));
        assert_eq!(t.len(), 2);
    }

    /// The inline key is masked to `len`; whatever follows the name in the input must not
    /// leak into it, or the same station would look different from row to row.
    #[test]
    fn trailing_bytes_do_not_leak_into_the_key() {
        let mut data = Vec::new();
        let mut spans = Vec::new();
        for suffix in [";1.0\n", ";-99.9\n", ";0.0\n", ";12.3\n"] {
            spans.push(data.len());
            data.extend_from_slice(b"Abha");
            data.extend_from_slice(suffix.as_bytes());
        }
        data.resize(data.len() + KEY_SLACK, 0xAA);

        let mut t = InlineTable::new(8);
        for off in spans {
            t.upsert(&data, off, 4, fx_hash(b"Abha"), 1);
        }
        assert_eq!(t.len(), 1, "one station split across several entries");
        assert_eq!(collect(&t)["Abha"], (1, 1, 4, 4));
    }

    #[test]
    fn distinct_keys_survive_forced_collisions() {
        let records = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("a", 10), ("c", 30)];
        let t = run(&records, 3);
        let got = collect(&t);
        assert_eq!(t.len(), 5);
        assert_eq!(got["a"], (1, 10, 11, 2));
        assert_eq!(got["c"], (3, 30, 33, 2));
    }

    /// The probe no longer compares lengths — it relies on a name being recoverable from its
    /// zero-padded key. That holds only while names contain no NUL, and the case it protects is
    /// a name that is a strict prefix of another. Check a whole prefix chain, including the one
    /// that runs across the 32-byte seam where the inline key stops covering the name.
    ///
    /// Both directions. Whether the shorter or the longer name is the *query* decides which
    /// one's length the probe gets to see, so a chain walked only short-to-long tests half the
    /// property — which is how the `len <= INLINE_KEY` off-by-one survived here.
    #[test]
    fn a_name_is_never_confused_with_its_own_prefixes() {
        let base = "Sankt_Peterburg_Oblast_Station_Nord";
        let names: Vec<String> = (1..=base.len()).map(|n| base[..n].to_string()).collect();
        let mut refs: Vec<(&str, i16)> = names.iter().map(|n| (n.as_str(), 1i16)).collect();

        for _ in 0..2 {
            let t = run(&refs, 8);
            assert_eq!(t.len(), names.len(), "prefixes merged into one entry");
            assert!(collect(&t).values().all(|&s| s == (1, 1, 1, 1)));
            refs.reverse();
        }
    }

    /// A name of *exactly* [`INLINE_KEY`] bytes fills the key with non-zero, so it is
    /// indistinguishable from the prefix of any longer name — the one length at which the
    /// zero-padding argument above does not hold. It must therefore take the tail compare like
    /// any other long name.
    ///
    /// Only the order below catches it: inserting the 32-byte name first makes the longer one
    /// the query, and the longer one does compare lengths. Inserting the longer one first makes
    /// the 32-byte name the query, and it is the query's length that decides whether the guard
    /// is skipped. The prefix chain above only ever runs short-to-long.
    #[test]
    fn a_name_that_exactly_fills_the_key_is_not_confused_with_a_longer_one() {
        let short = "Sankt_Peterburg_Oblast_Station_N";
        let long = "Sankt_Peterburg_Oblast_Station_Nord";
        assert_eq!(short.len(), INLINE_KEY);
        assert_eq!(&long[..INLINE_KEY], short);

        // Both orders, and a forced hash collision so the probe cannot avoid the comparison by
        // landing elsewhere — that is the case the key compare exists for.
        for recs in [[(long, 1i16), (short, 2i16)], [(short, 2i16), (long, 1i16)]] {
            let (data, spans) = layout(&recs);
            let mut t = InlineTable::new(8);
            for (i, (name, value)) in recs.iter().enumerate() {
                t.upsert(&data, spans[i].0, name.len(), 0xdead_beef_0000_0000, *value);
            }
            let got = collect(&t);
            assert_eq!(t.len(), 2, "{recs:?} merged into one entry");
            assert_eq!(got[short], (2, 2, 2, 1));
            assert_eq!(got[long], (1, 1, 1, 1));
        }
    }

    /// Exercises the `len > INLINE_KEY` fallback, including names that are identical for
    /// far longer than the inline key and differ only at the very end.
    #[test]
    fn long_names_fall_back_to_the_tail_compare() {
        let a = "x".repeat(99) + "A";
        let b = "x".repeat(99) + "B";
        let c = "x".repeat(50);
        let recs = [(a.as_str(), 1), (b.as_str(), 2), (c.as_str(), 3), (a.as_str(), 9)];
        let t = run(&recs, 8);
        let got = collect(&t);
        assert_eq!(t.len(), 3);
        assert_eq!(got[&a], (1, 9, 10, 2));
        assert_eq!(got[&b], (2, 2, 2, 1));
        assert_eq!(got[&c], (3, 3, 3, 1));
    }

    /// Growth must be invisible. A table started far too small has to end up with exactly
    /// the contents of one that never grew — same stats, same names, same count — after
    /// crossing the doubling boundary many times over.
    #[test]
    fn growth_is_indistinguishable_from_starting_large() {
        let names: Vec<String> = (0..5000).map(|i| format!("station_{i:05}")).collect();
        // Two rows per name, so the second one exercises the *lookup* path in a table that
        // has been re-slotted since the insert.
        let mut refs: Vec<(&str, i16)> = names.iter().map(|n| (n.as_str(), 7i16)).collect();
        refs.extend(names.iter().map(|n| (n.as_str(), -3i16)));

        let grown = run(&refs, 1);
        let preallocated = run(&refs, 14);
        assert_eq!(grown.len(), names.len());
        assert_eq!(collect(&grown), collect(&preallocated));
    }

    /// The arena is not rebuilt by a doubling, so a name too long to live inline has to
    /// still be found through `key_off` afterwards.
    #[test]
    fn growth_preserves_long_names() {
        let names: Vec<String> = (0..300).map(|i| format!("{}{i:03}", "z".repeat(97))).collect();
        let refs: Vec<(&str, i16)> = names.iter().map(|n| (n.as_str(), 1i16)).collect();
        let t = run(&refs, 1);
        assert_eq!(t.len(), 300);
        assert_eq!(collect(&t), collect(&run(&refs, 10)));
    }

    #[test]
    fn agrees_with_the_offset_keyed_table() {
        let mut records: Vec<(String, i16)> =
            crate::stations::STATIONS.iter().map(|(n, _)| (n.to_string(), 1i16)).collect();
        for i in 0..500 {
            records.push((format!("{}{i:03}", "x".repeat(97)), 2));
        }
        for len in 1..=100usize {
            records.push(("y".repeat(len), 3));
        }
        let refs: Vec<(&str, i16)> = records.iter().map(|(n, v)| (n.as_str(), *v)).collect();

        let inline = run(&refs, 12);
        let (data, spans) = layout(&refs);
        let mut plain = crate::table::Table::new(12);
        for (i, (name, value)) in refs.iter().enumerate() {
            plain.upsert(&data, spans[i].0, name.len(), fx_hash(name.as_bytes()), *value);
        }

        let plain_map: BTreeMap<String, (i16, i16, i64, u32)> = plain
            .iter(&data)
            .map(|(k, s)| (String::from_utf8(k.to_vec()).unwrap(), (s.min, s.max, s.sum, s.count)))
            .collect();
        assert_eq!(collect(&inline), plain_map);
        assert_eq!(inline.len(), refs.len());
    }
}
