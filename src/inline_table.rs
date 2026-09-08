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

/// One cache line. Two entries never share a line, so a probe touches exactly one.
#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct Entry {
    /// The name's first [`INLINE_KEY`] bytes, zero beyond `len`.
    key: [u128; 2],
    sum: i64,
    /// Where the owned copy of the name starts in the table's key arena.
    key_off: u32,
    count: u32,
    /// Doubles as the occupancy flag: names are at least 1 byte, so 0 means empty.
    len: u32,
    min: i16,
    max: i16,
}

const _: () = assert!(core::mem::size_of::<Entry>() == 64);

/// The name's first [`INLINE_KEY`] bytes, zeroed past `len`.
///
/// Reads [`KEY_SLACK`] bytes from `off` regardless of `len`, so callers must keep that much
/// slack in bounds. Masking rather than branching on the length keeps this off the critical
/// path: both halves are independent of the hash and of each other.
#[inline(always)]
fn probe_key(data: &[u8], off: usize, len: usize) -> [u128; 2] {
    let lo = u128::from_le_bytes(data[off..off + 16].try_into().unwrap());
    let hi = u128::from_le_bytes(data[off + 16..off + 32].try_into().unwrap());
    [lo & MASK[len.min(16)], hi & MASK[len.saturating_sub(16).min(16)]]
}

pub struct InlineTable {
    entries: Box<[Entry]>,
    /// Owned copy of every name inserted. The table therefore borrows nothing from the
    /// input, which the `pread` reader needs — it recycles one buffer per chunk, so an
    /// offset into it would dangle as soon as the next chunk was read.
    keys: Vec<u8>,
    mask: usize,
    used: usize,
}

impl InlineTable {
    /// Allocates `2^bits` slots. Allocate once per thread and reuse across chunks.
    pub fn new(bits: u32) -> Self {
        let cap = 1usize << bits;
        InlineTable {
            entries: vec![Entry::default(); cap].into_boxed_slice(),
            keys: Vec::new(),
            mask: cap - 1,
            used: 0,
        }
    }

    /// Accumulates one measurement. `hash` must be derived from `data[off..off + len]`, and
    /// [`KEY_SLACK`] bytes from `off` must be readable.
    ///
    /// A hash match is never taken as a key match: the full name is always compared.
    #[inline]
    pub fn upsert(&mut self, data: &[u8], off: usize, len: usize, hash: u64, value: i16) {
        debug_assert!(len > 0 && len <= 100);
        let key = probe_key(data, off, len);
        let mut idx = (hash as usize) & self.mask;

        // Probe read-only so the arena stays borrowable for the long-name comparison.
        loop {
            let e = &self.entries[idx];
            if e.len == 0 {
                break;
            }
            if e.len as usize == len
                && e.key == key
                && (len <= INLINE_KEY || tail_eq(&self.keys, e.key_off as usize, data, off, len))
            {
                let e = &mut self.entries[idx];
                e.sum += value as i64;
                e.count += 1;
                e.min = e.min.min(value);
                e.max = e.max.max(value);
                return;
            }
            idx = (idx + 1) & self.mask;
        }

        // Cold: at most once per distinct station.
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
        };
        self.used += 1;
        // Keep at least one empty slot so probing always terminates.
        let cap = self.entries.len();
        assert!(self.used < cap, "hash table full: more than {} distinct stations", cap - 1);
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

/// Compares the part of a name that does not fit inline: the stored copy in `keys` against
/// the probe in `data`. Cold — no name in the official list is long enough to reach it.
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
