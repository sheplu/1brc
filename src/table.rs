//! Open-addressed hash table specialised for this workload.
//!
//! Keys are not copied — an entry stores the offset and length of the name within the
//! input, so the table holds no heap allocations beyond its own slot array.

use crate::output::Stats;

/// 28 bytes of payload, padded to 32, so exactly two entries share a cache line.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Entry {
    pub sum: i64,
    pub offset: u64,
    pub count: u32,
    /// Doubles as the occupancy flag: names are at least 1 byte, so 0 means empty.
    pub len: u32,
    pub min: i16,
    pub max: i16,
}

const _: () = assert!(core::mem::size_of::<Entry>() == 32);

pub struct Table {
    entries: Box<[Entry]>,
    mask: usize,
    used: usize,
}

#[inline(always)]
fn word(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(data[at..at + 8].try_into().unwrap())
}

/// `data[a..a + len] == data[b..b + len]`, compared in 8-byte words.
///
/// Never reads outside either key: the last word is aligned to the *end* of the key rather
/// than masked, so it re-compares a few bytes instead of reaching past them. That keeps the
/// function usable everywhere `upsert` is, including the final line of the file.
#[inline(always)]
fn keys_eq(data: &[u8], a: usize, b: usize, len: usize) -> bool {
    if len < 8 {
        // No room for a word; at most 7 iterations, and no call.
        return (0..len).all(|i| data[a + i] == data[b + i]);
    }
    let mut i = 0;
    while i + 8 < len {
        if word(data, a + i) != word(data, b + i) {
            return false;
        }
        i += 8;
    }
    word(data, a + len - 8) == word(data, b + len - 8)
}

impl Table {
    /// Allocates `2^bits` slots. Allocate once per thread and reuse across chunks —
    /// re-zeroing this per chunk would cost more than the parsing.
    pub fn new(bits: u32) -> Self {
        let cap = 1usize << bits;
        Table { entries: vec![Entry::default(); cap].into_boxed_slice(), mask: cap - 1, used: 0 }
    }

    /// Accumulates one measurement. `hash` must be derived from `data[off..off + len]`.
    ///
    /// A hash match is never treated as a key match: the full name is compared on every
    /// probe, so distinct stations that collide stay distinct.
    #[inline]
    pub fn upsert(&mut self, data: &[u8], off: usize, len: usize, hash: u64, value: i16) {
        self.upsert_impl::<false>(data, off, len, hash, value)
    }

    /// [`Table::upsert`] with the key comparison done inline in 8-byte words.
    ///
    /// The slice comparison in `upsert` lowers to a `bcmp` call, which costs a branch, a
    /// call and a length dispatch on every single row. This compares the same bytes with
    /// straight-line word loads. Behaviour is identical; only the code generation differs.
    #[inline]
    pub fn upsert_wide_cmp(&mut self, data: &[u8], off: usize, len: usize, hash: u64, value: i16) {
        self.upsert_impl::<true>(data, off, len, hash, value)
    }

    /// **Diagnostic only — this is not a correct upsert.**
    ///
    /// Treats a length match as a key match, skipping the comparison entirely. Used by
    /// `hot_floor` to separate the cost of reaching the slot from the cost of comparing the
    /// key, which lives at a scattered offset in the input rather than in the entry. Any
    /// two same-length stations that collide will be merged, so never use this for output.
    #[inline]
    pub fn upsert_no_key_compare(
        &mut self,
        _data: &[u8],
        off: usize,
        len: usize,
        hash: u64,
        value: i16,
    ) {
        let cap = self.entries.len();
        let mut idx = (hash as usize) & self.mask;
        loop {
            let e = &mut self.entries[idx];
            if e.len == 0 {
                *e = Entry {
                    sum: value as i64,
                    offset: off as u64,
                    count: 1,
                    len: len as u32,
                    min: value,
                    max: value,
                };
                self.used += 1;
                assert!(self.used < cap);
                return;
            }
            if e.len as usize == len {
                e.sum += value as i64;
                e.count += 1;
                e.min = e.min.min(value);
                e.max = e.max.max(value);
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn upsert_impl<const WIDE_CMP: bool>(
        &mut self,
        data: &[u8],
        off: usize,
        len: usize,
        hash: u64,
        value: i16,
    ) {
        debug_assert!(len > 0 && len <= 100);
        let cap = self.entries.len();
        let mut idx = (hash as usize) & self.mask;
        loop {
            let e = &mut self.entries[idx];
            if e.len == 0 {
                *e = Entry {
                    sum: value as i64,
                    offset: off as u64,
                    count: 1,
                    len: len as u32,
                    min: value,
                    max: value,
                };
                self.used += 1;
                // Keep at least one empty slot so probing always terminates.
                assert!(
                    self.used < cap,
                    "hash table full: more than {} distinct stations",
                    cap - 1
                );
                return;
            }
            if e.len as usize == len {
                let start = e.offset as usize;
                let hit = if WIDE_CMP {
                    keys_eq(data, start, off, len)
                } else {
                    data[start..start + len] == data[off..off + len]
                };
                if hit {
                    e.sum += value as i64;
                    e.count += 1;
                    e.min = e.min.min(value);
                    e.max = e.max.max(value);
                    return;
                }
            }
            idx = (idx + 1) & self.mask;
        }
    }

    pub fn iter<'a>(&'a self, data: &'a [u8]) -> impl Iterator<Item = (&'a [u8], Stats)> + 'a {
        self.entries.iter().filter(|e| e.len != 0).map(move |e| {
            let start = e.offset as usize;
            (
                &data[start..start + e.len as usize],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::fx_hash;
    use std::collections::BTreeMap;

    fn collect(t: &Table, data: &[u8]) -> BTreeMap<String, (i16, i16, i64, u32)> {
        t.iter(data)
            .map(|(k, s)| (String::from_utf8(k.to_vec()).unwrap(), (s.min, s.max, s.sum, s.count)))
            .collect()
    }

    /// Build an input of `name;value` records and feed them all through the table.
    ///
    /// Runs both comparison paths and asserts they agree, so every table test below covers
    /// the word-wise compare as well as the slice compare.
    fn run(records: &[(&str, i16)], bits: u32) -> (Vec<u8>, Table) {
        let mut data = Vec::new();
        let mut spans = Vec::new();
        for (name, _) in records {
            spans.push((data.len(), name.len()));
            data.extend_from_slice(name.as_bytes());
        }

        let build = |wide: bool| {
            let mut t = Table::new(bits);
            for (i, (name, value)) in records.iter().enumerate() {
                let (off, len) = spans[i];
                let h = fx_hash(name.as_bytes());
                if wide {
                    t.upsert_wide_cmp(&data, off, len, h, *value);
                } else {
                    t.upsert(&data, off, len, h, *value);
                }
            }
            t
        };

        let slice_cmp = build(false);
        let wide_cmp = build(true);
        assert_eq!(collect(&slice_cmp, &data), collect(&wide_cmp, &data), "compare paths diverged");
        (data, slice_cmp)
    }

    /// The word-wise compare re-reads bytes near the end of the key; make sure a difference
    /// at *any* position is still caught, at every length, and that it never looks past the
    /// key (the two copies are followed by deliberately different bytes).
    #[test]
    fn wide_compare_catches_a_difference_at_every_position() {
        for len in 1..=100usize {
            let mut data = Vec::new();
            let a = data.len();
            data.extend((0..len).map(|i| b'a' + (i % 26) as u8));
            data.extend_from_slice(b"<<<<<<<<");
            let b = data.len();
            data.extend((0..len).map(|i| b'a' + (i % 26) as u8));
            data.extend_from_slice(b">>>>>>>>");
            assert!(keys_eq(&data, a, b, len), "equal keys of length {len}");

            for pos in 0..len {
                let mut d = data.clone();
                d[b + pos] ^= 0x20;
                assert!(!keys_eq(&d, a, b, len), "length {len}, difference at byte {pos}");
            }
        }
    }

    #[test]
    fn accumulates_per_key() {
        let (data, t) = run(&[("Abha", 10), ("Oslo", -50), ("Abha", 30), ("Abha", -5)], 8);
        let got = collect(&t, &data);
        assert_eq!(got["Abha"], (-5, 30, 35, 3));
        assert_eq!(got["Oslo"], (-50, -50, -50, 1));
        assert_eq!(t.len(), 2);
    }

    /// A table with only 8 slots forces heavy probing, so this exercises the collision
    /// path rather than the happy path.
    #[test]
    fn distinct_keys_survive_forced_collisions() {
        let records: Vec<(&str, i16)> =
            vec![("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("a", 10), ("c", 30)];
        let (data, t) = run(&records, 3);
        let got = collect(&t, &data);
        assert_eq!(t.len(), 5);
        assert_eq!(got["a"], (1, 10, 11, 2));
        assert_eq!(got["c"], (3, 30, 33, 2));
        assert_eq!(got["e"], (5, 5, 5, 1));
    }

    /// Keys sharing a long prefix must not be conflated — a prefix-only compare would.
    #[test]
    fn long_shared_prefixes_are_distinguished() {
        let a = "Station_with_a_very_long_shared_prefix_AAAA";
        let b = "Station_with_a_very_long_shared_prefix_BBBB";
        let c = "Station_with_a_very_long_shared_prefix_AAAA_extra";
        let (data, t) = run(&[(a, 1), (b, 2), (c, 3), (a, 5)], 8);
        let got = collect(&t, &data);
        assert_eq!(t.len(), 3);
        assert_eq!(got[a], (1, 5, 6, 2));
        assert_eq!(got[b], (2, 2, 2, 1));
        assert_eq!(got[c], (3, 3, 3, 1));
    }

    #[test]
    fn handles_the_full_station_list_plus_adversarial_names() {
        let mut records: Vec<(String, i16)> =
            crate::stations::STATIONS.iter().map(|(n, _)| (n.to_string(), 1i16)).collect();
        // Names at the 100-byte limit, differing only in the final byte.
        for i in 0..500 {
            let mut n = "x".repeat(97);
            n.push_str(&format!("{i:03}"));
            records.push((n, 2));
        }
        let refs: Vec<(&str, i16)> = records.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let (data, t) = run(&refs, 12);
        assert_eq!(t.len(), refs.len());
        assert_eq!(collect(&t, &data).len(), refs.len());
    }
}
