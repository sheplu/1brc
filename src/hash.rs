//! An FxHash-style hasher. std's default is SipHash, which is far too slow here.

use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
        // Length matters: without it "ab" and "ab\0" would collide after zero padding.
        self.add(bytes.len() as u64);
    }

    /// Fx pushes entropy toward the high bits, but hashbrown picks the bucket from the low
    /// bits. Without this finalizer the table clusters badly.
    #[inline]
    fn finish(&self) -> u64 {
        let mut h = self.hash;
        h ^= h >> 32;
        h = h.wrapping_mul(0xd6e8_feb8_6659_fd93);
        h ^= h >> 32;
        h
    }
}

/// The same hash as [`FxHasher`], without going through the `Hasher` trait — used by the
/// custom table in [`crate::table`], which indexes on the raw value.
#[inline]
pub fn fx_hash(bytes: &[u8]) -> u64 {
    let mut h = FxHasher::default();
    h.write(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stations::STATIONS;
    use std::collections::HashSet;

    fn hash(bytes: &[u8]) -> u64 {
        fx_hash(bytes)
    }

    #[test]
    fn zero_padding_does_not_collide() {
        assert_ne!(hash(b"ab"), hash(b"ab\0"));
        assert_ne!(hash(b""), hash(b"\0"));
        assert_ne!(hash(b"abcdefgh"), hash(b"abcdefgh\0"));
    }

    #[test]
    fn station_names_are_collision_free_in_the_low_bits() {
        // v3 indexes a 65536-slot table; check the real key set spreads over those bits.
        let mut seen = HashSet::new();
        for (name, _) in STATIONS {
            seen.insert(hash(name.as_bytes()));
        }
        assert_eq!(seen.len(), STATIONS.len(), "full-hash collision among station names");

        let buckets: HashSet<u64> =
            STATIONS.iter().map(|(n, _)| hash(n.as_bytes()) & 0xFFFF).collect();
        // Birthday bound for 413 keys in 65536 slots predicts ~412 distinct; allow slack.
        assert!(buckets.len() > 400, "poor low-bit spread: {} buckets", buckets.len());
    }
}
