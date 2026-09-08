//! SWAR (SIMD-within-a-register) scanning.
//!
//! v2/v3 walk the station name twice: once byte-by-byte looking for `;`, then again to
//! hash it. This does both in a single pass, 8 bytes at a time, using ordinary integer
//! registers — no NEON. Names average ~9 bytes here, so a 16-byte vector would spend most
//! of its width on bytes past the delimiter, and the scalar form avoids moving between the
//! integer and vector register files.

const SEMI: u64 = 0x3B3B_3B3B_3B3B_3B3B;
const LOW: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// Bytes of readable slack the fused scan may touch past a line's `;`.
pub const SLACK: usize = 8;

#[inline(always)]
fn mix(h: u64, word: u64) -> u64 {
    (h.rotate_left(5) ^ word).wrapping_mul(SEED)
}

#[inline(always)]
fn finalize(h: u64) -> u64 {
    let mut h = h;
    h ^= h >> 32;
    h = h.wrapping_mul(0xd6e8_feb8_6659_fd93);
    h ^= h >> 32;
    h
}

/// Finds the `;` at or after `pos` and hashes `data[pos..semi]` in the same pass.
///
/// Returns `(index_of_semicolon, hash)`.
///
/// Reads in 8-byte words, so it may touch up to [`SLACK`] bytes past the `;`. Callers must
/// only use this where that slack is in bounds; see the scalar tail in the workers.
#[inline]
pub fn find_semi_and_hash(data: &[u8], pos: usize) -> (usize, u64) {
    let mut h: u64 = 0;
    let mut p = pos;
    loop {
        let word = u64::from_le_bytes(data[p..p + 8].try_into().unwrap());
        let x = word ^ SEMI;
        let m = x.wrapping_sub(LOW) & !x & HIGH;
        if m != 0 {
            let idx = (m.trailing_zeros() >> 3) as usize;
            // Keep only the bytes before the ';'. idx == 0 yields a zero mask.
            let mask = (1u64 << (idx * 8)).wrapping_sub(1);
            return (p + idx, finalize(mix(h, word & mask)));
        }
        h = mix(h, word);
        p += 8;
    }
}

/// Byte-at-a-time equivalent, for the end of the file where the wide load would run off
/// the mapping. Must agree with [`find_semi_and_hash`] exactly.
pub fn find_semi_and_hash_scalar(data: &[u8], pos: usize) -> (usize, u64) {
    let semi = pos + data[pos..].iter().position(|&b| b == b';').expect("line without ';'");
    let mut h: u64 = 0;
    let mut chunks = data[pos..semi].chunks_exact(8);
    for c in &mut chunks {
        h = mix(h, u64::from_le_bytes(c.try_into().unwrap()));
    }
    let rest = chunks.remainder();
    let mut buf = [0u8; 8];
    buf[..rest.len()].copy_from_slice(rest);
    (semi, finalize(mix(h, u64::from_le_bytes(buf))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stations::STATIONS;
    use std::collections::HashSet;

    /// Pads so the wide load always has its slack in bounds.
    fn line(name: &str) -> Vec<u8> {
        let mut v = name.as_bytes().to_vec();
        v.extend_from_slice(b";12.3\n");
        v.extend_from_slice(&[0u8; 32]);
        v
    }

    #[test]
    fn finds_the_delimiter_at_every_name_length() {
        for len in 1..=100usize {
            let name: String = std::iter::repeat('x').take(len).collect();
            let d = line(&name);
            let (semi, _) = find_semi_and_hash(&d, 0);
            assert_eq!(semi, len, "name length {len}");
            assert_eq!(d[semi], b';');
        }
    }

    #[test]
    fn scalar_and_wide_paths_agree() {
        for len in 1..=100usize {
            // Vary the bytes so the hash actually depends on content.
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            let d = line(&name);
            assert_eq!(find_semi_and_hash(&d, 0), find_semi_and_hash_scalar(&d, 0));
        }
        for (name, _) in STATIONS {
            let d = line(name);
            assert_eq!(find_semi_and_hash(&d, 0), find_semi_and_hash_scalar(&d, 0), "{name}");
        }
    }

    #[test]
    fn works_at_a_nonzero_offset() {
        let mut d = b"prefix;9.9\n".to_vec();
        let tail = line("Zurich");
        let off = d.len();
        d.extend_from_slice(&tail);
        let (semi, h) = find_semi_and_hash(&d, off);
        assert_eq!(&d[off..semi], b"Zurich");
        assert_eq!(h, find_semi_and_hash(&line("Zurich"), 0).1);
    }

    #[test]
    fn empty_name_hashes_without_reading_past_the_delimiter() {
        let d = line("");
        let (semi, _) = find_semi_and_hash(&d, 0);
        assert_eq!(semi, 0);
    }

    #[test]
    fn station_names_do_not_collide() {
        let hashes: HashSet<u64> =
            STATIONS.iter().map(|(n, _)| find_semi_and_hash(&line(n), 0).1).collect();
        assert_eq!(hashes.len(), STATIONS.len());

        let buckets: HashSet<u64> =
            STATIONS.iter().map(|(n, _)| find_semi_and_hash(&line(n), 0).1 & 0xFFFF).collect();
        assert!(buckets.len() > 400, "poor low-bit spread: {}", buckets.len());
    }

    /// Names differing only past the first 8 bytes must hash differently, or the table
    /// would probe constantly.
    #[test]
    fn distinguishes_long_shared_prefixes() {
        let a = find_semi_and_hash(&line("Station_with_long_prefix_A"), 0).1;
        let b = find_semi_and_hash(&line("Station_with_long_prefix_B"), 0).1;
        assert_ne!(a, b);
    }
}
