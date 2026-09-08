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

/// Slack [`find_semi_and_hash_flat`] requires instead: it reads its 16-byte window up front.
pub const FLAT_SLACK: usize = 16;

#[inline(always)]
fn mix(h: u64, word: u64) -> u64 {
    (h.rotate_left(5) ^ word).wrapping_mul(SEED)
}

/// `0x80` at each byte of `word` that holds a `;`, zero elsewhere.
#[inline(always)]
fn semi_mask(word: u64) -> u64 {
    let x = word ^ SEMI;
    x.wrapping_sub(LOW) & !x & HIGH
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
        let m = semi_mask(word);
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

/// Names 16 bytes and over. Out of line so it does not share registers with the hot path.
#[cold]
#[inline(never)]
fn find_semi_and_hash_long(data: &[u8], pos: usize) -> (usize, u64) {
    find_semi_and_hash(data, pos)
}

/// [`find_semi_and_hash`] with the loop replaced by one fixed 16-byte window.
///
/// The loop it replaces exits after one iteration for names of 7 bytes or fewer and after
/// two for 8..=15. Across the official station list that split is 50.4% / 46.7%, and the
/// generator draws stations independently, so the exit test is a coin flip *by
/// construction* — no history predicts it, and a billion rows pay ~half a misprediction
/// each. Reading both words unconditionally and selecting between them is strictly more
/// work in instructions and strictly less in cycles.
///
/// 97.1% of the list fits the window; the rest take a [`cold`](find_semi_and_hash_long)
/// re-scan, and *that* branch is biased ~33:1, so it predicts.
///
/// The hash is bit-identical to [`find_semi_and_hash`] for every name this path handles,
/// which is what lets both share `find_semi_and_hash_scalar` as their reference.
///
/// Reads [`FLAT_SLACK`] bytes from `pos` — more than [`SLACK`], and unconditionally.
#[inline(always)]
pub fn find_semi_and_hash_flat(data: &[u8], pos: usize) -> (usize, u64) {
    let w0 = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    let w1 = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap());
    let m0 = semi_mask(w0);
    let m1 = semi_mask(w1);

    if m0 | m1 == 0 {
        return find_semi_and_hash_long(data, pos);
    }

    // `trailing_zeros` is 64 on a zero mask, so this is 8 exactly when word 0 holds no `;`.
    let len = if m0 != 0 {
        (m0.trailing_zeros() >> 3) as usize
    } else {
        8 + (m1.trailing_zeros() >> 3) as usize
    };

    // One mix or two, depending on which word the `;` fell in. Both cases end with a mix of
    // a masked word, and both mask the same *count* of bytes — `len` when the name fits in
    // word 0, `len - 8` when it does not, which are equal mod 8. So a single shift serves
    // both, and because the count never reaches 8 the shift is always in range.
    let keep = (1u64 << (8 * (len & 7))) - 1;
    let short = len < 8;
    let prev = if short { 0 } else { mix(0, w0) };
    let last = if short { w0 } else { w1 } & keep;

    (pos + len, finalize(mix(prev, last)))
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

    /// The flat scan must be a drop-in for the loop: same `;`, same hash, at every length
    /// including the 16-byte seam where it hands off to the cold path, and at both ends of
    /// the mask (`len & 7 == 0` is the case that would break a naive shift).
    #[test]
    fn flat_and_loop_paths_agree() {
        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            let d = line(&name);
            let want = find_semi_and_hash_scalar(&d, 0);
            assert_eq!(find_semi_and_hash(&d, 0), want, "loop path, length {len}");
            assert_eq!(find_semi_and_hash_flat(&d, 0), want, "flat path, length {len}");
        }
        for (name, _) in STATIONS {
            let d = line(name);
            assert_eq!(find_semi_and_hash_flat(&d, 0), find_semi_and_hash_scalar(&d, 0), "{name}");
        }
    }

    /// The window is read whole, so bytes past the `;` are in scope for the load even
    /// though they must not reach the hash. Vary them and require the hash to hold still.
    #[test]
    fn flat_ignores_bytes_after_the_delimiter() {
        for len in 0..16usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            let mut seen = None;
            for filler in [b'x', b'0', 0x00, 0xFF, b';', b'\n'] {
                let mut d = name.as_bytes().to_vec();
                d.push(b';');
                d.resize(name.len() + 40, filler);
                let got = find_semi_and_hash_flat(&d, 0);
                assert_eq!(got.0, len, "length {len}, filler {filler:#04x}");
                assert_eq!(*seen.get_or_insert(got), got, "length {len}, filler {filler:#04x}");
            }
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
