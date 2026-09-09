//! Delimiter bitmaps over 64-byte blocks, in NEON.
//!
//! Every version up to v10 finds one delimiter at a time and pays a serial dependency for
//! it: a row's name cannot be hashed until the previous row's temperature has been parsed,
//! because that is what says where the row starts. [`split_streams`](crate::chunk::split_streams)
//! papers over this by walking three rows at once, but the chain is still there.
//!
//! Here a whole 64-byte block is compared against `;` and `\n` at once, producing two 64-bit
//! masks with one bit per input byte. Every delimiter in the block is then known before any
//! row is touched, so rows become independent by construction.
//!
//! **The cost of not being x86.** AVX2 gets a 32-byte mask from one `vpmovmskb`. aarch64 has
//! no such instruction, so each 64-byte mask costs four `cmeq`, four `and` against a
//! bit-position vector, and a four-deep `addp` tree to fold 64 bytes of `0x00`/`0xFF` down
//! into 64 bits — 13 operations where AVX2 spends 2. That tax is the reason this may not pay
//! off at all, and it is why [`block_semis`] exists next to [`block_delims`]: half the
//! bitmap for half the price, if half is enough.

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::{
    uint8x16_t, vandq_u8, vceqq_u8, vdupq_n_u8, vgetq_lane_u64, vld1q_u8, vpaddq_u8,
    vreinterpretq_u64_u8,
};

/// Bytes covered by one mask. Also the readable slack a call needs past its pointer.
pub const BLOCK: usize = 64;

/// One bit per byte of the block, weighted by position within its 16-byte lane.
static BIT_POSITION: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// Folds four lanes of byte-wise compare results into one bit per byte, in order.
///
/// Each `addp` sums adjacent pairs, so three of them collapse 16 weighted bytes to one, and
/// the fourth interleaves the four lanes into bytes 0..8 of the result — which read back as a
/// little-endian `u64` is exactly `a`, `b`, `c`, `d` low-to-high.
///
/// # Safety
/// Caller must guarantee NEON is available, which `target_arch = "aarch64"` does.
#[inline(always)]
unsafe fn movemask(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t) -> u64 {
    let bits = vld1q_u8(BIT_POSITION.as_ptr());
    let ab = vpaddq_u8(vandq_u8(a, bits), vandq_u8(b, bits));
    let cd = vpaddq_u8(vandq_u8(c, bits), vandq_u8(d, bits));
    let abcd = vpaddq_u8(ab, cd);
    vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(abcd, abcd)), 0)
}

/// `(semicolons, newlines)` for the [`BLOCK`] bytes at `p`, one bit per byte, LSB first.
///
/// # Safety
/// `p` must have [`BLOCK`] readable bytes.
#[inline(always)]
pub unsafe fn block_delims(p: *const u8) -> (u64, u64) {
    let v0 = vld1q_u8(p);
    let v1 = vld1q_u8(p.add(16));
    let v2 = vld1q_u8(p.add(32));
    let v3 = vld1q_u8(p.add(48));

    let semi = vdupq_n_u8(b';');
    let nl = vdupq_n_u8(b'\n');

    (
        movemask(vceqq_u8(v0, semi), vceqq_u8(v1, semi), vceqq_u8(v2, semi), vceqq_u8(v3, semi)),
        movemask(vceqq_u8(v0, nl), vceqq_u8(v1, nl), vceqq_u8(v2, nl), vceqq_u8(v3, nl)),
    )
}

/// Just the `;` bits. Half of [`block_delims`], for the variant that keeps v9's scan for the
/// temperature end and only wants the name boundaries up front.
///
/// # Safety
/// `p` must have [`BLOCK`] readable bytes.
#[inline(always)]
pub unsafe fn block_semis(p: *const u8) -> u64 {
    let semi = vdupq_n_u8(b';');
    movemask(
        vceqq_u8(vld1q_u8(p), semi),
        vceqq_u8(vld1q_u8(p.add(16)), semi),
        vceqq_u8(vld1q_u8(p.add(32)), semi),
        vceqq_u8(vld1q_u8(p.add(48)), semi),
    )
}

/// Just the `\n` bits — the half that makes row starts independent of the previous parse.
///
/// # Safety
/// `p` must have [`BLOCK`] readable bytes.
#[inline(always)]
pub unsafe fn block_newlines(p: *const u8) -> u64 {
    let nl = vdupq_n_u8(b'\n');
    movemask(
        vceqq_u8(vld1q_u8(p), nl),
        vceqq_u8(vld1q_u8(p.add(16)), nl),
        vceqq_u8(vld1q_u8(p.add(32)), nl),
        vceqq_u8(vld1q_u8(p.add(48)), nl),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn masks(bytes: &[u8]) -> (u64, u64) {
        let mut buf = [0u8; BLOCK];
        buf[..bytes.len()].copy_from_slice(bytes);
        unsafe { block_delims(buf.as_ptr()) }
    }

    /// The mask has to agree with the obvious byte loop at every position, or a row boundary
    /// lands in the wrong place and the failure is silent.
    #[test]
    fn agrees_with_a_byte_loop() {
        let mut buf = [b'x'; BLOCK];
        for i in 0..BLOCK {
            for (needle, which) in [(b';', 0), (b'\n', 1)] {
                buf[i] = needle;
                let got = unsafe { block_delims(buf.as_ptr()) };
                let got = if which == 0 { got.0 } else { got.1 };
                assert_eq!(got, 1u64 << i, "{} at byte {i}", needle as char);
                buf[i] = b'x';
            }
        }
    }

    #[test]
    fn separates_the_two_classes() {
        let (s, n) = masks(b"ab;1.2\ncd;-3.4\n");
        assert_eq!(s, (1 << 2) | (1 << 9));
        assert_eq!(n, (1 << 6) | (1 << 14));
    }

    /// Both halves must equal the corresponding half of the combined call, since the row
    /// loop picks whichever is cheaper and they have to be interchangeable.
    #[test]
    fn halves_match_the_combined_call() {
        let mut buf = [0u8; BLOCK];
        for (i, b) in b"Zurich;-12.3\nAbha;5.0\nSaint_Petersburg;99.9\n".iter().enumerate() {
            buf[i] = *b;
        }
        let (s, n) = unsafe { block_delims(buf.as_ptr()) };
        assert_eq!(s, unsafe { block_semis(buf.as_ptr()) });
        assert_eq!(n, unsafe { block_newlines(buf.as_ptr()) });
        assert_eq!(s.count_ones(), 3);
        assert_eq!(n.count_ones(), 3);
    }

    /// Every byte set, so the fold has to carry a bit out of all 64 lanes without any pair
    /// summing into its neighbour.
    #[test]
    fn saturated_block_sets_every_bit() {
        assert_eq!(masks(&[b';'; BLOCK]).0, u64::MAX);
        assert_eq!(masks(&[b'\n'; BLOCK]).1, u64::MAX);
    }
}
