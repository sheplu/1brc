//! SWAR (SIMD-within-a-register) scanning.
//!
//! v2/v3 walk the station name twice: once byte-by-byte looking for `;`, then again to
//! hash it. This does both in a single pass, 8 bytes at a time, using ordinary integer
//! registers — no NEON. Names average ~9 bytes here, so a 16-byte vector would spend most
//! of its width on bytes past the delimiter, and the scalar form avoids moving between the
//! integer and vector register files.

const SEMI: u64 = 0x3B3B_3B3B_3B3B_3B3B;
const NL: u64 = 0x0A0A_0A0A_0A0A_0A0A;
const LOW: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// Bytes of readable slack the fused scan may touch past a line's `;`.
pub const SLACK: usize = 8;

/// Slack [`find_semi_and_hash_flat`] requires instead: it reads its 16-byte window up front.
pub const FLAT_SLACK: usize = 16;

/// Slack [`line_len_flat`] requires, and the value it returns when it finds no `\n`.
///
/// It is not larger than what the flat path already touches: a name that fits
/// [`FLAT_SLACK`] puts the `;` at offset 15 at worst, and the parser then loads eight bytes
/// from offset 16. Adding the scan costs no guard that was not already being paid.
pub const LINE_SLACK: usize = 24;

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

/// The same, for `\n`.
#[inline(always)]
fn nl_mask(word: u64) -> u64 {
    let x = word ^ NL;
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

/// The word-at-a-time scan, returning the accumulator *before* [`finalize`].
#[inline]
fn scan_loop(data: &[u8], pos: usize) -> (usize, u64) {
    let mut h: u64 = 0;
    let mut p = pos;
    loop {
        let word = u64::from_le_bytes(data[p..p + 8].try_into().unwrap());
        let m = semi_mask(word);
        if m != 0 {
            let idx = (m.trailing_zeros() >> 3) as usize;
            // Keep only the bytes before the ';'. idx == 0 yields a zero mask.
            let mask = (1u64 << (idx * 8)).wrapping_sub(1);
            return (p + idx, mix(h, word & mask));
        }
        h = mix(h, word);
        p += 8;
    }
}

/// Finds the `;` at or after `pos` and hashes `data[pos..semi]` in the same pass.
///
/// Returns `(index_of_semicolon, hash)`.
///
/// Reads in 8-byte words, so it may touch up to [`SLACK`] bytes past the `;`. Callers must
/// only use this where that slack is in bounds; see the scalar tail in the workers.
#[inline]
pub fn find_semi_and_hash(data: &[u8], pos: usize) -> (usize, u64) {
    let (semi, h) = scan_loop(data, pos);
    (semi, finalize(h))
}

/// Names 16 bytes and over. Out of line so it does not share registers with the hot path.
#[cold]
#[inline(never)]
fn scan_loop_long(data: &[u8], pos: usize) -> (usize, u64) {
    scan_loop(data, pos)
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
/// 97.1% of the list fits the window; the rest take a `#[cold]` re-scan through the loop, and
/// *that* branch is biased ~33:1, so it predicts.
///
/// The hash is bit-identical to [`find_semi_and_hash`] for every name this path handles,
/// which is what lets both share `find_semi_and_hash_scalar` as their reference.
///
/// Reads [`FLAT_SLACK`] bytes from `pos` — more than [`SLACK`], and unconditionally.
#[inline(always)]
pub fn find_semi_and_hash_flat(data: &[u8], pos: usize) -> (usize, u64) {
    let (semi, h) = scan_flat(data, pos);
    (semi, finalize(h))
}

/// [`find_semi_and_hash_flat`] without the final avalanche.
///
/// `mix` ends in a multiply, so the *high* bits of what it returns are already well mixed
/// and `finalize` only buys spread in the low ones. A table that takes its slot from the
/// high bits does not need it, and skipping it removes five operations from the dependency
/// chain that ends at the table load. The result is not a general-purpose hash: use it only
/// where the index comes off the top.
#[inline(always)]
pub fn find_semi_and_hash_flat_raw(data: &[u8], pos: usize) -> (usize, u64) {
    scan_flat(data, pos)
}

/// [`find_semi_and_hash_flat_raw`] for a caller that already knows where the name ends.
///
/// A delimiter bitmap yields the `;` for free, which kills the two SWAR masks, the
/// `trailing_zeros` and the branch that picks between the two words' results — everything but
/// the loads and the mix chain. The remaining select is on `len`, which the caller has in a
/// register well before the bytes arrive, so it no longer sits behind the load.
///
/// Bit-identical to [`find_semi_and_hash_flat_raw`] at every length, including the handoff at
/// 16 where both give up and re-scan.
#[inline(always)]
pub fn hash_len_raw(data: &[u8], pos: usize, len: usize) -> u64 {
    if len >= 16 {
        return scan_loop_long(data, pos).1;
    }
    let w0 = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    let w1 = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap());

    let keep = (1u64 << (8 * (len & 7))) - 1;
    let short = len < 8;
    let prev = if short { 0 } else { mix(0, w0) };
    let last = if short { w0 } else { w1 } & keep;
    mix(prev, last)
}

/// [`find_semi_and_hash_flat_raw`], also handing back the name's 16-byte window already
/// zero-padded — the two little-endian words of
/// [`probe_key`](crate::inline_table::probe_key)'s low half.
///
/// The scan loads those bytes anyway, and it already masks the word the `;` falls in, to keep
/// the trailing garbage out of the hash. That masked window *is* the probe key for any name
/// short enough to fit here, with the high half zero, so returning it costs one select and
/// saves the table four loads, two mask-table lookups and four `and`s — and saves the
/// register allocator having to keep a 32-byte key alive across the hash, which it was
/// spilling to the stack.
///
/// `None` when the window holds no `;`: the length is not known here, so neither is the key.
#[inline(always)]
pub fn scan_flat_keyed(data: &[u8], pos: usize) -> Option<(usize, u64, u64, u64)> {
    let w0 = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    let w1 = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap());
    let m0 = semi_mask(w0);
    let m1 = semi_mask(w1);

    if m0 | m1 == 0 {
        return None;
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
    // The same two words the hash consumed, in name order: a short name's masked word is the
    // whole key, a longer one's is the second half.
    let (klo, khi) = if short { (last, 0) } else { (w0, last) };

    Some((pos + len, mix(prev, last), klo, khi))
}

/// [`scan_flat_keyed`] with the key mask taken off `trailing_zeros` instead of off the length.
///
/// The mask that clears the bytes past the `;` is built from the length, which was built from
/// the bit position, which came from the mask:
///
/// ```text
///   len  = tz >> 3            (+8 on the w1 side)
///   keep = (1 << (8 * (len & 7))) - 1
/// ```
///
/// `8 * (len & 7)` is `tz & 0x38`, so the round trip through `len` is not needed at all. LLVM
/// sees this on the w0 side, where `len` *is* `tz >> 3` and it emits `and x13, x13, #0x38`. It
/// does not see it on the w1 side, because the `+ 8` gets between the shift and the mask, and
/// there it emits `lsr #3`, `add #8`, `ubfiz` — three instructions where the other side has
/// one.
///
/// Selecting the bit position first and deriving both `len` and `keep` from it puts the two
/// sides in the same shape. The `+ 8` still happens, but only on `len`, which nothing else
/// feeds.
///
/// Nothing else changes: same two loads, same two masks, same select, same hash. This is the
/// [`+cssc`](../index.html) shape — strictly fewer instructions for the same work, no new
/// branch, no new live value — which is the only shape this project has ever measured a win
/// from.
#[inline(always)]
pub fn scan_flat_keyed_tz(data: &[u8], pos: usize) -> Option<(usize, u64, u64, u64)> {
    let w0 = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    let w1 = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap());
    let m0 = semi_mask(w0);
    let m1 = semi_mask(w1);

    if m0 | m1 == 0 {
        return None;
    }

    let short = m0 != 0;
    // Where the `;` is inside its own word: 7, 15, .. 63, or 64 if that word holds none —
    // which cannot happen here, because the word is chosen by which mask is non-zero.
    let tz = if short { m0.trailing_zeros() } else { m1.trailing_zeros() };

    let len = (tz >> 3) as usize + if short { 0 } else { 8 };
    let keep = (1u64 << (tz & 0x38)) - 1;
    let prev = if short { 0 } else { mix(0, w0) };
    let last = if short { w0 } else { w1 } & keep;
    let (klo, khi) = if short { (last, 0) } else { (w0, last) };

    Some((pos + len, mix(prev, last), klo, khi))
}

/// [`scan_flat_keyed`] with the two delimiter masks built by one vector compare.
///
/// This is not the [NEON bitmap](../index.html) the project already rejected. That one compared
/// a 64-byte block, collected the results into a bitmap, and drove a *separate* drain loop off
/// the bits — the mask was cheap and the drain cost +194 ms. Here the row loop is untouched:
/// same window, same 16 bytes, same two masks handed to the same downstream code. Only the
/// instructions that produce the masks change, and v11's masked-key trick — the thing the bitmap
/// design had to forfeit — is preserved exactly, because the key still comes out of `w0`/`w1`.
///
/// The point is not the compare, it is the register file. [`semi_mask`] needs `SEMI` and `LOW`
/// resident in general-purpose registers: `HIGH` is an ARM64 logical immediate and free, but
/// `0x3B3B..` is not, and `SUB` takes arithmetic immediates only, so `LOW` cannot fold either.
/// A `cmeq` against a splat holds `SEMI` in a vector register instead, and `LOW` stops existing.
/// The v18 measurement says this loop converts a freed operation into a rematerialised constant
/// at 1:1 — so the thing to hand it is not fewer operations but **fewer live values**, and this
/// returns two of the five constants the loop currently pins.
///
/// It is also fewer instructions, which the same measurement says is worth ~1 ms each:
///
/// ```text
///   scalar   eor/sub/bic/and per word            8
///   vector   ldr q / cmeq / fmov / umov          4      + one 16-byte load
/// ```
///
/// The two extracts give `0xFF` at each matching byte where the SWAR form gives `0x80`. Both
/// have their lowest set bit at `8 * index`, so `trailing_zeros` and every mask derived from it
/// are unchanged, and `m0 | m1 == 0` still means "no `;` in sixteen bytes".
///
/// Reads [`FLAT_SLACK`] bytes from `pos`, as [`scan_flat_keyed`] does.
#[inline(always)]
pub fn scan_flat_keyed_neon(data: &[u8], pos: usize) -> Option<(usize, u64, u64, u64)> {
    use std::arch::aarch64::{
        vceqq_u8, vdupq_n_u8, vgetq_lane_u64, vld1q_u8, vreinterpretq_u64_u8,
    };

    let w0 = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    let w1 = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap());

    // Safe because the two reads above already proved `pos + 16 <= data.len()`.
    let (m0, m1) = unsafe {
        let eq = vreinterpretq_u64_u8(vceqq_u8(vld1q_u8(data.as_ptr().add(pos)), vdupq_n_u8(b';')));
        (vgetq_lane_u64(eq, 0), vgetq_lane_u64(eq, 1))
    };

    if m0 | m1 == 0 {
        return None;
    }

    let short = m0 != 0;
    let tz = if short { m0.trailing_zeros() } else { m1.trailing_zeros() };

    let len = (tz >> 3) as usize + if short { 0 } else { 8 };
    let keep = (1u64 << (tz & 0x38)) - 1;
    let prev = if short { 0 } else { mix(0, w0) };
    let last = if short { w0 } else { w1 } & keep;
    let (klo, khi) = if short { (last, 0) } else { (w0, last) };

    Some((pos + len, mix(prev, last), klo, khi))
}

/// [`scan_flat_keyed_neon`] with the key masking moved into the vector unit too.
///
/// v19 put the *compare* on the vector unit and won 23 ms against a prediction of 1.5, which
/// established that this loop is bound on integer ALU throughput rather than on instruction
/// issue. That reading has a corollary: every remaining integer operation that has a byte-lane
/// equivalent is mispriced, and should be moved for the same reason.
///
/// Three clusters are left in the scan, and this moves all three:
///
/// ```text
///   v19                                              here
///   ---------------------------------------------    -------------------------------------
///   two 8-byte loads, then mask in a GPR             one 16-byte load, mask in place
///   two lane extracts, `orr`, `cbz`, a `csel` for    one `shrn` + one extract; `tz >> 2`
///     which half `tz` comes from, `lsr`+`add`          is the length outright
///   `and #0x38`, `lsl`, `sub`, `and`, two `csel`     `cmhi` against a splat length, `and.16b`
/// ```
///
/// The last row is the point. `keep = (1 << (tz & 0x38)) - 1` is a *byte-lane select* written
/// in scalar arithmetic: it keeps the bytes before the `;` and drops the rest. NEON does that
/// with one compare against a constant `[0, 1, .. 15]`, and because the mask covers all sixteen
/// lanes at once, `klo` and `khi` come out of it directly — the two `csel` that chose which
/// word was the masked one are not replaced, they are unnecessary. Whichever half the `;` is
/// in, the other half is already correct: fully kept below it, fully zeroed above it.
///
/// The `shrn` is the standard aarch64 movemask. Narrowing eight 16-bit lanes by 4 leaves one
/// nibble per input byte, so a single 64-bit extract carries all sixteen compare results and
/// `trailing_zeros() >> 2` is the index of the first `;`. v19 needed two extracts and then a
/// select, because a 64-bit lane only holds eight of the answers.
///
/// The hash is untouched and bit-identical: `short` still means `len < 8`, and the two `csel`
/// that pick `prev` and `last` stay. They are on the multiply's dependency chain, not on the
/// masking, and changing them would change which hash the table sees.
///
/// One range check instead of two, as a side effect of there being one load instead of three.
/// That is not the mechanism — v15 and v20 both removed range checks and both lost — it is
/// simply what a single window costs.
///
/// Reads [`FLAT_SLACK`] bytes from `pos`, as [`scan_flat_keyed`] does.
#[inline(always)]
pub fn scan_flat_keyed_vmask(data: &[u8], pos: usize) -> Option<(usize, u64, u64, u64)> {
    use std::arch::aarch64::{
        vandq_u8, vceqq_u8, vcltq_u8, vdupq_n_u8, vget_lane_u64, vgetq_lane_u64, vld1q_u8,
        vreinterpret_u64_u8, vreinterpretq_u16_u8, vreinterpretq_u64_u8, vshrn_n_u16,
    };

    /// The lane indices the length is compared against. One `ldr q` from `.rodata`, hoisted.
    const IOTA: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    let w: &[u8; 16] = data[pos..pos + FLAT_SLACK].try_into().unwrap();

    // SAFETY: `w` is sixteen readable bytes, and `IOTA` is sixteen more.
    let (v, nib) = unsafe {
        let v = vld1q_u8(w.as_ptr());
        let eq = vreinterpretq_u16_u8(vceqq_u8(v, vdupq_n_u8(b';')));
        (v, vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16::<4>(eq)), 0))
    };

    if nib == 0 {
        return None;
    }

    // Four bits per input byte, so the first `;` is at byte `tz / 4`.
    let len = (nib.trailing_zeros() >> 2) as usize;

    // SAFETY: no memory is touched here; `IOTA` is read through a sixteen-byte load.
    let (klo, khi) = unsafe {
        let keep = vcltq_u8(vld1q_u8(IOTA.as_ptr()), vdupq_n_u8(len as u8));
        let k = vreinterpretq_u64_u8(vandq_u8(v, keep));
        (vgetq_lane_u64(k, 0), vgetq_lane_u64(k, 1))
    };

    let short = len < 8;
    let prev = if short { 0 } else { mix(0, klo) };
    let last = if short { klo } else { khi };

    Some((pos + len, mix(prev, last), klo, khi))
}

/// [`scan_flat_keyed_vmask`] with the lane mask built from the compare rather than from `len`.
///
/// v21 measured the vector masking as worth 8 ms where the integer-operation count predicted 20
/// to 55. The account it gives is that the mask is built from a *scalar*: `ctz` produces `len` in
/// a general-purpose register, `dup.16b` carries it back to the vector unit, and the masked words
/// come out again — three register-file crossings, all in series on the chain that feeds the hash.
///
/// This builds the same mask without a scalar ever being involved. An inclusive prefix-OR across
/// the compare result marks every lane at or after the first `;`, and `bic` keeps the rest:
///
/// ```text
///   p  = cmeq(v, ';')
///   p |= p << 1 lane      p |= p << 2      p |= p << 4      p |= p << 8
///   masked = v & ~p
/// ```
///
/// Four `ext`/`orr` pairs and a `bic`, which is five vector operations more than v21 spends and
/// nine more than v19. What it buys is that `klo` and `khi` no longer depend on `len`. The two
/// derivations become independent chains off the same `cmeq` — `shrn`/`fmov`/`ctz` for the length,
/// `ext`-`orr`-`bic` for the key — where v21 has one chain twice as long with a round trip through
/// the general-purpose file in the middle of it.
///
/// So this is the discriminator for v21's own explanation, and all three outcomes say something.
/// Faster than v21: the serial crossings were the term, and the way to spend the idle vector unit
/// is on work that never comes back. The same: they were not, and the loop is bound on something
/// this project has not named. Slower: five extra vector operations a row cost more than the
/// shortened chain saves, which would be the first evidence that the vector side has a ceiling
/// too — and would date the "completely idle" claim v19 has been resting on.
///
/// Bit-identical to [`scan_flat_keyed`], hash included; `pfx_agrees_with_the_scalar_masking`
/// holds it there.
///
/// Reads [`FLAT_SLACK`] bytes from `pos`, as [`scan_flat_keyed`] does.
#[inline(always)]
pub fn scan_flat_keyed_pfx(data: &[u8], pos: usize) -> Option<(usize, u64, u64, u64)> {
    use std::arch::aarch64::{
        vbicq_u8, vceqq_u8, vdupq_n_u8, vextq_u8, vget_lane_u64, vgetq_lane_u64, vld1q_u8,
        vorrq_u8, vreinterpret_u64_u8, vreinterpretq_u16_u8, vreinterpretq_u64_u8, vshrn_n_u16,
    };

    let w: &[u8; 16] = data[pos..pos + FLAT_SLACK].try_into().unwrap();

    // SAFETY: `w` is sixteen readable bytes.
    let (v, eq, nib) = unsafe {
        let v = vld1q_u8(w.as_ptr());
        let eq = vceqq_u8(v, vdupq_n_u8(b';'));
        let nib = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq))), 0);
        (v, eq, nib)
    };

    if nib == 0 {
        return None;
    }

    let len = (nib.trailing_zeros() >> 2) as usize;

    // SAFETY: register-to-register only. `vextq_u8(zero, p, 16 - k)` takes the top `k` bytes of
    // the zero vector followed by the low `16 - k` of `p`, which is `p` shifted up by `k` lanes;
    // four of those OR the compare forward into every higher lane.
    let (klo, khi) = unsafe {
        let z = vdupq_n_u8(0);
        let p = vorrq_u8(eq, vextq_u8(z, eq, 15));
        let p = vorrq_u8(p, vextq_u8(z, p, 14));
        let p = vorrq_u8(p, vextq_u8(z, p, 12));
        let p = vorrq_u8(p, vextq_u8(z, p, 8));
        let k = vreinterpretq_u64_u8(vbicq_u8(v, p));
        (vgetq_lane_u64(k, 0), vgetq_lane_u64(k, 1))
    };

    let short = len < 8;
    let prev = if short { 0 } else { mix(0, klo) };
    let last = if short { klo } else { khi };

    Some((pos + len, mix(prev, last), klo, khi))
}

/// [`scan_flat_keyed`] taking the 16-byte window instead of a slice and an index, and
/// returning the name's *length* rather than the absolute position of the `;`.
///
/// Identical arithmetic; only the load differs. Building the two words from `data[pos..]`
/// costs four compares and four branches to a panic, because nothing relates `pos` to
/// `data.len()`. An `&[u8; 16]` carries that proof in the type, so the caller fetches the
/// window once — one compare in `first_chunk` — and the scan checks nothing.
///
/// The body is duplicated rather than shared, for the reason given on
/// [`parse_temp_branchless_win`](crate::parse::parse_temp_branchless_win);
/// `win_agrees_with_the_slice_form` holds them equal.
#[inline(always)]
pub fn scan_flat_keyed_win(w: &[u8; 16]) -> Option<(usize, u64, u64, u64)> {
    let w0 = u64::from_le_bytes(w[..8].try_into().unwrap());
    let w1 = u64::from_le_bytes(w[8..].try_into().unwrap());
    let m0 = semi_mask(w0);
    let m1 = semi_mask(w1);

    if m0 | m1 == 0 {
        return None;
    }

    let len = if m0 != 0 {
        (m0.trailing_zeros() >> 3) as usize
    } else {
        8 + (m1.trailing_zeros() >> 3) as usize
    };

    let keep = (1u64 << (8 * (len & 7))) - 1;
    let short = len < 8;
    let prev = if short { 0 } else { mix(0, w0) };
    let last = if short { w0 } else { w1 } & keep;
    let (klo, khi) = if short { (last, 0) } else { (w0, last) };

    Some((len, mix(prev, last), klo, khi))
}

/// [`scan_flat_keyed_neon`] on a window, which is [`scan_flat_keyed_win`]'s relation to
/// [`scan_flat_keyed`].
///
/// The two changes compose, and the reason to compose them is that v15 measured what happens
/// when only one is made. v15 dropped the range checks exactly this way and lost 9.5 ms: the
/// allocator took the 28 freed instructions per pair and put 19 back as rematerialised
/// constants, rebuilding `SEED` with four `mov`/`movk` and `SEMI` with a `mov`/`orr` pair. Half
/// of that specific failure is a register holding `0x3B3B..`, and this form does not have one.
///
/// `neon_win_agrees_with_the_slice_form` holds it equal to [`scan_flat_keyed`], as the other
/// copies are held.
#[inline(always)]
pub fn scan_flat_keyed_neon_win(w: &[u8; 16]) -> Option<(usize, u64, u64, u64)> {
    use std::arch::aarch64::{
        vceqq_u8, vdupq_n_u8, vgetq_lane_u64, vld1q_u8, vreinterpretq_u64_u8,
    };

    let w0 = u64::from_le_bytes(w[..8].try_into().unwrap());
    let w1 = u64::from_le_bytes(w[8..].try_into().unwrap());

    // SAFETY: `w` is sixteen readable bytes by construction, which is what `vld1q_u8` reads.
    let (m0, m1) = unsafe {
        let eq = vreinterpretq_u64_u8(vceqq_u8(vld1q_u8(w.as_ptr()), vdupq_n_u8(b';')));
        (vgetq_lane_u64(eq, 0), vgetq_lane_u64(eq, 1))
    };

    if m0 | m1 == 0 {
        return None;
    }

    let short = m0 != 0;
    let tz = if short { m0.trailing_zeros() } else { m1.trailing_zeros() };

    let len = (tz >> 3) as usize + if short { 0 } else { 8 };
    let keep = (1u64 << (tz & 0x38)) - 1;
    let prev = if short { 0 } else { mix(0, w0) };
    let last = if short { w0 } else { w1 } & keep;
    let (klo, khi) = if short { (last, 0) } else { (w0, last) };

    Some((len, mix(prev, last), klo, khi))
}

/// Offset from `pos` of the `\n` that ends the row starting there — the row's length.
///
/// Every other way of reaching the end of a row goes through the `;` first: find the
/// delimiter, load the value that follows it, find the dot in *that*, and the newline is two
/// bytes further on. Three of those four steps are a dependency chain rooted at the load of
/// the name, and the last two cannot even begin until the first `trailing_zeros` has resolved,
/// because the value's address is not known before then. Scanning for the `\n` directly needs
/// none of it. All three words come off `pos` alone, so all three loads issue together, and
/// what the caller wants — where the next row starts — falls out of one `trailing_zeros`
/// instead of two chained ones with a chained load between them.
///
/// It costs a third load and a third mask to buy that. The first two words are the ones
/// [`scan_flat_keyed`] already reads, so on a caller that does both the marginal cost is one
/// load, one mask and the select.
///
/// Correct only for a row that ends within [`LINE_SLACK`] bytes, which is every row the flat
/// path handles: its name fits [`FLAT_SLACK`], and a value is at most six bytes more. A row
/// that does not is the cold path's business, and this returns [`LINE_SLACK`] rather than
/// anything meaningful — callers must discard it there.
///
/// The bytes past the row are read but cannot mislead: `trailing_zeros` takes the *first*
/// `\n`, and a row's own terminator precedes anything the window picks up from the row after
/// it. Where the file's last row carries no terminator at all, the first `\n` this can find
/// lies past the bytes the worker was given, so the offset it returns is past the end of the
/// range and the caller stops — which is what the dot search it replaces also does.
///
/// **The combine has no conditional in it, and that is not a style choice.** Written the
/// obvious way — `if n0 != 0 { .. } else if n1 != 0 { .. } else { .. }` — LLVM emits a real
/// `b.eq` and sinks the third load into the arm that needs it. That is the whole idea
/// destroyed: the load becomes dependent on the second mask, which is what this function
/// exists to avoid, and it lands behind a branch whose direction is the name's length and so
/// cannot be predicted. Selects would be fine; branches are not, and there is no way to ask
/// for one and not the other.
///
/// So the arithmetic carries the case analysis instead. `trailing_zeros` is 64 on a zero mask
/// and at most 63 otherwise, which makes `t >> 6` a ready-made "this word held no `\n`" flag
/// and `-(t >> 6)` a mask over the next word's contribution — no compare, nothing to
/// speculate, and all three loads unconditionally live. `t >> 3` is the byte index, and it is
/// 8 on a zero mask, which is exactly the offset to the next word: the three terms add up on
/// their own, and the all-empty case lands on [`LINE_SLACK`] without being special-cased.
#[inline(always)]
pub fn line_len_flat(data: &[u8], pos: usize) -> usize {
    // One slice, one bounds check: three separate ones cost three, and the pair loop is
    // already paying more for range checks than for the parse.
    let w: &[u8; LINE_SLACK] = data[pos..pos + LINE_SLACK].try_into().unwrap();
    let t0 = nl_mask(u64::from_le_bytes(w[..8].try_into().unwrap())).trailing_zeros() as usize;
    let t1 = nl_mask(u64::from_le_bytes(w[8..16].try_into().unwrap())).trailing_zeros() as usize;
    let t2 = nl_mask(u64::from_le_bytes(w[16..].try_into().unwrap())).trailing_zeros() as usize;

    let m0 = 0usize.wrapping_sub(t0 >> 6);
    let m1 = 0usize.wrapping_sub(t1 >> 6);
    (t0 >> 3) + (m0 & ((t1 >> 3) + (m1 & (t2 >> 3))))
}

#[inline(always)]
fn scan_flat(data: &[u8], pos: usize) -> (usize, u64) {
    match scan_flat_keyed(data, pos) {
        Some((semi, h, _, _)) => (semi, h),
        None => scan_loop_long(data, pos),
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

    /// The length-driven hash drops the scan on the claim that it only ever used it to derive
    /// the length. Check that against the scan it replaces at every length, including 16 where
    /// both fall back, and on every real name.
    #[test]
    fn length_driven_hash_agrees_with_the_flat_scan() {
        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            let d = line(&name);
            let (semi, want) = find_semi_and_hash_flat_raw(&d, 0);
            assert_eq!(semi, len);
            assert_eq!(hash_len_raw(&d, 0, len), want, "length {len}");
        }
        for (name, _) in STATIONS {
            let d = line(name);
            let want = find_semi_and_hash_flat_raw(&d, 0).1;
            assert_eq!(hash_len_raw(&d, 0, name.len()), want, "{name}");
        }
    }

    /// v11 stops building the probe key from memory and takes the scan's masked words
    /// instead. That is only sound if the two agree bit for bit, so check them against each
    /// other at every length the window covers — 0 and 8 included, where the mask is empty —
    /// and on every real name. Also require the scan to decline exactly when the key would be
    /// wrong, at 16 and above, since past there the high half is no longer zero.
    #[test]
    fn the_scans_key_matches_the_one_built_from_memory() {
        use crate::inline_table::{key_from_words, probe_key};

        let check = |name: &str| {
            let d = line(name);
            match scan_flat_keyed(&d, 0) {
                Some((semi, _, klo, khi)) => {
                    assert_eq!(semi, name.len(), "{name:?}");
                    assert!(name.len() < 16, "{name:?} should have declined");
                    assert_eq!(key_from_words(klo, khi), probe_key(&d, 0, semi), "{name:?}");
                }
                None => assert!(name.len() >= 16, "{name:?} declined but fits the window"),
            }
        };

        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            check(&name);
        }
        for (name, _) in STATIONS {
            check(name);
        }
    }

    /// [`scan_flat_keyed_tz`] claims `8 * (len & 7)` and `tz & 0x38` name the same shift on
    /// both sides of the select. The w1 side is where they could differ, because there `len`
    /// carries a `+ 8` that `tz` does not — so check every length across the 8-byte seam, plus
    /// every real station, plus a window with no `;` in it.
    #[test]
    fn tz_agrees_with_the_length_driven_mask() {
        let check = |d: &[u8]| {
            assert_eq!(scan_flat_keyed_tz(d, 0), scan_flat_keyed(d, 0), "{:?}", &d[..16]);
        };

        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            check(&line(&name));
        }
        for (name, _) in STATIONS {
            check(&line(name));
        }
        check(&[b'x'; 64]);
    }

    /// [`scan_flat_keyed_neon`] swaps the SWAR masks for a vector compare, which changes the
    /// non-zero bytes of the masks from `0x80` to `0xFF`. Everything downstream reads those
    /// masks through `trailing_zeros`, so the two forms are equal only if no byte outside the
    /// `;` positions ever sets a lower bit — check it the same way as the rest: every length
    /// across both 8-byte seams, every real station, and a window with no `;` in it.
    #[test]
    fn neon_agrees_with_the_swar_masks() {
        let check = |d: &[u8]| {
            assert_eq!(scan_flat_keyed_neon(d, 0), scan_flat_keyed(d, 0), "{:?}", &d[..16]);
        };

        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            check(&line(&name));
        }
        for (name, _) in STATIONS {
            check(&line(name));
        }
        check(&[b'x'; 64]);
    }

    /// The window form of the vector scan is a third copy of the same arithmetic, differing from
    /// [`scan_flat_keyed`] in both the load and the compare. Held to it the same way.
    #[test]
    fn neon_win_agrees_with_the_slice_form() {
        let check = |d: &[u8]| {
            let w: &[u8; 16] = d[..16].try_into().unwrap();
            let want = scan_flat_keyed(d, 0).map(|(semi, h, klo, khi)| (semi, h, klo, khi));
            assert_eq!(scan_flat_keyed_neon_win(w), want, "{:?}", &d[..16]);
        };

        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            check(&line(&name));
        }
        for (name, _) in STATIONS {
            check(&line(name));
        }
        check(&[b'x'; 64]);
    }

    /// [`scan_flat_keyed_vmask`] derives `klo` and `khi` a completely different way: a lane
    /// compare against `[0, 1, .. 15]` rather than a shift-and-subtract in a GPR, with no select
    /// deciding which of the two words got masked. The claim is that the two agree *bit for
    /// bit*, including the hash — so this is the strictest of these differentials and the seam
    /// cases matter most. `len == 8` in particular is the one where `khi` is fully zeroed but
    /// `short` is still false, which is the case a "mask the non-zero half" shortcut would get
    /// wrong.
    #[test]
    fn vmask_agrees_with_the_scalar_masking() {
        let check = |d: &[u8]| {
            assert_eq!(scan_flat_keyed_vmask(d, 0), scan_flat_keyed(d, 0), "{:?}", &d[..16]);
        };

        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            check(&line(&name));
        }
        for (name, _) in STATIONS {
            check(&line(name));
        }
        check(&[b'x'; 64]);
    }

    /// [`scan_flat_keyed_pfx`] reaches the same two words a fourth way — an inclusive prefix-OR
    /// across the compare, then `bic` — and the whole point of it is that no scalar is involved,
    /// so the usual "does `trailing_zeros` see the same thing" argument does not cover it. What
    /// has to hold instead is that the prefix-OR is *inclusive* and saturates: a `;` at lane 0
    /// must zero all sixteen, a `;` at lane 15 must keep fifteen, and a second `;` later in the
    /// window must change nothing. Every length across both seams covers all three.
    #[test]
    fn pfx_agrees_with_the_scalar_masking() {
        let check = |d: &[u8]| {
            assert_eq!(scan_flat_keyed_pfx(d, 0), scan_flat_keyed(d, 0), "{:?}", &d[..16]);
        };

        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            check(&line(&name));
        }
        for (name, _) in STATIONS {
            check(&line(name));
        }
        check(&[b'x'; 64]);
    }

    /// Two `;` inside one window is the case the prefix-OR could plausibly get wrong, since it
    /// propagates every match forward rather than just the first.
    #[test]
    fn pfx_finds_the_first_semicolon_at_an_offset() {
        let mut d = vec![b'q'; 8];
        d.extend_from_slice(b"Abha;1.2\nSaint-Pierre;3.4\n");
        d.resize(d.len() + 32, b'z');
        assert_eq!(scan_flat_keyed_pfx(&d, 8), scan_flat_keyed(&d, 8));
        assert_eq!(scan_flat_keyed_pfx(&d, 17), scan_flat_keyed(&d, 17));

        let mut two = b"ab;cd;ef;gh;ij;k".to_vec();
        two.resize(64, b'z');
        assert_eq!(scan_flat_keyed_pfx(&two, 0), scan_flat_keyed(&two, 0));
    }

    #[test]
    fn vmask_finds_the_first_semicolon_at_an_offset() {
        let mut d = vec![b'q'; 8];
        d.extend_from_slice(b"Abha;1.2\nSaint-Pierre;3.4\n");
        d.resize(d.len() + 32, b'z');
        assert_eq!(scan_flat_keyed_vmask(&d, 8), scan_flat_keyed(&d, 8));
        assert_eq!(scan_flat_keyed_vmask(&d, 17), scan_flat_keyed(&d, 17));
    }

    /// A `;` in the window is found at the same place whatever the window sits at, and a second
    /// `;` past the first must not move the answer — the vector compare sees all sixteen bytes
    /// at once, where the SWAR form saw two words.
    #[test]
    fn neon_finds_the_first_semicolon_at_an_offset() {
        let mut d = vec![b'q'; 8];
        d.extend_from_slice(b"Abha;1.2\nSaint-Pierre;3.4\n");
        d.resize(d.len() + 32, b'z');
        assert_eq!(scan_flat_keyed_neon(&d, 8), scan_flat_keyed(&d, 8));
        assert_eq!(scan_flat_keyed_neon(&d, 17), scan_flat_keyed(&d, 17));
    }

    /// [`scan_flat_keyed_win`] is a copy of [`scan_flat_keyed`] with a different load, kept
    /// separate so that adding it cannot move the code v1..v12 already compile to. Copies
    /// drift, so the equality is a test rather than a comment — over every name length the
    /// window can see, every real station, and a window with no `;` at all.
    #[test]
    fn win_agrees_with_the_slice_form() {
        let check = |d: &[u8]| {
            let w: &[u8; 16] = d[..16].try_into().unwrap();
            let want = scan_flat_keyed(d, 0).map(|(semi, h, klo, khi)| (semi, h, klo, khi));
            let got = scan_flat_keyed_win(w).map(|(len, h, klo, khi)| (len, h, klo, khi));
            assert_eq!(got, want, "{:?}", &d[..16]);
        };

        for len in 0..=100usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            check(&line(&name));
        }
        for (name, _) in STATIONS {
            check(&line(name));
        }
        check(&[b'x'; 32]);
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

    /// [`line_len_flat`] replaces the dot search as the thing that decides where the next row
    /// begins, so the only property that matters is that the two never disagree. Check it at
    /// every name length the flat path handles, against every shape a value can take, with the
    /// bytes after the row varied — a scan that ran past its own terminator would show up here.
    #[test]
    fn line_len_matches_the_dot_search() {
        use crate::parse::parse_temp_branchless;

        for len in 0..16usize {
            let name: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            for value in ["1.2", "-1.2", "12.3", "-12.3", "0.0", "-0.0", "99.9", "-99.9"] {
                for filler in [b'\n', b'x', b'0', b';', 0x00, 0xFF] {
                    let row = format!("{name};{value}\n");
                    let want = row.len() - 1;
                    let mut d = row.into_bytes();
                    d.resize(d.len() + 40, filler);
                    let case = format!("{name:?};{value} filler {filler:#04x}");
                    assert_eq!(line_len_flat(&d, 0), want, "{case}");
                    assert_eq!(parse_temp_branchless(&d, len + 1).1, want + 1, "{case}");
                }
            }
        }

        // `line` appends ";12.3\n", so the terminator sits at name.len() + 5.
        for (name, _) in STATIONS {
            if name.len() < 16 {
                assert_eq!(line_len_flat(&line(name), 0), name.len() + 5, "{name}");
            }
        }
    }

    /// The two cases the caller has to discard: a row too long to end inside the window, and a
    /// window with no `\n` in it at all. Both must come back as the sentinel rather than as
    /// some offset that happens to be in range. The pair at 18 and 19 straddles the seam.
    #[test]
    fn line_len_reports_the_sentinel_when_the_row_does_not_end_in_the_window() {
        assert_eq!(line_len_flat(&line(&"x".repeat(18)), 0), LINE_SLACK - 1);
        assert_eq!(line_len_flat(&line(&"x".repeat(19)), 0), LINE_SLACK);
        assert_eq!(line_len_flat(&line(&"x".repeat(40)), 0), LINE_SLACK);
        assert_eq!(line_len_flat(&[b'x'; 64], 0), LINE_SLACK);
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
    }

    /// The table takes its slot from the *high* bits, so that is where the spread has to be.
    /// `find_semi_and_hash_flat_raw` skips the avalanche and relies on this being true of a
    /// bare multiply — the assertion is the whole justification for dropping it, so it has to
    /// hold over more than this dataset. Two name sets: the 413 official ones, and 10,000
    /// synthetic ones sharing a long prefix, which is the shape the spec permits and the one
    /// most likely to defeat a hash that only mixes upward. Bit widths run to 18 because the
    /// table doubles on demand and 10,000 names reach 2^15.
    #[test]
    fn high_bits_spread_across_every_table_size() {
        let official: Vec<String> = STATIONS.iter().map(|(n, _)| n.to_string()).collect();
        let adversarial: Vec<String> =
            (0..10_000).map(|i| format!("Sankt_Peterburg_Oblast_{i:05}")).collect();

        for (names, set) in [(&official, "official"), (&adversarial, "prefixed")] {
            for bits in 11..=18u32 {
                let slots = 1usize << bits;
                for (label, hash) in [
                    ("finalized", find_semi_and_hash as fn(&[u8], usize) -> (usize, u64)),
                    ("raw", find_semi_and_hash_flat_raw),
                ] {
                    let buckets: HashSet<u64> =
                        names.iter().map(|n| hash(&line(n), 0).1 >> (64 - bits)).collect();
                    // n names into `slots` buckets collide ~n^2/(2*slots) times by chance.
                    // Allow twice that and a floor of 5 for the small-n cases.
                    let expected = names.len().pow(2) / (2 * slots);
                    let floor = names.len().saturating_sub(2 * expected + 5);
                    assert!(
                        buckets.len() >= floor,
                        "{set}/{label} at {bits} bits: {} slots for {} names, wanted {floor}",
                        buckets.len(),
                        names.len(),
                    );
                }
            }
        }
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
