//! Scalar temperature parsing, shared by v2/v2b/v3. v4 replaces this with a branchless
//! variant that must agree with it exactly.

/// Parses the temperature starting at `p` (the first byte after `;`) into tenths of a
/// degree, and returns the index of the next line's first byte.
///
/// Relies only on the guarantees in the spec: an optional `-`, one or two integer digits,
/// a `.`, and exactly one fractional digit. The returned index may be one past the end of
/// the data when the file has no trailing newline; the caller's bound check handles that.
#[inline]
pub fn parse_temp(d: &[u8], p: usize) -> (i16, usize) {
    let neg = d[p] == b'-';
    let mut p = p + neg as usize;

    let mut v = (d[p] - b'0') as i16;
    p += 1;
    if d[p] != b'.' {
        v = v * 10 + (d[p] - b'0') as i16;
        p += 1;
    }
    // d[p] is '.', d[p + 1] is the fractional digit, d[p + 2] is '\n'.
    v = v * 10 + (d[p + 1] - b'0') as i16;

    (if neg { -v } else { v }, p + 3)
}

/// Branchless equivalent of [`parse_temp`].
///
/// The scalar version branches on the sign and on whether there are one or two integer
/// digits. Both are data-dependent and unpredictable, costing a misprediction on a large
/// fraction of rows. This version has no data-dependent branches at all.
///
/// How it works. Load 8 bytes little-endian at the value start; the layout is one of
/// `d.d\n`, `dd.d\n`, `-d.d\n`, `-dd.d\n`. ASCII digits `0x30..=0x39` all have bit `0x10`
/// set, while `.` (`0x2E`), `-` (`0x2D`) and `\n` (`0x0A`) do not — so the lowest set bit
/// of `!word & 0x10101000` locates the `.`, which can only be at byte 1, 2 or 3. Shifting
/// left by `28 - dot_bit` moves the `.` to byte 3 in every layout, which pins the tens
/// digit to byte 1, the units to byte 2 and the fraction to byte 4. The sign byte is
/// cleared before the shift so it cannot masquerade as a tens digit.
///
/// Reads 8 bytes from `p`, so the caller must guarantee that much slack.
#[inline]
pub fn parse_temp_branchless(d: &[u8], p: usize) -> (i16, usize) {
    let word = u64::from_le_bytes(d[p..p + 8].try_into().unwrap());
    let inv = !word;

    // Bit position of the `.`: 12, 20 or 28 for byte 1, 2 or 3.
    let dot_bit = (inv & 0x1010_1000).trailing_zeros();

    // All ones when byte 0 is '-', otherwise zero.
    let signed = ((inv << 59) as i64) >> 63;

    let masked = word & !((signed as u64) & 0xFF);
    let v = masked << (28 - dot_bit);

    let tens = (v >> 8) & 0x0F;
    let ones = (v >> 16) & 0x0F;
    let frac = (v >> 32) & 0x0F;
    let abs = (tens * 100 + ones * 10 + frac) as i64;

    // Negate without branching: (x ^ -1) - -1 == -x.
    let value = (abs ^ signed) - signed;

    (value as i16, p + (dot_bit as usize >> 3) + 3)
}

/// [`parse_temp_branchless`] taking the 8 bytes it reads instead of a slice and an index.
///
/// Identical arithmetic; only the load differs. `d[p..p + 8].try_into().unwrap()` costs the
/// caller two compares and two branches to a panic, because nothing in the types relates `p`
/// to `d.len()`. An `&[u8; 8]` carries that proof in the type, so the window is fetched once
/// — one compare in `first_chunk` — and the parse itself checks nothing.
///
/// Returns the value and how far past the window start the next line begins, which is what
/// [`parse_temp_branchless`] adds to `p`.
///
/// The body is duplicated rather than shared: routing the slice form through this one would
/// change the code every shipped version up to v12 compiles to, and the point of the
/// experiment is to move exactly one thing. `win_agrees_with_the_slice_form` holds them equal.
#[inline]
pub fn parse_temp_branchless_win(w: &[u8; 8]) -> (i16, usize) {
    let word = u64::from_le_bytes(*w);
    let inv = !word;

    let dot_bit = (inv & 0x1010_1000).trailing_zeros();
    let signed = ((inv << 59) as i64) >> 63;

    let masked = word & !((signed as u64) & 0xFF);
    let v = masked << (28 - dot_bit);

    let tens = (v >> 8) & 0x0F;
    let ones = (v >> 16) & 0x0F;
    let frac = (v >> 32) & 0x0F;
    let abs = (tens * 100 + ones * 10 + frac) as i64;

    let value = (abs ^ signed) - signed;

    (value as i16, (dot_bit as usize >> 3) + 3)
}

/// [`parse_temp_branchless`] with the three digits folded into one multiply.
///
/// The digit combine above is the parser's last piece of long division. After the shift the
/// three digits sit at bytes 1, 2 and 4, and it extracts them one at a time and reassembles
/// them with two chained multiply-accumulates:
///
/// ```text
///   ubfx w13, x12, #8,  #4      ubfx w14, x12, #16, #4      ubfx w15, x12, #32, #4
///   mov  w16, #0xa              umaddl x14, w14, w16, x15
///   mov  w16, #0x64             umaddl x13, w13, w16, x14
/// ```
///
/// Seven instructions, two serialised multiplies, and two constants that have to live
/// somewhere — and on this loop they do not fit, so LLVM rebuilds `#10` and `#100` inside the
/// body and evicts `-LOW` to make room (see the v15 section of the README).
///
/// One multiply does the whole thing. Masking the shifted word leaves
/// `d1 << 8 | d2 << 16 | d3 << 32`, and multiplying by `0x640A0001` — which is
/// `100 << 24 | 10 << 16 | 1` — lines each digit's weight up with its byte position so that
/// `d1*100 + d2*10 + d3` lands together at bit 32:
///
/// ```text
///   and x13, x12, MASK          mul x13, x13, MUL
///   lsr x13, x13, #32           and x13, x13, #0x3ff
/// ```
///
/// The `& 0x3FF` is not a range assertion, it is load-bearing. The product also carries
/// `d2*100` at bit 40 and `d3*10`/`d3*100` above that, and they vanish only because
/// `100 << 8`, `10 << 16` and `100 << 24` are all multiples of 1024. Below bit 32 the largest
/// term is `d1*10 << 24` ≤ `0x5A000000`, so nothing carries up into the answer either.
///
/// Four instructions for seven, one multiply for two, and the two constants it needs are
/// loop-invariant instead of rematerialised per row.
///
/// This is royvanrijn's own combine; the project has been carrying the expanded form since v5.
/// It went unmeasured for as long as the cost model said the value branch could not matter,
/// which [`v16_newline`](../v16_newline/index.html) has since disproved.
#[inline]
pub fn parse_temp_branchless_mul(d: &[u8], p: usize) -> (i16, usize) {
    let word = u64::from_le_bytes(d[p..p + 8].try_into().unwrap());
    let inv = !word;

    let dot_bit = (inv & 0x1010_1000).trailing_zeros();
    let signed = ((inv << 59) as i64) >> 63;

    let masked = word & !((signed as u64) & 0xFF);
    let v = masked << (28 - dot_bit);

    let digits = v & 0x0000_000F_000F_0F00;
    let abs = ((digits.wrapping_mul(0x640A_0001) >> 32) & 0x3FF) as i64;

    let value = (abs ^ signed) - signed;

    (value as i16, p + (dot_bit as usize >> 3) + 3)
}

/// [`parse_temp_branchless`] for a caller that already knows where the line ends.
///
/// `len` is the byte count from `p` to the `\n`, which a delimiter bitmap yields for free.
/// The legal layouts are `d.d`, `dd.d`, `-d.d` and `-dd.d`, and in every one the `.` sits at
/// byte `len - 2` — including the two that share `len == 4`. The dot search therefore
/// collapses to `28 - dot_bit` = `40 - 8 * len`, one subtract that does not wait on the load.
///
/// Reads 8 bytes from `p`, as [`parse_temp_branchless`] does. Returns no next index: the
/// caller already has it.
#[inline]
pub fn parse_temp_len(d: &[u8], p: usize, len: usize) -> i16 {
    let word = u64::from_le_bytes(d[p..p + 8].try_into().unwrap());

    let signed = ((!word << 59) as i64) >> 63;
    let masked = word & !((signed as u64) & 0xFF);
    let v = masked << (40 - 8 * len as u32);

    let tens = (v >> 8) & 0x0F;
    let ones = (v >> 16) & 0x0F;
    let frac = (v >> 32) & 0x0F;
    let abs = (tens * 100 + ones * 10 + frac) as i64;

    ((abs ^ signed) - signed) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every legal temperature string, -99.9 ..= 99.9, is only 1999 cases — so check all
    /// of them rather than sampling.
    pub fn all_legal_values() -> Vec<(String, i16)> {
        (-999i16..=999)
            .map(|t| {
                let s =
                    format!("{}{}.{}", if t < 0 { "-" } else { "" }, t.abs() / 10, t.abs() % 10);
                (s, t)
            })
            .collect()
    }

    #[test]
    fn parses_every_legal_value() {
        for (s, expected) in all_legal_values() {
            let line = format!("{s}\n");
            let (got, next) = parse_temp(line.as_bytes(), 0);
            assert_eq!(got, expected, "parsing {s:?}");
            assert_eq!(next, line.len(), "next index after {s:?}");
        }
    }

    /// The parser reads a fixed window; make sure whatever follows the line cannot change
    /// the result.
    #[test]
    fn trailing_bytes_do_not_affect_the_result() {
        for (s, expected) in all_legal_values() {
            for suffix in ["", "x", "Station;1.0\n", "\n\n\n", "999999"] {
                let line = format!("{s}\n{suffix}");
                let (got, next) = parse_temp(line.as_bytes(), 0);
                assert_eq!(got, expected, "parsing {s:?} followed by {suffix:?}");
                assert_eq!(next, s.len() + 1);
            }
        }
    }

    /// Pads to guarantee the 8-byte load stays in bounds, and lets the caller vary the
    /// bytes past the line so we prove they cannot leak into the result.
    fn padded(line: &str, filler: u8) -> Vec<u8> {
        let mut v = line.as_bytes().to_vec();
        v.resize(line.len() + 16, filler);
        v
    }

    #[test]
    fn branchless_agrees_with_scalar_on_every_legal_value() {
        for (s, expected) in all_legal_values() {
            for filler in [b'\n', b'x', b'0', b'9', b';', 0x00, 0xFF] {
                let d = padded(&format!("{s}\n"), filler);
                let scalar = parse_temp(&d, 0);
                let branchless = parse_temp_branchless(&d, 0);
                assert_eq!(branchless.0, expected, "value of {s:?} with filler {filler:#04x}");
                assert_eq!(branchless, scalar, "parsing {s:?} with filler {filler:#04x}");
            }
        }
    }

    /// [`parse_temp_branchless_win`] is a copy with a different load, kept separate so that
    /// adding it cannot move the code v1..v12 already compile to. Copies drift, so the
    /// equality is a test: every legal value, under every filler the window might see past
    /// the line.
    #[test]
    fn win_agrees_with_the_slice_form() {
        for (s, expected) in all_legal_values() {
            for filler in [b'\n', b'x', b'0', b'9', b';', 0x00, 0xFF] {
                let d = padded(&format!("{s}\n"), filler);
                let w: &[u8; 8] = d[..8].try_into().unwrap();
                let (value, advance) = parse_temp_branchless_win(w);
                assert_eq!(value, expected, "value of {s:?} with filler {filler:#04x}");
                assert_eq!(
                    (value, advance),
                    parse_temp_branchless(&d, 0),
                    "parsing {s:?} with filler {filler:#04x}"
                );
            }
        }
    }

    /// The magic multiply is only correct because three of the product's terms are multiples
    /// of 1024 and a fourth cannot carry into bit 32. That is an argument, not a proof, so
    /// check it the same way everything else here is checked: every legal value, every filler
    /// the 8-byte window might see past the line, against the form it replaces.
    #[test]
    fn magic_multiply_agrees_with_the_expanded_combine() {
        for (s, expected) in all_legal_values() {
            for filler in [b'\n', b'x', b'0', b'9', b';', 0x00, 0xFF] {
                let d = padded(&format!("{s}\n"), filler);
                let got = parse_temp_branchless_mul(&d, 0);
                assert_eq!(got.0, expected, "value of {s:?} with filler {filler:#04x}");
                assert_eq!(got, parse_temp_branchless(&d, 0), "{s:?}/{filler:#04x}");
            }
        }
    }

    #[test]
    fn magic_multiply_parses_at_a_nonzero_offset() {
        let d = padded("Abha;-12.3\n", b'x');
        assert_eq!(parse_temp_branchless_mul(&d, 5), (-123, 11));
    }

    /// The length-driven form drops the dot search on the claim that `len - 2` always names
    /// the same byte. That claim is the whole optimisation, so check it against the branchless
    /// parser on every legal value and every filler, not just the four layouts.
    #[test]
    fn length_driven_agrees_with_the_dot_search() {
        for (s, expected) in all_legal_values() {
            for filler in [b'\n', b'x', b'0', b'9', b';', 0x00, 0xFF] {
                let d = padded(&format!("{s}\n"), filler);
                let got = parse_temp_len(&d, 0, s.len());
                assert_eq!(got, expected, "value of {s:?} with filler {filler:#04x}");
                assert_eq!(got, parse_temp_branchless(&d, 0).0, "{s:?}/{filler:#04x}");
            }
        }
    }

    #[test]
    fn length_driven_parses_at_a_nonzero_offset() {
        let d = padded("Abha;-12.3\n", b'x');
        assert_eq!(parse_temp_len(&d, 5, 5), -123);
    }

    #[test]
    fn branchless_parses_at_a_nonzero_offset() {
        let d = padded("Abha;-12.3\n", b'x');
        assert_eq!(parse_temp_branchless(&d, 5), (-123, 11));
    }

    #[test]
    fn parses_at_a_nonzero_offset() {
        let d = b"Abha;-12.3\nx";
        let (v, next) = parse_temp(d, 5);
        assert_eq!(v, -123);
        assert_eq!(next, 11);
    }

    #[test]
    fn boundary_values() {
        assert_eq!(parse_temp(b"-99.9\n", 0).0, -999);
        assert_eq!(parse_temp(b"99.9\n", 0).0, 999);
        assert_eq!(parse_temp(b"0.0\n", 0).0, 0);
        assert_eq!(parse_temp(b"-0.0\n", 0).0, 0);
    }
}
