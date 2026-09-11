#!/usr/bin/env python3
"""Count the instructions in a binary's hot pair loop, by category.

Three versions in a row (v13, v15, v18) removed work at the source level and got it back as
rematerialised constants. That is visible in the disassembly and costs nothing to check, so it
is worth checking before spending a batch on it — v18's +7.5 instructions a row were known
before the benchmark ran.

Finding the loop without hand-picking addresses, and without a symbol name — `worker` is
inlined into an unnameable spawn closure. Disassemble every function, drop the instructions
only reachable from a `bl` (the `#[cold]` fixups, which otherwise get counted as loop body),
take every backward branch as a candidate loop, and keep the smallest one that looks like two
rows of this program: see `MIN_MUL` below.

Usage: scripts/loopcount.py target/release/v12_pairs [...]
"""

import re
import subprocess
import sys

CATEGORIES = [
    ("bounds", r"^(cmn|ccmp|ccmn)$"),
    ("bitfield", r"^(ubfx|ubfiz|sbfx|sbfiz|bfi|bfxil)$"),
    ("mul", r"^(mul|umull|umaddl|madd|smull)$"),
    ("const", r"^(mov|movk|movz|movn|orr)$"),
    ("load", r"^(ldr|ldp|ldrb|ldrh|ldrsh|ldrsw|ldur|ldurb|ld1|ldursw)$"),
    ("store", r"^(str|stp|strb|strh|stur|st1)$"),
    ("branch", r"^(b|bl|br|cbz|cbnz|tbz|tbnz|b\.\w+)$"),
]

#: `vector` and `xfer` are decided by the *operands*, not the mnemonic, and are tested before
#: the table above. Naming mnemonics was wrong twice and both times in the direction that
#: flattered the argument being made: `orr.16b` fell through to `const` and `ext.16b` to `other`,
#: putting eight of v22's vector operations in the integer column, and `dup.16b v1, w4` — a
#: general-purpose register being written *into* the vector file — was filed as plain vector work
#: while the whole v21 write-up turned on counting exactly that. There are too many spellings of
#: a NEON instruction to enumerate, and only one structural question worth asking: which register
#: files does this instruction touch?
#:
#: So: an instruction naming both a GPR and a vector register moves data between the two files
#: and is a crossing, and one naming only vector registers is vector work. Loads, stores and
#: branches are settled first, because `ldr q0, [x8]` names both and crosses nothing.
GPR = re.compile(r"\b[wx](\d+|zr|sp)\b")
VEC = re.compile(r"\b[vqdsh]\d+\b")

BACKWARD = re.compile(r"^b(\.\w+)?$|^cb(n)?z$|^tb(n)?z$")

#: What makes a loop *the* loop, without naming addresses or counting on a particular version's
#: shape. It hashes twice a row, so four multiplies; it finds a `;` and a `.` per row, so four
#: `ctz`; and it writes four stats fields per row, so it stores.
#:
#: The counts started as `ctz == 6` — the scan's two plus the parser's one, twice — and that was
#: a v12 fact rather than a hot-loop fact. v21 gets the delimiter index off a single `shrn`
#: extract and so has four, fell through the filter, and the smallest match in the binary turned
#: out to be an archive-header parser inside `std`'s backtrace machinery. Requiring a store is
#: what excludes that one and `memchr`, neither of which writes anything.
MIN_MUL = 4
MIN_CTZ = 4
MIN_STORE = 1


def disassemble(path):
    """Every function, as a list of (address, mnemonic, operands)."""
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", path],
        capture_output=True, text=True, check=True,
    ).stdout
    funcs, current = [], None
    for line in out.splitlines():
        if re.match(r"^([0-9a-f]{8,16}) <(.+)>:$", line.strip()):
            current = []
            funcs.append(current)
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m and current is not None:
            current.append((int(m.group(1), 16), m.group(2), m.group(3).strip()))
    return [f for f in funcs if f]


BRANCH = re.compile(r"^(b|bl|br|ret|cb(n)?z|tb(n)?z|b\.\w+)$")


def target_of(op, operands):
    """The target is the last operand — after dropping objdump's `<symbol+0x..>` annotation,
    and only for branches, since `cmp x9, #0x9` would otherwise look like one."""
    if not BRANCH.match(op):
        return None
    hexes = re.findall(r"0x([0-9a-f]+)", re.sub(r"<[^>]*>", "", operands))
    return int(hexes[-1], 16) if hexes else None


def reachable_without_call(insns):
    """Addresses reachable from the entry by fallthrough and branches, not counting `bl`
    returns. Blocks only reachable that way are the out-of-line cold paths."""
    by_addr = {a: i for i, (a, _, _) in enumerate(insns)}
    seen, stack = set(), [insns[0][0]]
    while stack:
        a = stack.pop()
        if a in seen or a not in by_addr:
            continue
        seen.add(a)
        _, op, ops = insns[by_addr[a]]
        t = target_of(op, ops) if op != "bl" else None
        if t is not None and t in by_addr:
            stack.append(t)
        if op not in ("b", "br", "ret"):
            nxt = by_addr[a] + 1
            if nxt < len(insns):
                stack.append(insns[nxt][0])
    return seen


def loops(insns, live):
    """Every backward branch, as (start, end, body)."""
    body = [(a, op, ops) for (a, op, ops) in insns if a in live]
    out = []
    for a, op, ops in body:
        if not BACKWARD.match(op):
            continue
        t = target_of(op, ops)
        if t is None or t >= a:
            continue
        span = [x for x in body if t <= x[0] <= a]
        out.append((t, a, span))
    return out


def hot_loop(path):
    cands = []
    for insns in disassemble(path):
        live = reachable_without_call(insns)
        for cand in loops(insns, live):
            ops = [op.split(".")[0] for _, op, _ in cand[2]]
            if (
                ops.count("ctz") >= MIN_CTZ
                and sum(o.startswith("mul") for o in ops) >= MIN_MUL
                and sum(o in ("str", "stp", "strb", "strh", "stur") for o in ops) >= MIN_STORE
            ):
                cands.append(cand)
    return min(cands, key=lambda l: len(l[2])) if cands else None


def classify(op, operands):
    """One bucket per instruction. Register files first, mnemonics second."""
    base = op.split(".")[0]
    for name, pat in CATEGORIES:
        if name in ("load", "store", "branch") and (re.match(pat, op) or re.match(pat, base)):
            return name
    # objdump annotates branch and literal targets with `<symbol+0x..>`; those are not registers.
    bare = re.sub(r"<[^>]*>", "", operands)
    if VEC.search(bare):
        return "xfer" if GPR.search(bare) else "vector"
    for name, pat in CATEGORIES:
        if re.match(pat, op) or re.match(pat, base):
            return name
    return "other"


def report(path):
    found = hot_loop(path)
    if not found:
        print(f"{path}: nothing shaped like two rows of this program")
        return
    start, end, span = found

    counts = {}
    for _, op, operands in span:
        counts[classify(op, operands)] = counts.get(classify(op, operands), 0) + 1

    n = len(span)
    print(f"{path.split('/')[-1]:16} {start:#x}..{end:#x}  {n} insns / 2 rows = {n / 2:.1f} per row")
    order = [k for k, _ in CATEGORIES] + ["vector", "xfer", "other"]
    print("   " + "  ".join(f"{k} {counts.get(k, 0) / 2:.1f}" for k in order))

    # The column every cost model in the README is fitted against: everything that occupies an
    # integer ALU slot, which is the total less the work that occupies something else.
    elsewhere = sum(counts.get(k, 0) for k in ("load", "store", "branch", "vector", "xfer"))
    print(f"   integer {(n - elsewhere) / 2:.1f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
