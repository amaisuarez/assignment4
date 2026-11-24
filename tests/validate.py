#!/usr/bin/env python3
"""Correctness checks for Assignment 4, run BEFORE any benchmark.

Adapted from tests/validate.py of Assignment 3 (distributed checks added).

1. Known answers: identity, zero and a hand-computed 2x2, for the local methods
   and for every tested cluster configuration.
2. Sequential baseline vs a reference library: `seq` is compared with NumPy's
   A @ B (reference only) using the forward error bound of a dot product of
   length n (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
   Sec. 3.5):  |C - C_exact| <= gamma_n (|A| |B|),  gamma_n = n u / (1 - n u),
   u = 2^-53. Both results satisfy it, so we accept |C - C_numpy| <= 2 gamma_n (|A||B|).
3. Distributed == sequential, BIT FOR BIT, for several cluster configurations
   (m members x t threads) and sizes, including n smaller than the number of
   blocks (empty blocks), n not divisible by the number of blocks, and m = 3.
   Every C[i][j] is the same sum in the same k order, so any difference reveals a
   partitioning, transfer or assembly error (a missing, duplicated or misplaced row).
4. Placement and transfers: every task ran on the member that stores its A block
   (a_local = true), each member fetched B exactly once per multiplication, and
   the remote B bytes it reported equal 8 n (n - rows of its own B panel).
5. Repeatability: all multiplications of one driver process (warm-ups and
   measured) give the same checksum as `seq`.
6. Inputs: dense files are byte-identical to the Assignment 1-3 inputs (SHA-256
   in inputs/manifest.csv).

Usage: python3 tests/validate.py      (after ./build.sh)
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_matrices  # noqa: E402
import run_benchmarks as rb  # noqa: E402

TMP = ROOT / "tests" / "tmp"
CFG = json.load(open(ROOT / "config.json"))
rb.cfg_global.update(CFG)
TMP.mkdir(parents=True, exist_ok=True)
(TMP / "stderr").mkdir(exist_ok=True)
U = 2.0 ** -53
CLUSTERS = [(1, 1), (2, 1), (2, 2), (3, 1)]
failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def write_dense(path, M):
    np.ascontiguousarray(M, dtype="<f8").tofile(path)


def run_local(method, n, pa, pb, p=1):
    out = TMP / f"C_{method}_{n}_{p}.bin"
    cmd = (["java"] + CFG["java_opts"] + ["-cp", rb.CP, "LocalMatMul", "--method", method, "--n", str(n),
           "--threads", str(p), "--a", str(pa), "--b", str(pb), "--warmup", "0", "--reps", "1",
           "--batch", "1", "--out", str(out)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr}")
    chk = [l.split(",")[4] for l in r.stdout.splitlines() if l.startswith("measured")]
    return np.fromfile(out, dtype="<f8").reshape(n, n), chk


class Cluster:
    def __init__(self, m, t):
        self.m, self.t = m, t
        self.name = f"a4-validate-{m}-{t}"
        self.procs, self.members, start = rb.start_members(CFG, m, t, self.name, TMP, f"m{m}_t{t}")
        if start is None:
            rb.stop_members(self.procs)
            raise RuntimeError(f"cluster m={m} t={t} did not start (see tests/tmp/stderr)")

    def multiply(self, n, pa, pb, reps=1):
        out = TMP / f"C_dist_{n}_{self.m}_{self.t}.bin"
        cmd = (["java"] + CFG["hz_jvm_opts"] + CFG["driver_java_opts"]
               + ["-cp", rb.CP, "DistMatMul", "--cluster", self.name, "--members", self.members,
                  "--expect-members", str(self.m), "--threads-per-member", str(self.t), "--n", str(n),
                  "--a", str(pa), "--b", str(pb), "--warmup", "1", "--reps", str(reps), "--out", str(out)])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr[-3000:]}")
        iters = [l.split(",") for l in r.stdout.splitlines() if l.startswith("iter,")]
        tasks = [l.split(",") for l in r.stdout.splitlines() if l.startswith("task,")]
        return np.fromfile(out, dtype="<f8").reshape(n, n), iters, tasks

    def close(self):
        rb.stop_members(self.procs)


def placement_ok(n, m, tasks):
    """Check 4 for every multiplication in `tasks`."""
    bounds = [n * i // m for i in range(m + 1)]
    own_rows = sorted(bounds[i + 1] - bounds[i] for i in range(m))
    by_iter = defaultdict(list)
    for f in tasks:
        by_iter[(f[1], f[2])].append(f)
    problems = []
    for key, ts in by_iter.items():
        if not all(f[7] == "true" for f in ts):
            problems.append(f"{key}: task not on owner of its A block")
        fetch = defaultdict(list)
        for f in ts:
            if f[8] == "true":
                fetch[f[4]].append(int(f[14]))
        members_with_tasks = {f[4] for f in ts}
        if set(fetch) != members_with_tasks or any(len(v) != 1 for v in fetch.values()):
            problems.append(f"{key}: B fetched {dict((k, len(v)) for k, v in fetch.items())} times per member")
        else:
            remote = sorted(8 * n * (n - r) for r in own_rows)
            if sorted(v[0] for v in fetch.values()) != remote:
                problems.append(f"{key}: remote B bytes {sorted(v[0] for v in fetch.values())} != {remote}")
    return problems


def main():
    if not (ROOT / "java" / "build" / "DistMatMul.class").exists():
        sys.exit("Build first: ./build.sh")

    known = {
        "identity x M": (np.eye(4), np.arange(16, dtype=float).reshape(4, 4) - 7.5),
        "M x identity": (np.arange(16, dtype=float).reshape(4, 4) * 0.25, np.eye(4)),
        "zero x M": (np.zeros((3, 3)), np.arange(9, dtype=float).reshape(3, 3)),
        "2x2 by hand": (np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]])),
    }
    expected = {"2x2 by hand": np.array([[19., 22.], [43., 50.]])}
    for name, (A, B) in known.items():
        write_dense(TMP / f"k_{name}_A.bin".replace(" ", "_"), A)
        write_dense(TMP / f"k_{name}_B.bin".replace(" ", "_"), B)

    rng = np.random.default_rng(20262027)
    rand = {}
    for n in [1, 3, 5, 7, 16, 37, 64, 131]:
        A, B = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
        pa, pb = TMP / f"r{n}_A.bin", TMP / f"r{n}_B.bin"
        write_dense(pa, A)
        write_dense(pb, B)
        rand[n] = (A, B, pa, pb)

    print("1. Known answers (local methods)")
    for name, (A, B) in known.items():
        pa = TMP / f"k_{name}_A.bin".replace(" ", "_")
        pb = TMP / f"k_{name}_B.bin".replace(" ", "_")
        want = expected.get(name, A @ B)
        for meth, p in [("seq", 1), ("static", 2), ("static", 4)]:
            C, _ = run_local(meth, A.shape[0], pa, pb, p)
            check(np.array_equal(C, want), f"{name}: {meth} p={p}")

    print("2. Sequential baseline vs NumPy (error bound)")
    seq = {}
    for n, (A, B, pa, pb) in rand.items():
        C, chk = run_local("seq", n, pa, pb)
        seq[n] = (C, chk[0])
        gamma = n * U / (1 - n * U)
        bound = 2 * gamma * (np.abs(A) @ np.abs(B))
        err = np.abs(C - A @ B)
        check(np.all(err <= bound), f"n={n:3d} seq vs NumPy: max err {err.max():.2e}, max allowed {bound.max():.2e}")

    print("1, 3-5. Distributed (Hazelcast, members on this machine)")
    for m, t in CLUSTERS:
        print(f"  -- cluster m={m} members x t={t} threads ({m * t} blocks)")
        cl = Cluster(m, t)
        try:
            for name, (A, B) in known.items():
                pa = TMP / f"k_{name}_A.bin".replace(" ", "_")
                pb = TMP / f"k_{name}_B.bin".replace(" ", "_")
                C, _, _ = cl.multiply(A.shape[0], pa, pb)
                check(np.array_equal(C, expected.get(name, A @ B)), f"{name}: m={m} t={t}")
            for n, (A, B, pa, pb) in rand.items():
                C, iters, tasks = cl.multiply(n, pa, pb, reps=2)
                check(np.array_equal(C, seq[n][0]), f"n={n:3d} m={m} t={t} == seq (bitwise)"
                      + (" [n < blocks: empty blocks]" if n < m * t else ""))
                sums = {f[7] for f in iters}
                check(sums == {seq[n][1]}, f"n={n:3d} m={m} t={t}: {len(iters)} multiplications, "
                                           f"checksums all equal to seq")
                if n >= m:
                    probs = placement_ok(n, m, tasks)
                    check(not probs, f"n={n:3d} m={m} t={t}: A local in all tasks, B fetched once per "
                                     f"member, remote bytes = 8n(n - own panel rows)"
                          + (f"  [{'; '.join(probs[:3])}]" if probs else ""))
        finally:
            cl.close()

    print("6. Input generators")
    for n in [16, 64]:
        try:
            gen_matrices.ensure(n, 20262027)
            check(True, f"dense n={n}: byte-identical to the Assignment 1-3 inputs (SHA-256)")
        except SystemExit as e:
            check(False, f"dense n={n}: {e}")

    print()
    if failures:
        print(f"{len(failures)} TEST(S) FAILED")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
