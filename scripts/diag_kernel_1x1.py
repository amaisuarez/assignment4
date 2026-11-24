#!/usr/bin/env python3
"""Diagnostic run (report Section 5.2): kernel time of one Hazelcast member with one
thread, n = 2048, with the default settings, with 4 warm-ups, and with a 4 GB heap.
Usage: python3 scripts/diag_kernel_1x1.py results/diag_kernel_1x1   (after the study)"""
import sys, json, subprocess, statistics
sys.path.insert(0, "scripts")
import run_benchmarks as rb
from pathlib import Path
cfg = json.load(open("config.json")); rb.cfg_global.update(cfg)
out = Path(sys.argv[1]); (out/"stderr").mkdir(parents=True, exist_ok=True)
for label, mopts, W in [("default", cfg["member_java_opts"], 1), ("default_W4", cfg["member_java_opts"], 4),
                        ("xmx4g", ["-XX:+UseSerialGC","-Xms256m","-Xmx4g"], 1)]:
    c = dict(cfg); c["member_java_opts"] = mopts
    procs, ml, s = rb.start_members(c, 1, 1, f"diag-{label}", out, label)
    cmd = ["java"] + cfg["hz_jvm_opts"] + cfg["driver_java_opts"] + ["-cp", rb.CP, "DistMatMul", "--cluster", f"diag-{label}",
           "--members", ml, "--expect-members", "1", "--threads-per-member", "1", "--n", "2048",
           "--a", "data/A_n2048.bin", "--b", "data/B_n2048.bin", "--warmup", str(W), "--reps", "3"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rb.stop_members(procs)
    k = [int(l.split(",")[12])/1e9 for l in r.stdout.splitlines() if l.startswith("task,")]
    print(label, "kernel per multiplication (warm-ups first):", [round(x,2) for x in k], flush=True)
