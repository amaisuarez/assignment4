#!/usr/bin/env python3
"""Benchmark harness for Assignment 4 (adapted from the Assignment 3 harness).

Three experiments, all Java, float64, the same i-k-j row kernel and the same inputs
as Assignments 1-3:

  local   single-process references, measured in this same session:
          `seq` (Assignment 2/3 sequential kernel) and `static` (Assignment 3
          thread pool, p threads). Driver: java/LocalMatMul.java.
  strong  strong scaling of the distributed version (Hazelcast): n fixed, the
          resources change. A configuration is (m members, t threads per member);
          m = number of member processes (JVMs), t = executor threads in each, so
          workers = m * t. All members run on this one machine (see README).
  weak    weak scaling: t = 1 and m = W members, with n chosen so that the
          arithmetic work per worker stays constant: 2 n^3 / W = 2 n0^3, i.e.
          n = round(n0 * W^(1/3)). (Keeping n per worker fixed would NOT give equal
          work, because the work grows with n^3.)

For every distributed configuration a NEW cluster is started: m member JVMs on
ports base_port .. base_port + m - 1 with a unique cluster name. When all members
have printed "#member_ready" the driver JVM (a Hazelcast client) is started; it
waits until the cluster is complete and safe, does W warm-up multiplications and
R measured ones, and prints the phase timings (see java/DistMatMul.java). Then
the members are terminated and their peak RSS is read with os.wait4().

Budgets (declared before running, in the config):
  * a process is killed if one output line (one multiplication) takes longer than
    per_repetition_budget_s (+ a start-up allowance for the first line) -> "timeout";
  * if the estimated memory of all processes exceeds memory_budget_bytes the run
    is not started -> "skipped_memory_budget".
  After a failure, larger sizes are not attempted for that configuration. No time is
  ever invented for a failed run.

Output: results/run_<timestamp>/
  raw_measurements.csv  one row per warm-up / measured multiplication (all experiments)
  tasks.csv             one row per distributed task (block) and multiplication
  members.csv           one row per member process and configuration (memory, CPU, start-up)
  runs.csv              one row per configuration (status, driver memory, command ...)
  environment.json, config.json, stderr/
"""
import argparse
import csv
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import collect_env  # noqa: E402
import gen_matrices  # noqa: E402

CP = f"{ROOT / 'java' / 'build'}{os.pathsep}{ROOT / 'lib' / 'hazelcast-5.5.0.jar'}"
RSS_SCALE = 1 if sys.platform == "darwin" else 1024      # ru_maxrss: bytes on macOS, KiB on Linux

RAW_FIELDS = ["run_id", "experiment", "method", "n", "members", "threads_per_member", "workers",
              "dtype", "kind", "rep_index", "elapsed_ns", "batch", "time_per_mult_s", "scatter_s",
              "compute_s", "gather_s", "checksum", "status"]
TASK_FIELDS = ["run_id", "experiment", "n", "members", "threads_per_member", "kind", "rep_index",
               "block", "member", "thread", "rows", "a_local", "fetched_b", "read_a_ns", "get_b_ns",
               "fetch_b_ns", "kernel_ns", "put_c_ns", "remote_b_bytes"]
MEMBER_FIELDS = ["run_id", "experiment", "n", "members", "threads_per_member", "member_index", "port",
                 "pid", "address", "startup_s", "peak_rss_bytes", "process_user_s", "process_sys_s",
                 "processors", "heap_used_bytes", "heap_used_after_gc_bytes", "heap_max_bytes",
                 "entry_cost_A_bytes", "entry_cost_B_bytes", "entry_cost_C_bytes", "live_threads",
                 "peak_threads", "exit_code"]
RUN_FIELDS = ["run_id", "experiment", "method", "n", "members", "threads_per_member", "workers",
              "oversubscribed", "status", "exit_code", "warmups", "repetitions", "batch", "blocks",
              "cluster_start_s", "connect_ns", "key_placement_ns", "scatter_payload_bytes",
              "gather_payload_bytes", "b_redistribution_payload_bytes", "tasks_with_local_a",
              "tasks_total", "array_storage_bytes", "estimated_bytes", "driver_peak_rss_bytes",
              "driver_heap_used_bytes", "driver_peak_thread_count", "members_peak_rss_sum_bytes",
              "memory_metric", "process_user_s", "process_sys_s", "wall_time_s", "line_timeout_s",
              "order_in_group", "command"]


def warmups_for(cfg, n):
    for rule in cfg["warmup_rules"]:
        if n <= rule["max_n"]:
            return rule["warmups"]
    return cfg["warmup_rules"][-1]["warmups"]


def estimate_bytes(n, m):
    """Planning estimate (bytes) of the dense float64 data held by all processes.

    local (m = 0): A, B, C                                    -> 3 * 8n^2
    distributed:   driver A, B, C + its block copies (6 * 8n^2) and, over all
                   members, the stored A, B, C blocks, one assembled B per member,
                   the fetched A blocks and new C blocks ((5 + m) * 8n^2)."""
    e = 8 * n * n
    return 3 * e if m == 0 else (6 + 5 + m) * e


def start_members(cfg, m, t, cluster, out_dir, tag):
    """Start m member JVMs; return list of dicts with the Popen and start-up time."""
    host, base = cfg["host"], cfg["base_port"]
    member_list = ",".join(f"{host}:{base + i}" for i in range(m))
    procs = []
    t0 = time.monotonic()
    for i in range(m):
        cmd = (["java"] + cfg["hz_jvm_opts"] + cfg["member_java_opts"]
               + ["-cp", CP, "ClusterMember", "--cluster", cluster, "--host", host,
                  "--port", str(base + i), "--threads", str(t), "--members", member_list])
        err = open(out_dir / "stderr" / f"{tag}_member{i}.txt", "w")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        procs.append({"proc": p, "err": err, "port": base + i, "ready": None, "line": None, "cmd": cmd})

    def wait_ready(rec):
        for line in rec["proc"].stdout:
            if line.startswith("#member_ready"):
                rec["ready"] = time.monotonic() - t0
                rec["line"] = line.strip()
                break
        # keep draining stdout so the member never blocks on a full pipe
        for _ in rec["proc"].stdout:
            pass

    for rec in procs:
        threading.Thread(target=wait_ready, args=(rec,), daemon=True).start()
    deadline = time.monotonic() + cfg["cluster_start_timeout_s"]
    while time.monotonic() < deadline and any(r["ready"] is None for r in procs):
        if any(r["proc"].poll() is not None for r in procs):
            break
        time.sleep(0.05)
    ok = all(r["ready"] is not None for r in procs)
    return procs, member_list, (time.monotonic() - t0) if ok else None


def stop_members(procs):
    """Terminate the members and collect their rusage (peak RSS, CPU)."""
    for r in procs:
        if r["proc"].poll() is None:
            r["proc"].send_signal(signal.SIGTERM)
    out = []
    for r in procs:
        deadline = time.monotonic() + 30
        while True:
            pid, status, usage = os.wait4(r["proc"].pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() > deadline:
                r["proc"].kill()
                pid, status, usage = os.wait4(r["proc"].pid, 0)
                break
            time.sleep(0.05)
        r["proc"].returncode = os.waitstatus_to_exitcode(status)
        r["err"].close()
        out.append({"peak_rss_bytes": usage.ru_maxrss * RSS_SCALE, "process_user_s": round(usage.ru_utime, 3),
                    "process_sys_s": round(usage.ru_stime, 3), "exit_code": r["proc"].returncode})
    time.sleep(cfg_global["port_release_pause_s"])
    return out


def run_process(cmd, line_budget_s, startup_s, stderr_path):
    """Run cmd; kill it if one output line takes longer than the budget.

    Returns (stdout_lines, exit_code, timed_out, rusage, wall_s). (From Assignment 3.)"""
    lines = []
    last = [time.monotonic()]
    with open(stderr_path, "w") as err:
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)

        def reader():
            for line in p.stdout:
                lines.append(line.rstrip("\n"))
                last[0] = time.monotonic()

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        timed_out = False
        while th.is_alive():
            th.join(0.2)
            started = any(not l.startswith("#") for l in lines)
            allowed = line_budget_s + (0 if started else startup_s)
            if th.is_alive() and time.monotonic() - last[0] > allowed:
                timed_out = True
                p.kill()
                th.join()
        _, status, usage = os.wait4(p.pid, 0)
        wall = time.monotonic() - t0
        p.returncode = os.waitstatus_to_exitcode(status)
        p.stdout.close()
    return lines, p.returncode, timed_out, usage, wall


def jobs(cfg):
    """Ordered list of groups (one per experiment block and size), shuffled inside."""
    rng = random.Random(cfg["seed"])
    groups = []
    loc, strong = cfg.get("local"), cfg.get("strong")
    sizes = sorted(set((loc or {}).get("sizes", [])) | set((strong or {}).get("sizes", [])))
    for n in sizes:
        g = []
        if loc and n in loc["sizes"]:
            g += [dict(experiment="local", method=m, n=n, members=0, tpm=p) for m, p in loc["methods"]]
        if strong and n in strong["sizes"]:
            g += [dict(experiment="strong", method="hazelcast", n=n, members=m, tpm=t)
                  for m, t in strong["configs"]]
        rng.shuffle(g)
        groups.append(g)
    weak = cfg.get("weak")
    if weak:
        g = []
        for w in weak["workers"]:
            n = round(weak["base_n"] * w ** (1 / 3))
            g.append(dict(experiment="weak", method="hazelcast", n=n, members=w, tpm=1))
            if weak.get("include_seq"):
                g.append(dict(experiment="weak", method="seq", n=n, members=0, tpm=1))
        rng.shuffle(g)
        groups.append(g)
    return groups


def parse_meta(line, meta):
    k, value = line[1:].split(",", 1)
    meta[k] = int(value)


def run_local(cfg, job, W, R, out_dir, run_id, raw, label):
    m, n, p = job["method"], job["n"], job["tpm"]
    pa, pb = gen_matrices.ensure(n, cfg["seed"])
    cmd = (["java"] + cfg["java_opts"] + ["-cp", CP, "LocalMatMul", "--method", m, "--n", str(n),
           "--threads", str(p), "--a", str(pa), "--b", str(pb), "--warmup", str(W), "--reps", str(R),
           "--batch", "0", "--target-ms", str(cfg["batch_target_ms"])])
    lines, code, timed_out, usage, wall = run_process(
        cmd, cfg["per_repetition_budget_s"], cfg["process_startup_allowance_s"],
        out_dir / "stderr" / f"{label}.txt")
    status = "timeout" if timed_out else ("ok" if code == 0 else "error")
    meta, batch = {}, ""
    for line in lines:
        if line.startswith("#"):
            parse_meta(line, meta)
            continue
        kind, idx, ns, b, chk, _cpu = line.split(",")
        if kind == "measured":
            batch = int(b)
        raw.writerow({"run_id": run_id, "experiment": job["experiment"], "method": m, "n": n, "members": 0,
                      "threads_per_member": p, "workers": p, "dtype": "float64", "kind": kind,
                      "rep_index": int(idx), "elapsed_ns": int(ns), "batch": int(b),
                      "time_per_mult_s": int(ns) / 1e9 / int(b), "checksum": chk, "status": status})
    row = {"status": status, "exit_code": code, "batch": batch, "array_storage_bytes": 3 * 8 * n * n,
           "driver_peak_rss_bytes": usage.ru_maxrss * RSS_SCALE,
           "driver_heap_used_bytes": meta.get("java_heap_used_bytes"),
           "driver_peak_thread_count": meta.get("jvm_peak_thread_count"),
           "memory_metric": "ru_maxrss_wait4 (single JVM)", "process_user_s": round(usage.ru_utime, 3),
           "process_sys_s": round(usage.ru_stime, 3), "wall_time_s": round(wall, 3), "command": " ".join(cmd)}
    return row, f"{status} ({wall:.1f}s, peak RSS {row['driver_peak_rss_bytes'] / 2**20:.0f} MiB)"


def run_distributed(cfg, job, W, R, out_dir, run_id, raw, tasks_w, members_w, label, cluster):
    n, m, t = job["n"], job["members"], job["tpm"]
    pa, pb = gen_matrices.ensure(n, cfg["seed"])
    procs, member_list, start_s = start_members(cfg, m, t, cluster, out_dir, label)
    common = {"run_id": run_id, "experiment": job["experiment"], "n": n, "members": m, "threads_per_member": t}
    if start_s is None:
        res = stop_members(procs)
        for i, (rec, r) in enumerate(zip(procs, res)):
            members_w.writerow({**common, "member_index": i, "port": rec["port"], **r})
        return {"status": "cluster_start_failed"}, "cluster start failed"

    cmd = (["java"] + cfg["hz_jvm_opts"] + cfg["driver_java_opts"]
           + ["-cp", CP, "DistMatMul", "--cluster", cluster, "--members", member_list,
              "--expect-members", str(m), "--threads-per-member", str(t), "--n", str(n),
              "--a", str(pa), "--b", str(pb), "--warmup", str(W), "--reps", str(R)])
    lines, code, timed_out, usage, wall = run_process(
        cmd, cfg["per_repetition_budget_s"], cfg["process_startup_allowance_s"],
        out_dir / "stderr" / f"{label}_driver.txt")
    status = "timeout" if timed_out else ("ok" if code == 0 else "error")
    res = stop_members(procs)

    meta, memstats, local_a, n_tasks = {}, {}, 0, 0
    for line in lines:
        if line.startswith("#"):
            parse_meta(line, meta)
        elif line.startswith("iter,"):
            _, kind, idx, tot, sc, co, ga, chk = line.split(",")
            raw.writerow({"run_id": run_id, "experiment": job["experiment"], "method": "hazelcast", "n": n,
                          "members": m, "threads_per_member": t, "workers": m * t, "dtype": "float64",
                          "kind": kind, "rep_index": int(idx), "elapsed_ns": int(tot), "batch": 1,
                          "time_per_mult_s": int(tot) / 1e9, "scatter_s": int(sc) / 1e9,
                          "compute_s": int(co) / 1e9, "gather_s": int(ga) / 1e9, "checksum": chk,
                          "status": status})
        elif line.startswith("task,"):
            f = line.split(",")
            tasks_w.writerow({**common, "kind": f[1], "rep_index": int(f[2]), "block": int(f[3]),
                              "member": f[4], "thread": f[5], "rows": int(f[6]), "a_local": f[7],
                              "fetched_b": f[8], "read_a_ns": f[9], "get_b_ns": f[10], "fetch_b_ns": f[11],
                              "kernel_ns": f[12], "put_c_ns": f[13], "remote_b_bytes": f[14]})
            if f[1] == "measured":
                n_tasks += 1
                local_a += f[7] == "true"
        elif line.startswith("memstat,"):
            f = line.split(",")
            memstats[int(f[1])] = f[2:]
    for i, (rec, r) in enumerate(zip(procs, res)):
        pid = int(rec["line"].split(",")[1])
        s = memstats.get(pid)
        row = {**common, "member_index": i, "port": rec["port"], "pid": pid,
               "address": rec["line"].split(",", 2)[2], "startup_s": round(rec["ready"], 3), **r}
        if s:
            keys = ["processors", "heap_used_bytes", "heap_used_after_gc_bytes", "heap_max_bytes",
                    "entry_cost_A_bytes", "entry_cost_B_bytes", "entry_cost_C_bytes", "live_threads",
                    "peak_threads"]
            row.update(dict(zip(keys, s[1:])))
        members_w.writerow(row)
    row = {"status": status, "exit_code": code, "batch": 1, "blocks": meta.get("blocks"),
           "cluster_start_s": round(start_s, 3), "connect_ns": meta.get("connect_ns"),
           "key_placement_ns": meta.get("key_placement_ns"),
           "scatter_payload_bytes": meta.get("scatter_payload_bytes"),
           "gather_payload_bytes": meta.get("gather_payload_bytes"),
           "b_redistribution_payload_bytes": meta.get("b_redistribution_payload_bytes"),
           "tasks_with_local_a": local_a, "tasks_total": n_tasks, "array_storage_bytes": 3 * 8 * n * n,
           "driver_peak_rss_bytes": usage.ru_maxrss * RSS_SCALE,
           "driver_heap_used_bytes": meta.get("driver_heap_used_bytes"),
           "driver_peak_thread_count": meta.get("driver_peak_thread_count"),
           "members_peak_rss_sum_bytes": sum(r["peak_rss_bytes"] for r in res),
           "memory_metric": "ru_maxrss_wait4 per JVM (driver and each member)",
           "process_user_s": round(usage.ru_utime + sum(r["process_user_s"] for r in res), 3),
           "process_sys_s": round(usage.ru_stime + sum(r["process_sys_s"] for r in res), 3),
           "wall_time_s": round(wall, 3), "command": " ".join(cmd)}
    msg = (f"{status} ({wall:.1f}s, start {start_s:.1f}s, driver RSS {row['driver_peak_rss_bytes'] / 2**20:.0f} MiB, "
           f"members RSS {row['members_peak_rss_sum_bytes'] / 2**20:.0f} MiB, local A {local_a}/{n_tasks})")
    return row, msg


cfg_global = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--only", nargs="*", help="run only these experiments (local, strong, weak)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    cfg_global.update(cfg)
    if args.only:
        for key in ("local", "strong", "weak"):
            if key not in args.only:
                cfg.pop(key, None)

    if not (ROOT / "java" / "build" / "DistMatMul.class").exists():
        sys.exit("Build first: ./build.sh")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / f"run_{run_id}"
    (out_dir / "stderr").mkdir(parents=True)
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    env = collect_env.collect(cfg)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    logical = env["cpu"]["logical_processors"]

    files = {name: open(out_dir / f"{name}.csv", "w", newline="")
             for name in ("raw_measurements", "tasks", "members", "runs")}
    raw = csv.DictWriter(files["raw_measurements"], RAW_FIELDS, lineterminator="\n")
    tasks_w = csv.DictWriter(files["tasks"], TASK_FIELDS, lineterminator="\n")
    members_w = csv.DictWriter(files["members"], MEMBER_FIELDS, lineterminator="\n")
    runs = csv.DictWriter(files["runs"], RUN_FIELDS, lineterminator="\n")
    for w in (raw, tasks_w, members_w, runs):
        w.writeheader()

    R = cfg["repetitions"]
    failed = {}
    t_start = time.monotonic()
    counter = 0
    for group in jobs(cfg):
        for pos, job in enumerate(group):
            exp, meth, n, m, t = job["experiment"], job["method"], job["n"], job["members"], job["tpm"]
            workers = t if m == 0 else m * t
            W = warmups_for(cfg, n)
            est = estimate_bytes(n, m)
            base = {"run_id": run_id, "experiment": exp, "method": meth, "n": n, "members": m,
                    "threads_per_member": t, "workers": workers, "oversubscribed": workers > logical,
                    "warmups": W, "repetitions": R, "estimated_bytes": est,
                    "line_timeout_s": cfg["per_repetition_budget_s"], "order_in_group": pos}
            tag = f"{exp}_{meth}_n{n}_m{m}_t{t}"
            label = f"[{exp:6s} n={n:5d}] {meth:9s} m={m} t={t}"
            key = (exp, meth, m, t)
            if key in failed and n >= failed[key][0]:
                runs.writerow({**base, "status": f"not_attempted_after_{failed[key][1]}"})
                print(f"{label} not attempted (earlier {failed[key][1]})")
                continue
            if est > cfg["memory_budget_bytes"]:
                runs.writerow({**base, "status": "skipped_memory_budget"})
                failed[key] = (n, "memory_budget")
                print(f"{label} skipped (estimated {est / 1e9:.2f} GB > budget)")
                continue
            print(f"{label} W={W} R={R} ...", end=" ", flush=True)
            if m == 0:
                row, msg = run_local(cfg, job, W, R, out_dir, run_id, raw, tag)
            else:
                counter += 1
                row, msg = run_distributed(cfg, job, W, R, out_dir, run_id, raw, tasks_w, members_w, tag,
                                           f"a4-{run_id}-{counter}")
            runs.writerow({**base, **row})
            for f in files.values():
                f.flush()
            print(msg)
            if row["status"] != "ok":
                failed[key] = (n, row["status"])

    for f in files.values():
        f.close()
    mins = (time.monotonic() - t_start) / 60
    print(f"\nFinished in {mins:.1f} min. Results written to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
