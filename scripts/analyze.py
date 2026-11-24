#!/usr/bin/env python3
"""Tables and figures for the Assignment 4 report from one benchmark run.

Usage: python3 scripts/analyze.py [results/run_XXXX]   (default: latest run)

Statistics use the MEASURED repetitions only (warm-ups stay in the raw CSV):
median, Q1, Q3, IQR. Times are per multiplication.

Timing boundaries (never mixed without saying so):
  local seq / static   kernel only: one multiplication call (LocalMatMul, as in A3)
  distributed total    end-to-end inside the driver: scatter + compute + gather
                       (excludes reading files, starting the cluster, connecting)
  distributed compute  only the compute phase (tasks submitted -> all finished),
                       the closest equivalent of the local kernel time, but it
                       still includes the B redistribution and storing C

Derived values (medians):
  S_seq     = T_seq(n) / T(n)            speedup against the sequential kernel
  S_1x1     = T_dist(1x1)(n) / T(n)      speedup against the same distributed code
                                         with one member and one thread
  E         = S / workers                efficiency (workers = members x threads)
  weak scaling efficiency  E_w(W) = T(W=1, n0) / T(W, n_W),  n_W = round(n0 W^(1/3))

Outputs: report/tables/*.csv|*.md and report/figures/*.png
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Same palette as Assignments 2-3.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
PHASE_COLORS = {"scatter": ORANGE, "compute": BLUE, "gather": AQUA}


def latest_run():
    runs = sorted((ROOT / "results").glob("run_*"))
    if not runs:
        sys.exit("No runs found in results/")
    return runs[-1]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_table(rows, cols, name, tables):
    with open(tables / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (tables / f"{name}.md").write_text("\n".join(lines) + "\n")


def fmt_time(s):
    if s is None:
        return "–"
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} s"


def mib(b):
    return f"{int(float(b)) / 2**20:.0f}" if b not in (None, "") else "–"


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)


def q(ts):
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return {"median": med, "q1": q1, "q3": q3, "iqr": q3 - q1, "reps": len(ts)}


def cfg_label(m, t):
    return f"{m}×{t}"


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    run = run if run.is_absolute() else (ROOT / run)
    raw = load_csv(run / "raw_measurements.csv")
    runs = load_csv(run / "runs.csv")
    tasks = load_csv(run / "tasks.csv")
    members = load_csv(run / "members.csv")
    env = json.load(open(run / "environment.json"))
    tables, figs = ROOT / "report" / "tables", ROOT / "report" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"})
    logical = env["cpu"]["logical_processors"]

    # key: (experiment, method, n, members, threads_per_member)
    def key(r):
        return (r["experiment"], r["method"], int(r["n"]), int(r["members"]), int(r["threads_per_member"]))

    samples = defaultdict(lambda: defaultdict(list))
    for r in raw:
        if r["kind"] != "measured" or r["status"] != "ok":
            continue
        k = key(r)
        samples[k]["total"].append(float(r["time_per_mult_s"]))
        for ph in ("scatter", "compute", "gather"):
            if r[f"{ph}_s"]:
                samples[k][ph].append(float(r[f"{ph}_s"]))
    stats = {k: {ph: q(v) for ph, v in d.items()} for k, d in samples.items()}
    runinfo = {key(r): r for r in runs}

    # ------------------------------------------------------------ environment
    c = env["cpu"]
    env_rows = [
        {"item": "Machine", "value": "1 physical machine (all members and the driver on it, loopback 127.0.0.1)"},
        {"item": "CPU", "value": c["model"]},
        {"item": "Physical cores / logical processors", "value": f"{c['physical_cores']} / {logical} (Hyper-Threading)"},
        {"item": "Caches (L1d / L2 per core, L3 shared)",
         "value": f"{int(c.get('hw.l1dcachesize', 0)) // 1024} KiB / {int(c.get('hw.l2cachesize', 0)) // 1024} KiB / "
                  f"{int(c.get('hw.l3cachesize', 0)) // 2**20} MiB"},
        {"item": "RAM", "value": f"{env['ram_bytes'] / 2**30:.0f} GiB"},
        {"item": "OS", "value": f"{env['os']} (Darwin {env['kernel']})"},
        {"item": "Java", "value": env["java_version"].splitlines()[0]},
        {"item": "Hazelcast", "value": env.get("hazelcast_version", "")},
        {"item": "Member JVM", "value": " ".join(env.get("member_java_opts") or [])},
        {"item": "Driver JVM (client)", "value": " ".join(env.get("driver_java_opts") or [])},
        {"item": "Local reference JVM", "value": " ".join(env.get("java_opts") or [])},
        {"item": "Power", "value": " ".join(c.get("power_source", []))},
    ]
    write_table(env_rows, ["item", "value"], "environment", tables)

    # ------------------------------------------------------------ strong scaling
    sizes = sorted({k[2] for k in stats if k[0] in ("local", "strong")})
    configs = sorted({(k[3], k[4]) for k in runinfo if k[0] == "strong"},
                     key=lambda mt: (mt[0] * mt[1], -mt[0]))
    local_methods = sorted({(k[1], k[4]) for k in runinfo if k[0] == "local"},
                           key=lambda x: (x[0] != "seq", x[1]))
    rows = []
    for n in sizes:
        seq = stats.get(("local", "seq", n, 0, 1), {}).get("total")
        d11 = stats.get(("strong", "hazelcast", n, 1, 1), {}).get("total")
        for meth, p in local_methods:
            k = ("local", meth, n, 0, p)
            s = stats.get(k, {}).get("total")
            info = runinfo.get(k, {})
            row = {"n": n, "method": meth if meth == "seq" else f"{meth} (A3)", "config": f"p={p}",
                   "workers": p, "status": info.get("status", "missing"), "boundary": "kernel"}
            if s:
                row.update(total=fmt_time(s["median"]), total_s=f"{s['median']:.6g}",
                           iqr_pct=f"{100 * s['iqr'] / s['median']:.1f}", reps=s["reps"],
                           gflops=f"{2 * n ** 3 / s['median'] / 1e9:.2f}")
                if seq:
                    S = seq["median"] / s["median"]
                    row.update(S_seq=f"{S:.2f}", E_seq=f"{S / p:.2f}")
            rows.append(row)
        for m, t in configs:
            k = ("strong", "hazelcast", n, m, t)
            st = stats.get(k, {})
            s = st.get("total")
            info = runinfo.get(k, {})
            w = m * t
            row = {"n": n, "method": "hazelcast", "config": cfg_label(m, t), "workers": w,
                   "status": info.get("status", "missing"), "boundary": "end-to-end",
                   "oversub": "yes" if w > logical else ""}
            if s:
                row.update(total=fmt_time(s["median"]), total_s=f"{s['median']:.6g}",
                           iqr_pct=f"{100 * s['iqr'] / s['median']:.1f}", reps=s["reps"],
                           gflops=f"{2 * n ** 3 / s['median'] / 1e9:.2f}",
                           scatter=fmt_time(st["scatter"]["median"]), compute=fmt_time(st["compute"]["median"]),
                           gather=fmt_time(st["gather"]["median"]),
                           compute_s=f"{st['compute']['median']:.6g}",
                           comm_pct=f"{100 * (st['scatter']['median'] + st['gather']['median']) / s['median']:.0f}")
                if seq:
                    S = seq["median"] / s["median"]
                    row.update(S_seq=f"{S:.2f}", E_seq=f"{S / w:.2f}",
                               S_seq_compute=f"{seq['median'] / st['compute']['median']:.2f}")
                if d11:
                    S1 = d11["median"] / s["median"]
                    row.update(S_1x1=f"{S1:.2f}", E_1x1=f"{S1 / w:.2f}")
            rows.append(row)
    cols = ["n", "method", "config", "workers", "status", "boundary", "total", "iqr_pct", "reps",
            "scatter", "compute", "gather", "comm_pct", "S_seq", "E_seq", "S_seq_compute", "S_1x1", "E_1x1",
            "gflops", "total_s", "compute_s"]
    write_table(rows, cols, "strong_all", tables)

    # compact table: one row per n, columns = configurations (median total time)
    compact = []
    for n in sizes:
        row = {"n": n}
        for meth, p in local_methods:
            s = stats.get(("local", meth, n, 0, p), {}).get("total")
            row[f"{meth} p={p}"] = fmt_time(s["median"]) if s else "–"
        for m, t in configs:
            s = stats.get(("strong", "hazelcast", n, m, t), {}).get("total")
            st = runinfo.get(("strong", "hazelcast", n, m, t), {}).get("status", "")
            row[f"HZ {cfg_label(m, t)}"] = fmt_time(s["median"]) if s else (st or "–")
        compact.append(row)
    compact_cols = (["n"] + [f"{meth} p={p}" for meth, p in local_methods]
                    + [f"HZ {cfg_label(m, t)}" for m, t in configs])
    write_table(compact, compact_cols, "strong_compact", tables)

    # ------------------------------------------------------------ per-task breakdown
    tk = defaultdict(lambda: defaultdict(list))
    for r in tasks:
        if r["kind"] != "measured" or r["experiment"] != "strong":
            continue
        k = (int(r["n"]), int(r["members"]), int(r["threads_per_member"]))
        it = r["rep_index"]
        tk[k]["kernel"].append(int(r["kernel_ns"]) / 1e9)
        tk[k]["get_b"].append(int(r["get_b_ns"]) / 1e9)
        tk[k]["read_a"].append(int(r["read_a_ns"]) / 1e9)
        tk[k]["put_c"].append(int(r["put_c_ns"]) / 1e9)
        if r["fetched_b"] == "true":
            tk[k]["fetch_b"].append(int(r["fetch_b_ns"]) / 1e9)
            tk[k]["remote_bytes"].append(int(r["remote_b_bytes"]))
        tk[k][f"kernel_iter_{it}"].append(int(r["kernel_ns"]) / 1e9)
        tk[k]["a_local"].append(r["a_local"] == "true")
    trows = []
    for (n, m, t), d in sorted(tk.items()):
        iters = [v for kk, v in d.items() if kk.startswith("kernel_iter_")]
        imb = np.median([max(v) / min(v) for v in iters if min(v) > 0]) if iters else float("nan")
        fetch = np.median(d["fetch_b"]) if d["fetch_b"] else 0.0
        rb = np.median(d["remote_bytes"]) if d["remote_bytes"] else 0
        comp = stats.get(("strong", "hazelcast", n, m, t), {}).get("compute")
        trows.append({"n": n, "config": cfg_label(m, t), "workers": m * t,
                      "read_a": fmt_time(np.median(d["read_a"])), "get_b": fmt_time(np.median(d["get_b"])),
                      "fetch_b": fmt_time(fetch) if d["fetch_b"] else "–",
                      "kernel": fmt_time(np.median(d["kernel"])), "put_c": fmt_time(np.median(d["put_c"])),
                      "kernel_imbalance": f"{imb:.2f}",
                      "remote_b_mib_per_member": f"{rb / 2**20:.1f}",
                      "b_fetch_mib_s": f"{rb / 2**20 / fetch:.0f}" if fetch > 0 and rb > 0 else "–",
                      "a_local": f"{sum(d['a_local'])}/{len(d['a_local'])}",
                      "compute_phase": fmt_time(comp["median"]) if comp else "–",
                      "kernel_s": f"{np.median(d['kernel']):.6g}", "get_b_s": f"{np.median(d['get_b']):.6g}"})
    write_table(trows, ["n", "config", "workers", "read_a", "get_b", "fetch_b", "kernel", "put_c",
                        "kernel_imbalance", "remote_b_mib_per_member", "b_fetch_mib_s", "a_local",
                        "compute_phase", "kernel_s", "get_b_s"], "tasks", tables)

    # ------------------------------------------------------------ transfers
    xrows = []
    for n in sizes:
        for m, t in configs:
            k = ("strong", "hazelcast", n, m, t)
            st, info = stats.get(k), runinfo.get(k)
            if not st or not info:
                continue
            sc_b, ga_b = int(info["scatter_payload_bytes"]), int(info["gather_payload_bytes"])
            rd_b = int(info["b_redistribution_payload_bytes"])
            xrows.append({"n": n, "config": cfg_label(m, t),
                          "scatter_mib": f"{sc_b / 2**20:.0f}", "scatter": fmt_time(st["scatter"]["median"]),
                          "scatter_mib_s": f"{sc_b / 2**20 / st['scatter']['median']:.0f}",
                          "gather_mib": f"{ga_b / 2**20:.0f}", "gather": fmt_time(st["gather"]["median"]),
                          "gather_mib_s": f"{ga_b / 2**20 / st['gather']['median']:.0f}",
                          "b_redistribution_mib": f"{rd_b / 2**20:.0f}"})
    write_table(xrows, ["n", "config", "scatter_mib", "scatter", "scatter_mib_s", "gather_mib", "gather",
                        "gather_mib_s", "b_redistribution_mib"], "transfers", tables)

    # ------------------------------------------------------------ memory
    mem_by = defaultdict(list)
    for r in members:
        if r["peak_rss_bytes"]:
            mem_by[(r["experiment"], int(r["n"]), int(r["members"]), int(r["threads_per_member"]))].append(r)
    mrows = []
    for n in sizes:
        for meth, p in local_methods:
            info = runinfo.get(("local", meth, n, 0, p))
            if info and info.get("driver_peak_rss_bytes"):
                mrows.append({"n": n, "config": f"{meth} p={p}", "processes": 1,
                              "driver_rss_mib": mib(info["driver_peak_rss_bytes"]),
                              "total_rss_mib": mib(info["driver_peak_rss_bytes"]),
                              "array_mib": f"{3 * 8 * n * n / 2**20:.0f}"})
        for m, t in configs:
            info = runinfo.get(("strong", "hazelcast", n, m, t))
            ms = mem_by.get(("strong", n, m, t), [])
            if not info or not ms or not info.get("driver_peak_rss_bytes"):
                continue
            rss = [int(r["peak_rss_bytes"]) for r in ms]
            heap = [int(r["heap_used_after_gc_bytes"]) for r in ms if r.get("heap_used_after_gc_bytes")]
            entry = [sum(int(r[f"entry_cost_{x}_bytes"]) for x in "ABC") for r in ms if r.get("entry_cost_A_bytes")]
            mrows.append({"n": n, "config": cfg_label(m, t), "processes": m + 1,
                          "driver_rss_mib": mib(info["driver_peak_rss_bytes"]),
                          "member_rss_mib_mean": f"{np.mean(rss) / 2**20:.0f}",
                          "member_rss_mib_max": f"{max(rss) / 2**20:.0f}",
                          "total_rss_mib": f"{(sum(rss) + int(info['driver_peak_rss_bytes'])) / 2**20:.0f}",
                          "member_heap_after_gc_mib_mean": f"{np.mean(heap) / 2**20:.0f}" if heap else "–",
                          "member_map_entries_mib_mean": f"{np.mean(entry) / 2**20:.1f}" if entry else "–",
                          "array_mib": f"{3 * 8 * n * n / 2**20:.0f}"})
    write_table(mrows, ["n", "config", "processes", "driver_rss_mib", "member_rss_mib_mean", "member_rss_mib_max",
                        "total_rss_mib", "member_heap_after_gc_mib_mean", "member_map_entries_mib_mean",
                        "array_mib"], "memory", tables)

    # ------------------------------------------------------------ cluster start-up
    srows = []
    for m, t in configs:
        st = [float(r["cluster_start_s"]) for r in runs if r["experiment"] == "strong" and r["cluster_start_s"]
              and int(r["members"]) == m and int(r["threads_per_member"]) == t]
        cn = [int(r["connect_ns"]) / 1e9 for r in runs if r["experiment"] == "strong" and r["connect_ns"]
              and int(r["members"]) == m and int(r["threads_per_member"]) == t]
        if st:
            srows.append({"config": cfg_label(m, t), "members": m, "clusters_started": len(st),
                          "start_median_s": f"{np.median(st):.1f}", "start_range_s": f"{min(st):.1f}–{max(st):.1f}",
                          "client_connect_median_s": f"{np.median(cn):.1f}" if cn else "–"})
    write_table(srows, ["config", "members", "clusters_started", "start_median_s", "start_range_s",
                        "client_connect_median_s"], "startup", tables)

    # ------------------------------------------------------------ weak scaling
    wk = sorted({k for k in stats if k[0] == "weak" and k[1] == "hazelcast"}, key=lambda k: k[3])
    wrows = []
    base = stats[wk[0]]["total"]["median"] if wk else None
    base_c = stats[wk[0]]["compute"]["median"] if wk else None
    for k in wk:
        _, _, n, m, t = k
        st = stats[k]
        seq = stats.get(("weak", "seq", n, 0, 1), {}).get("total")
        wrows.append({"workers": m, "n": n, "flops_per_worker_G": f"{2 * n ** 3 / m / 1e9:.2f}",
                      "total": fmt_time(st["total"]["median"]),
                      "iqr_pct": f"{100 * st['total']['iqr'] / st['total']['median']:.1f}",
                      "compute": fmt_time(st["compute"]["median"]),
                      "E_weak_total": f"{base / st['total']['median']:.2f}",
                      "E_weak_compute": f"{base_c / st['compute']['median']:.2f}",
                      "T_seq": fmt_time(seq["median"]) if seq else "–",
                      "S_seq": f"{seq['median'] / st['total']['median']:.2f}" if seq else "–"})
    write_table(wrows, ["workers", "n", "flops_per_worker_G", "total", "iqr_pct", "compute", "E_weak_total",
                        "E_weak_compute", "T_seq", "S_seq"], "weak", tables)

    # ------------------------------------------------------------ limits
    lrows = [{"experiment": r["experiment"], "method": r["method"], "n": r["n"],
              "config": cfg_label(r["members"], r["threads_per_member"]) if r["members"] != "0"
              else f"p={r['threads_per_member']}", "status": r["status"],
              "estimated_gb": f"{int(r['estimated_bytes']) / 1e9:.2f}"}
             for r in runs]
    write_table(lrows, ["experiment", "method", "n", "config", "status", "estimated_gb"], "limits", tables)

    # ================================================================ figures
    # Fig 1: speedup vs sequential, by workers, for the largest sizes
    big = [n for n in sizes if n >= 1024]
    fig, axes = plt.subplots(1, len(big), figsize=(3.3 * len(big), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, big):
        seq = stats.get(("local", "seq", n, 0, 1), {}).get("total")
        if not seq:
            continue
        for fam, sel, color, mk, lab in [
            ("proc", lambda m, t: t == 1, BLUE, "o", "HZ m members × 1 thread"),
            ("thr", lambda m, t: m == 1, ORANGE, "s", "HZ 1 member × t threads"),
        ]:
            xs, ys = [], []
            for m, t in configs:
                s = stats.get(("strong", "hazelcast", n, m, t), {}).get("total")
                if sel(m, t) and s:
                    xs.append(m * t)
                    ys.append(seq["median"] / s["median"])
            ax.plot(xs, ys, marker=mk, color=color, label=lab, linewidth=1.6)
        s22 = stats.get(("strong", "hazelcast", n, 2, 2), {}).get("total")
        if s22:
            ax.plot([4], [seq["median"] / s22["median"]], marker="D", color=VIOLET, linestyle="none",
                    label="HZ 2 × 2")
        xs, ys = [], []
        for meth, p in local_methods:
            s = stats.get(("local", meth, n, 0, p), {}).get("total")
            if s:
                xs.append(p)
                ys.append(seq["median"] / s["median"])
        ax.plot(xs, ys, marker="^", color=INK2, linestyle=":", label="Local threads (A3, kernel only)")
        ax.axhline(1, color=INK2, linewidth=0.8)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("workers (threads in total)")
        ax.set_xticks([1, 2, 3, 4])
        style_axes(ax)
    axes[0].set_ylabel("speedup vs sequential  T_seq / T")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "strong_speedup.png", dpi=200)
    plt.close(fig)

    # Fig 2: phase breakdown (stacked) for each config, two largest sizes
    show = [n for n in sizes if n >= 2048][-2:] or sizes[-2:]
    fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 3.2))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, show):
        labels, bottoms = [], None
        vals = {ph: [] for ph in PHASE_COLORS}
        for m, t in configs:
            st = stats.get(("strong", "hazelcast", n, m, t))
            if not st:
                continue
            labels.append(cfg_label(m, t))
            for ph in PHASE_COLORS:
                vals[ph].append(st[ph]["median"])
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))
        for ph, col in PHASE_COLORS.items():
            ax.bar(x, vals[ph], bottom=bottoms, color=col, label=ph, width=0.65)
            bottoms += np.array(vals[ph])
        seq = stats.get(("local", "seq", n, 0, 1), {}).get("total")
        if seq:
            ax.axhline(seq["median"], color=INK, linestyle="--", linewidth=1, label="sequential (kernel)")
        s4 = stats.get(("local", "static", n, 0, 4), {}).get("total")
        if s4:
            ax.axhline(s4["median"], color=INK2, linestyle=":", linewidth=1, label="local 4 threads (A3)")
        ax.set_xticks(x, labels)
        ax.set_xlabel("configuration  members × threads per member")
        ax.set_title(f"n = {n}")
        style_axes(ax)
    axes[0].set_ylabel("median time per multiplication (s)")
    handles, labels_ = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=7, frameon=False, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figs / "phases.png", dpi=200)
    plt.close(fig)

    # Fig 3: inside the compute phase — task kernel vs waiting for B, n = largest common
    n3 = show[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels, kern, getb = [], [], []
    for m, t in configs:
        d = tk.get((n3, m, t))
        if not d:
            continue
        labels.append(cfg_label(m, t))
        kern.append(np.median(d["kernel"]))
        getb.append(np.median(d["get_b"]))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, getb, width=0.36, color=ORANGE, label="obtain B (fetch or wait)")
    ax.bar(x + 0.18, kern, width=0.36, color=BLUE, label="kernel (own block)")
    ax.set_xticks(x, labels)
    ax.set_xlabel("configuration  members × threads per member")
    ax.set_ylabel("median per task (s)")
    ax.set_title(f"Inside the compute phase, n = {n3}")
    ax.legend(fontsize=7, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(figs / "tasks.png", dpi=200)
    plt.close(fig)

    # Fig 4: weak scaling
    if wk:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ws = [k[3] for k in wk]
        ax.plot(ws, [base / stats[k]["total"]["median"] for k in wk], marker="o", color=BLUE,
                label="end-to-end")
        ax.plot(ws, [base_c / stats[k]["compute"]["median"] for k in wk], marker="s", color=ORANGE,
                linestyle="--", label="compute phase only")
        ax.axhline(1, color=INK2, linewidth=0.8)
        for k in wk:
            ax.annotate(f"n={k[2]}", (k[3], base / stats[k]["total"]["median"]), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7, color=INK2)
        ax.set_xticks(ws)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("members (1 thread each)")
        ax.set_ylabel("weak-scaling efficiency  T(1) / T(W)")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(figs / "weak.png", dpi=200)
        plt.close(fig)

    # Fig 5: memory (total peak RSS of all processes) per configuration
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    cfg_order = [r["config"] for r in mrows if r["n"] == sizes[-1]]
    width = 0.8 / len(sizes)
    for i, n in enumerate(sizes):
        vals = {r["config"]: float(r["total_rss_mib"]) for r in mrows if r["n"] == n}
        ax.bar(np.arange(len(cfg_order)) + (i - (len(sizes) - 1) / 2) * width,
               [vals.get(cname, 0) for cname in cfg_order], width=width,
               color=RAMP[min(i, len(RAMP) - 1)], label=f"n = {n}")
    ax.set_xticks(range(len(cfg_order)), cfg_order, rotation=30, fontsize=7)
    ax.set_ylabel("sum of peak RSS of all JVMs (MiB)")
    ax.set_xlabel("configuration")
    ax.legend(fontsize=7, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(figs / "memory.png", dpi=200)
    plt.close(fig)

    print(f"Tables in {tables.relative_to(ROOT)}, figures in {figs.relative_to(ROOT)} (from {run.name})")


if __name__ == "__main__":
    main()
