#!/usr/bin/env python3
"""Record the execution environment (hardware, OS, runtime versions).

Reused from Assignment 3 (repo amaisuarez/assignment3, commit b8447c5; originally
from Assignment 1). Changed for Assignment 4: Hazelcast version and notes on how
workers, processes and machines are counted."""
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # tool missing
        return f"unavailable ({e.__class__.__name__})"


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def cpu_info():
    info = {"logical_processors": os.cpu_count()}
    if sys.platform.startswith("linux"):
        cpuinfo = read("/proc/cpuinfo")
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        info["model"] = m.group(1).strip() if m else platform.processor()
        cores = set()
        phys = core = None
        for line in cpuinfo.splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
            elif not line.strip() and core is not None:
                cores.add((phys, core))
                phys = core = None
        info["physical_cores"] = len(cores) or None
        info["lscpu"] = sh(["lscpu"])
        gov = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip()
        info["scaling_governor"] = gov or "unknown"
        info["affinity_cpus_available"] = sorted(os.sched_getaffinity(0))
    elif sys.platform == "darwin":
        info["model"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["physical_cores"] = int(sh(["sysctl", "-n", "hw.physicalcpu"]) or 0) or None
        for key in ["hw.l1dcachesize", "hw.l2cachesize", "hw.l3cachesize", "hw.cpufrequency",
                    "machdep.cpu.thread_count", "machdep.cpu.core_count"]:
            info[key] = sh(["sysctl", "-n", key])
        feats = (sh(["sysctl", "-n", "machdep.cpu.features"]) + " "
                 + sh(["sysctl", "-n", "machdep.cpu.leaf7_features"])).split()
        info["simd_features"] = [f for f in feats if f.startswith(("SSE", "AVX", "FMA"))]
        info["power_source"] = sh(["pmset", "-g", "batt"]).splitlines()[:1]
    else:
        info["model"] = platform.processor()
        info["physical_cores"] = None
    return info


def ram_bytes():
    if sys.platform.startswith("linux"):
        m = re.search(r"MemTotal:\s+(\d+) kB", read("/proc/meminfo"))
        return int(m.group(1)) * 1024 if m else None
    if sys.platform == "darwin":
        try:
            return int(sh(["sysctl", "-n", "hw.memsize"]))
        except ValueError:
            return None
    return None


def collect(config):
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": cpu_info(),
        "ram_bytes": ram_bytes(),
        "python": {"implementation": platform.python_implementation(),
                   "version": sys.version, "executable": sys.executable},
        "numpy_version_reference_only": numpy_version,
        "java_version": sh(["java", "-version"]),
        "javac_version": sh(["javac", "-version"]),
        "java_opts": config.get("java_opts"),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain"]).strip()) if Path(".git").exists() else None,
        "hazelcast_version": "5.5.0 (open-source, lib/hazelcast-5.5.0.jar, "
                             "SHA-256 cf560b40b988526e780e4fae12cb6fe01c5158bb9e00839882857f32b544a19e)",
        "member_java_opts": config.get("member_java_opts"),
        "driver_java_opts": config.get("driver_java_opts"),
        "hz_jvm_opts": config.get("hz_jvm_opts"),
        "deployment_note": ("All Hazelcast members and the driver run on THIS machine (one physical "
                            "machine), communicating over the loopback interface 127.0.0.1. "
                            "members = number of member JVM processes; threads_per_member = size of "
                            "each member's 'matmul' executor; workers = members * threads_per_member. "
                            "No inter-machine network is involved."),
        "threads_note": ("Worker threads = executor threads that run BlockTasks. Each JVM also runs "
                         "Hazelcast's own threads (partition operation threads, I/O threads, "
                         "housekeeping), the JIT compilers and 1 SerialGC thread; their peak "
                         "number is recorded per member (members.csv, peak_threads). No numerical "
                         "library is used. Local references: p = threads of the LocalMatMul pool."),
        "affinity_note": ("No CPU affinity / pinning: macOS does not let user programs bind threads "
                          "or processes to cores, so the OS scheduler places them freely."),
    }


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    print(json.dumps(collect(cfg), indent=2))
