# Distributed matrix multiplication in Java with Hazelcast

**Big Data – Individual Assignment 4** · Amai Suárez Navarro · Grado en Ciencia e Ingeniería de Datos (ULPGC), 2026-2027

This folder has everything behind my fourth Big Data assignment. It continues
Assignment 3 (https://github.com/amaisuarez/assignment3) and asks:

> How does a distributed matrix multiplication built on Hazelcast scale when I add
> workers, and how much of its time goes to computation, to moving data and to the
> framework itself, compared with the sequential and the shared-memory parallel
> versions of the same kernel?

The full write-up is in `Assignment04_SuarezNavarro_Report.pdf`.

**Repository:** https://github.com/amaisuarez/assignment4

> **Environment label.** All Hazelcast members and the driver run as separate JVM
> processes on **one physical machine** (my laptop) and talk over the loopback
> interface. This shows how the framework, the partitioning and the extra processes
> behave, but it does **not** measure a real network between machines or the memory
> of several machines. (The assignment allows local multi-worker runs only if the
> instructor permits it.) The code also runs on several machines: see
> [Running on several machines](#running-on-several-machines).

---

## What I found (short version)

All numbers come from my laptop, an Intel Core i5-5350U MacBook Air with **2 physical cores,
4 logical processors** and 8 GB of RAM (run `results/run_20261124-095140/`, 48/48 configurations OK).

- **The distributed code scales up to the 2 cores.** For n ≥ 2048, two or more workers made it
  1.73–1.95× faster than the same code with one worker. Beyond 2 workers the time improved by at
  most 10 %: adding processes on one machine adds no hardware.
- **It doesn't beat the local thread pool.** Best end-to-end speedup against the sequential kernel:
  1.45 at n = 2048 (4.76 s against 6.91 s) and 1.50 at n = 3072, while the A3 thread pool reached
  1.80 and 1.76. Distributed reaches 81–85 % of the local pool's speed at large n.
- **One worker costs extra.** 1 member × 1 thread took 1.2–1.6× as long as `seq`. Most of it is the
  kernel running more slowly inside the member JVM (a diagnostic re-run ruled out warm-up and heap
  size, `results/diag_kernel_1x1.txt`).
- **Moving data matters only for small matrices.** Scatter + gather took 3–7 % of the time for
  n ≥ 2048 but 15–27 % at n = 512, where no distributed configuration beat the local pool.
- **Copying B grows with the members.** Each member needs all of B: at n = 3072 obtaining it took
  0.44 s with 2 members and 1.00 s with 4 (8n²(m−1) bytes in total).
- **Threads beat processes on one machine** in most cases (e.g. n = 1024: 1×4 608 ms vs 4×1 861 ms).
- **Memory and start-up.** Distributed runs used 5–16× more memory (all JVMs together, up to
  3.4 GiB at n = 3072 vs 401 MiB local), and starting a 4-member cluster took ~13 s.
- **Weak scaling** (2.15 GFLOP per worker): efficiency 0.59, 0.51, 0.40 for 2, 3, 4 members.

All methods give **bitwise-identical results** to the sequential kernel (`results/validation_report.txt`).

---

## What I reuse from earlier assignments

| Reused | From |
|---|---|
| Kernel | `java/Kernels.java` (i-k-j kernel and the row-range kernel `ikjRows`) copied **unchanged** from [`amaisuarez/assignment3`](https://github.com/amaisuarez/assignment3), commit `b8447c5`. The i-k-j kernel itself comes from Assignment 2 (`Dense.ikj`) |
| Parallel version | `java/Parallel.java` from A3, unchanged except that the Vector API kernel was removed (`rowKernel`). `static` = fixed thread pool with static row blocks |
| Local driver | `java/LocalMatMul.java` = A3's `MatMul3.java` without the Vector API methods. Same protocol (fresh JVM, warm-ups, batching, 5 repetitions) |
| Inputs | `scripts/gen_matrices.py` and `inputs/manifest.csv`, unchanged (originally from Assignment 1). Seed `20262027`; the files are byte-identical to the A1–A3 inputs (the script checks the SHA-256) |
| Harness | `scripts/run_benchmarks.py`, `scripts/analyze.py`, `tests/validate.py`, `scripts/collect_env.py` adapted from A3 (cluster start/stop, per-member memory and per-task timings added) |

---

## The methods

All methods compute C = A·B for n × n float64 matrices stored as row-major `double[]`,
with the same i-k-j kernel and the same inputs, and give **bit-identical results**.

| Method | What it is | Where it runs | File |
|---|---|---|---|
| `seq` | Sequential baseline (A2/A3 i-k-j kernel) | 1 JVM, 1 thread | `Kernels.java`, `LocalMatMul.java` |
| `static` p | Shared-memory parallel version of A3 (thread pool, static row blocks) | 1 JVM, p threads | `Parallel.java`, `LocalMatMul.java` |
| `hazelcast` m×t | **Distributed version**: Hazelcast cluster with m member processes, each running t executor threads (workers = m·t) | m member JVMs + 1 driver JVM | `DistMatMul.java`, `BlockTask.java`, `ClusterMember.java` |

### How the distributed version works

- **Framework:** [Hazelcast](https://hazelcast.com) 5.5.0 open-source (in-memory data grid). Data is kept
  in distributed maps (`IMap`) and work is sent to members with the distributed executor
  (`IExecutorService`).
- **Processes:** every *member* is a separate JVM (`ClusterMember`). The *driver* (`DistMatMul`) is a Hazelcast
  **client**: it stores no data and runs no tasks.
- **Partitioning (row blocks):** A and C are cut into P = m·t row blocks (sizes differ by ≤ 1 row);
  block b belongs to member b mod m, so each member gets t blocks, one per executor thread.
  B is cut into m row panels, one per member.
- **Placement:** Hazelcast decides where a key lives by hashing it into one of 271 partitions. To control
  placement, the driver looks for keys whose partition is owned by the wanted member (once, before timing).
  The task for block b is sent with `submitToKeyOwner`, so it runs **on the member that stores its A block**
  and writes its C block on the same member.
- **B redistribution:** every task needs all of B. The first task of each member fetches the panels of the
  other members with one `getAll` and builds a full copy of B; the other tasks of that member reuse it.
  So B travels once per member and multiplication: 8n²(m−1)/m bytes per member.
- **Result:** the driver collects the C blocks with `getAll` and copies them into C.

One multiplication has three timed phases (driver, `System.nanoTime()`), which run one after the other:

| Phase | Includes |
|---|---|
| `scatter` | cutting A and B into blocks (array copies) + `putAll` into the maps A and B (driver → members) |
| `compute` | submitting the P tasks and waiting for all of them: B redistribution, kernel, storing C (on the members) |
| `gather` | `getAll` of the C blocks (members → driver) + copying them into C |

**End-to-end time = scatter + compute + gather.** Not included: reading the input files, starting the
members, connecting the client and the cleanup between multiplications (all reported separately where
measured). Every task also reports its own timings (read A, obtain B, kernel, store C); tasks run at the
same time, so these are reported per task and never added up.

---

## What's in the folder

```
java/Kernels.java          i-k-j kernels (from A3, unchanged)
java/Parallel.java         shared-memory parallel versions (from A3, Vector API removed)
java/LocalMatMul.java      driver for the single-JVM references seq / static (from A3's MatMul3)
java/HzSetup.java          Hazelcast configuration of members and client (one place for all settings)
java/ClusterMember.java    one Hazelcast member = one worker process
java/BlockTask.java        the distributed task: one row block of C on the member that owns it
java/MemberOps.java        small untimed tasks: cluster-safe check, clear cache, memory statistics
java/DistMatMul.java       driver of the distributed multiplication (Hazelcast client)

scripts/gen_matrices.py    dense inputs (reused from A1-A3, same files)
scripts/run_benchmarks.py  runs the whole study (starts and stops a cluster for every configuration)
scripts/collect_env.py     saves info about the machine (CPU, RAM, versions...)
scripts/analyze.py         builds the tables and plots from the raw data
scripts/diag_kernel_1x1.py diagnostic re-run of 1 member x 1 thread (report Section 5.2)
tests/validate.py          correctness checks (run before benchmarking)

config.json                settings of the real study (sizes, configurations, repetitions, limits)
config_quick.json          1-minute smoke test
fetch_deps.sh              downloads Hazelcast 5.5.0 from Maven Central and checks its SHA-256
build.sh                   compiles the Java code
build_report.sh            builds the PDF from report/report.md (pandoc + headless Chrome)

inputs/manifest.csv        seeds and SHA-256 of the inputs (same as A1-A3)
results/                   raw data of the study, its log and the validation report
report/                    report source (report.md), tables and figures generated by analyze.py
Assignment04_SuarezNavarro_Report.pdf   the report
```

---

## What you need

- macOS or Linux (on Windows, use WSL)
- **Java JDK 21** (I used OpenJDK 21.0.6)
- **Hazelcast 5.5.0** (one jar, Apache 2.0). `bash fetch_deps.sh` downloads it into `lib/` and checks its
  SHA-256; `build.sh` calls it automatically. No other Java dependency.
- Python 3.9 or newer, with NumPy (only to check results) and Matplotlib (plots):

```bash
pip install -r requirements.txt
```

- Free TCP ports 5701–5704 on 127.0.0.1 (the members listen there).

---

## How to run it

**1. Compile** (downloads Hazelcast the first time)

```bash
bash build.sh
```

**2. Check correctness** (about 2 minutes; starts four small clusters)

```bash
python3 tests/validate.py | tee results/validation_report.txt
```

It must finish with `ALL TESTS PASSED`.

**3. (Optional) Quick test, about 1 minute**

```bash
python3 scripts/run_benchmarks.py --config config_quick.json
```

**4. The real study** (about 30 minutes on my laptop)

Keep the laptop plugged in and don't use it meanwhile. On a Mac, `caffeinate` stops it from sleeping:

```bash
caffeinate -i python3 scripts/run_benchmarks.py    # on Linux, drop "caffeinate -i"
```

You can run the experiments separately with `--only local`, `--only strong` or `--only weak`.
At the end it prints the folder with the results, like `results/run_YYYYMMDD-HHMMSS`.

**5. Tables and plots**

```bash
python3 scripts/analyze.py results/run_YYYYMMDD-HHMMSS
```

**6. (Optional) The PDF**

```bash
bash build_report.sh      # needs pandoc and Chrome/Chromium
```

### Running one configuration by hand

```bash
CP=java/build:lib/hazelcast-5.5.0.jar
HZ="--add-modules java.se --add-exports java.base/jdk.internal.ref=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED --add-opens java.management/sun.management=ALL-UNNAMED --add-opens jdk.management/com.sun.management.internal=ALL-UNNAMED"
ML=127.0.0.1:5701,127.0.0.1:5702
python3 scripts/gen_matrices.py --sizes 1024
# two members, 2 executor threads each (in two terminals, or with &)
java $HZ -XX:+UseSerialGC -Xmx1g -cp $CP ClusterMember --cluster demo --port 5701 --threads 2 --members $ML &
java $HZ -XX:+UseSerialGC -Xmx1g -cp $CP ClusterMember --cluster demo --port 5702 --threads 2 --members $ML &
# driver: 1 warm-up + 3 measured multiplications
java $HZ -XX:+UseSerialGC -Xmx2g -cp $CP DistMatMul --cluster demo --members $ML --expect-members 2 \
     --threads-per-member 2 --n 1024 --a data/A_n1024.bin --b data/B_n1024.bin --warmup 1 --reps 3
kill %1 %2
```

### Running on several machines

Start the members on each machine with `--host <that machine's IP>` and the same `--members` list
(all members of all machines, `ip:port`). Open the ports in the firewall. Run the driver on any
machine with the same `--members` list. Nothing in the code assumes one machine; the study in this
repository was only run on one.

---

## How I made the comparisons fair

| | What I did |
|---|---|
| Inputs | The A1–A3 dense files: values uniform in [−1, 1), seed `20262027`, float64. Every method and configuration multiplies exactly the same matrices |
| Kernel | The same i-k-j kernel everywhere; every C[i][j] is the same sum in the same order, so all results are bit-identical (checked) |
| Sizes | Strong scaling: n = 512, 1024, 2048, 3072. Weak scaling: n = 1024·W^(1/3) for W = 1…4 members (1024, 1290, 1477, 1625), so the arithmetic per worker (2n³/W) stays constant |
| Configurations | (members × threads per member) = 1×1, 2×1, 3×1, 4×1, 1×2, 2×2, 1×4. The laptop has 2 physical cores and 4 logical processors, so 4 workers fill all logical processors. *m×1* adds processes, *1×t* adds threads inside one process |
| References | `seq` and `static` with p = 2 and 4, measured in the same session with the A3 protocol (kernel-only timing) |
| Fresh state | A **new cluster** (new JVMs, new cluster name) for every distributed configuration; fresh JVM for every local run |
| Hazelcast settings | Backups off (backup-count 0) so data is stored once; TCP/IP discovery only; executor pool size = t; logging WARNING; phone-home off (`java/HzSetup.java`) |
| Timer | `System.nanoTime()` (monotonic). The checksum of C is printed after every multiplication; C is filled with NaN before each one, so a missing block would show |
| Warm-up | 2 warm-up multiplications for n ≤ 1024, 1 above (not in the statistics) |
| Repetitions | 5 measured multiplications per configuration; median and IQR. Speedup and efficiency use the medians |
| Order | Configurations shuffled within each size (fixed seed) |
| Memory | Peak RSS of **every** JVM (driver and each member) from `wait4` (`ru_maxrss`); per member also heap in use after GC and the bytes of the map entries it stores (`LocalMapStats.getOwnedEntryMemoryCost`); analytical array storage 3·8n² |
| Transfers | Payload bytes (8 bytes per element; serialization headers not counted): scatter 16n², gather 8n², B redistribution 8n²(m−1) in total. Loopback only: no physical network |
| Limits (set beforehand) | 120 s per multiplication and 4 GB of estimated data in all processes together. A configuration that exceeds them is recorded as `timeout` / `skipped_memory_budget`, and larger sizes are not attempted for it |

---

## Raw data format

- `results/run_*/raw_measurements.csv`: one row per warm-up or measured multiplication (all experiments):
  experiment, method, n, members, threads per member, workers, kind, time, and for the distributed runs the
  three phases.
- `results/run_*/tasks.csv`: one row per distributed task and multiplication: block, member, thread, rows,
  whether A was local, whether this task fetched B, time to read A / obtain B / fetch B / kernel / store C,
  remote B bytes.
- `results/run_*/members.csv`: one row per member process and configuration: start-up time, peak RSS,
  CPU time, heap after GC, map entry bytes, thread counts.
- `results/run_*/runs.csv`: one row per configuration: status, cluster start time, client connect time,
  payload bytes, driver memory, command.
- `results/run_*/environment.json`: machine and version details.

---

## Limitations

Everything ran on one laptop with 2 cores. Adding member processes does not add hardware: all workers share
the same 2 cores, caches and memory bandwidth, and "network" transfers are copies through the loopback
interface. The results therefore describe the cost of the framework and of the data movement, not the
speedup a real cluster would reach. The driver and the member JVMs also compete for the same CPU. macOS does
not allow pinning processes to cores.

---

## Use of AI

I used Claude (Anthropic) to help design the experiment, explain Hazelcast, generate the first versions of
the code and scripts, and draft the report from my results. I ran all the experiments on my own computer,
and every number in the report comes from those runs.
