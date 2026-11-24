#!/usr/bin/env bash
# Compiles the Java code with the fixed settings used in the study.
# Needs lib/hazelcast-5.5.0.jar (downloaded and checked by fetch_deps.sh).
set -euo pipefail
cd "$(dirname "$0")"
[ -f lib/hazelcast-5.5.0.jar ] || bash fetch_deps.sh
mkdir -p java/build
echo "[build] javac $(javac -version 2>&1 | cut -d' ' -f2) --release 21"
javac -Xlint:all,-serial --release 21 -cp lib/hazelcast-5.5.0.jar -d java/build java/*.java
echo "[build] done"
