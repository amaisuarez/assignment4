#!/usr/bin/env bash
# Downloads the only dependency, Hazelcast 5.5.0 (open-source edition, Apache 2.0),
# from Maven Central and checks its SHA-256.
set -euo pipefail
cd "$(dirname "$0")"
JAR=lib/hazelcast-5.5.0.jar
SHA=cf560b40b988526e780e4fae12cb6fe01c5158bb9e00839882857f32b544a19e
mkdir -p lib
[ -f "$JAR" ] || curl -sSfL -o "$JAR" https://repo1.maven.org/maven2/com/hazelcast/hazelcast/5.5.0/hazelcast-5.5.0.jar
echo "$SHA  $JAR" | shasum -a 256 -c -
