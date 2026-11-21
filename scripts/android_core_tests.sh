#!/usr/bin/env bash
# Compile and run the Android core's JVM unit tests WITHOUT Gradle / Android Studio.
#
#   bash scripts/android_core_tests.sh fetch   # once: Kotlin compiler + test jars (~120 MB, hash-verified)
#   bash scripts/android_core_tests.sh         # compile android/core and run its JUnit tests
#
# The jars come from Maven Central into data/tooling/jvm/ (gitignored), large ones through scripts/pget.sh.
# Needs a JDK 17+ on PATH. Gradle (`cd android && ./gradlew :core:test`) runs the same tests.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd -W 2>/dev/null || pwd)" # Windows-style path for java.exe under Git Bash
J="$ROOT/data/tooling/jvm"
SEP=":"
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) SEP=";" ;; esac
KOTLIN=2.4.21
ORT=1.32.0
ARTIFACTS="
org/jetbrains/kotlin/kotlin-compiler-embeddable/$KOTLIN
org/jetbrains/kotlin/kotlin-stdlib/$KOTLIN
org/jetbrains/kotlin/kotlin-script-runtime/$KOTLIN
org/jetbrains/kotlin/kotlin-reflect/1.6.10
org/jetbrains/kotlin/kotlin-daemon-embeddable/$KOTLIN
org/jetbrains/kotlin/kotlin-build-tools-api/$KOTLIN
org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.8.0
org/jetbrains/annotations/13.0
com/microsoft/onnxruntime/onnxruntime/$ORT
junit/junit/4.13.2
org/hamcrest/hamcrest-core/1.3
org/json/json/20260814
"
jar() { # group/artifact/version -> local jar path
  local ver=${1##*/} rest=${1%/*}
  echo "$J/${rest##*/}-$ver.jar"
}

if [ "${1:-}" = "fetch" ]; then
  mkdir -p "$J"
  M=https://repo1.maven.org/maven2
  for a in $ARTIFACTS; do
    out=$(jar "$a")
    f=$(basename "$out")
    [ -f "$out" ] && { echo "have $f"; continue; }
    url="$M/$a/$f"
    sha=$(curl -sS --fail --max-time 60 "$url.sha1" | cut -c1-40)
    size=$(curl -sSI --fail --max-time 60 "$url" | grep -i '^content-length' | tr -dc '0-9')
    if [ "$size" -gt 4000000 ]; then
      bash scripts/pget.sh "$url" "$out" "$size" "sha1:$sha" 12 | tail -1
    else
      curl -sS --fail --max-time 600 -o "$out.tmp" "$url"
      echo "$sha  $out.tmp" | sha1sum -c - >/dev/null && mv "$out.tmp" "$out" && echo "OK $f"
    fi
  done
  exit 0
fi

for a in $ARTIFACTS; do
  [ -f "$(jar "$a")" ] || { echo "missing $(basename "$(jar "$a")"): run  bash scripts/android_core_tests.sh fetch" >&2; exit 2; }
done
cp_of() { local IFS="$SEP"; echo "$*"; }
COMPILER=$(cp_of "$(jar org/jetbrains/kotlin/kotlin-compiler-embeddable/$KOTLIN)" \
  "$(jar org/jetbrains/kotlin/kotlin-stdlib/$KOTLIN)" "$(jar org/jetbrains/kotlin/kotlin-script-runtime/$KOTLIN)" \
  "$(jar org/jetbrains/kotlin/kotlin-reflect/1.6.10)" "$(jar org/jetbrains/kotlin/kotlin-daemon-embeddable/$KOTLIN)" \
  "$(jar org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.8.0)" "$(jar org/jetbrains/annotations/13.0)" \
  "$(jar org/jetbrains/kotlin/kotlin-build-tools-api/$KOTLIN)")
LIBS=$(cp_of "$(jar org/jetbrains/kotlin/kotlin-stdlib/$KOTLIN)" "$(jar org/json/json/20260814)" \
  "$(jar com/microsoft/onnxruntime/onnxruntime/$ORT)")
TESTLIBS=$(cp_of "$LIBS" "$(jar junit/junit/4.13.2)" "$(jar org/hamcrest/hamcrest-core/1.3)")
CORE="$ROOT/android/core"
OUT="$CORE/build/jvm-tests"
rm -rf "$OUT" && mkdir -p "$OUT/main" "$OUT/test"

kotlinc() { java -Xmx1g -cp "$COMPILER" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -no-reflect -jvm-target 17 "$@"; }
echo "compiling android/core (main)..."
kotlinc -cp "$LIBS" -d "$OUT/main" "$CORE/src/main/kotlin"
echo "compiling android/core (test)..."
kotlinc -cp "$(cp_of "$OUT/main" "$TESTLIBS")" -d "$OUT/test" "$CORE/src/test/kotlin"
TESTS=$(cd "$CORE/src/test/kotlin" && grep -rl '@Test' . | sed 's#^\./##; s#\.kt$##; s#/#.#g')
echo "running: $(echo "$TESTS" | tr '\n' ' ')"
# shellcheck disable=SC2086
java -cp "$(cp_of "$OUT/test" "$OUT/main" "$CORE/src/test/resources" "$TESTLIBS")" org.junit.runner.JUnitCore $TESTS
