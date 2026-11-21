#!/usr/bin/env bash
# Parallel, resumable ranged download with hash verification (for slow, per-connection-throttled links).
#
#   bash scripts/pget.sh URL OUTFILE SIZE HASH [PARTS]
#
# HASH is a SHA-256 hex digest (optionally "sha256:"-prefixed, as in uv.lock) or "sha1:<hex>" (Maven Central
# publishes <artifact>.sha1). SIZE and SHA-256 come from uv.lock (wheels = [{ url, hash, size }]) or the PyPI
# JSON API; for Maven use `curl -sI URL` (Content-Length) and `curl -s URL.sha1`. Each part
# resumes from its current length, so re-running after an interruption continues where it stopped.
# Install the result with:  uv pip install --no-deps OUTFILE   (then run commands with `uv run --no-sync`).
set -u
if [ "$#" -lt 4 ]; then
  echo "usage: $0 URL OUTFILE SIZE HASH [PARTS]" >&2
  exit 2
fi
URL="$1"; OUT="$2"; SIZE="$3"; N="${5:-8}"
case "$4" in
  sha1:*) SUM=sha1sum; SHA="${4#sha1:}" ;;
  *) SUM=sha256sum; SHA="${4#sha256:}" ;;
esac
CHUNK=$(( (SIZE + N - 1) / N ))
mkdir -p "$(dirname "$OUT")"

fetch_part() {
  local i=$1
  local start=$(( i * CHUNK ))
  local end=$(( start + CHUNK - 1 ))
  if [ "$end" -ge "$SIZE" ]; then end=$(( SIZE - 1 )); fi
  local want=$(( end - start + 1 ))
  local part="$OUT.part$i"
  touch "$part"
  for attempt in $(seq 1 500); do
    local have
    have=$(stat -c %s "$part")
    if [ "$have" -ge "$want" ]; then echo "part $i done"; return 0; fi
    # --fail: never append an HTTP error body to the part file
    curl -sS -L --fail --connect-timeout 30 --speed-limit 2000 --speed-time 60 \
      -r "$(( start + have ))-$end" "$URL" >> "$part" || true
    sleep 2
  done
  echo "part $i FAILED after $attempt attempts"; return 1
}

for i in $(seq 0 $(( N - 1 ))); do fetch_part "$i" & done
wait

: > "$OUT.tmp"
for i in $(seq 0 $(( N - 1 ))); do cat "$OUT.part$i" >> "$OUT.tmp"; done
got=$(stat -c %s "$OUT.tmp")
echo "assembled $got / $SIZE bytes"
if echo "$SHA  $OUT.tmp" | "$SUM" -c - ; then
  mv "$OUT.tmp" "$OUT"; rm -f "$OUT".part*; echo "OK $OUT"
else
  echo "HASH MISMATCH (parts kept; delete $OUT.part* to restart from scratch)"; rm -f "$OUT.tmp"; exit 1
fi
