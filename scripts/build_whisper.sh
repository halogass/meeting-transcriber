#!/usr/bin/env bash
# Clone and build whisper.cpp (whisper-server + whisper-cli).
# Uses CUDA when nvcc is available, otherwise a CPU (OpenBLAS if present) build.
#
#   WHISPER_CPP_REF=v1.9.5  scripts/build_whisper.sh
#   FORCE_CPU=1             scripts/build_whisper.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${WHISPER_CPP_REF:-v1.9.5}"
SRC="$ROOT/vendor/whisper.cpp"
OUT="$ROOT/bin"

mkdir -p "$ROOT/vendor" "$OUT"

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/ggml-org/whisper.cpp.git "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin tag "$REF" && git -C "$SRC" checkout -q "$REF"
fi

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DBUILD_SHARED_LIBS=OFF)

if [ -z "${FORCE_CPU:-}" ] && command -v nvcc >/dev/null 2>&1; then
  echo ">> CUDA detected: $(nvcc --version | tail -1)"
  CMAKE_ARGS+=(-DGGML_CUDA=ON)
  # CUDA 12.0-12.3 refuse GCC 13+; fall back to an older host compiler when available.
  CUDA_MAJOR_MINOR="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
  GCC_MAJOR="$(gcc -dumpversion | cut -d. -f1)"
  if [ "$GCC_MAJOR" -ge 13 ] && awk "BEGIN{exit !($CUDA_MAJOR_MINOR < 12.4)}"; then
    for v in 12 11; do
      if command -v "g++-$v" >/dev/null 2>&1; then
        echo ">> Using g++-$v as CUDA host compiler (CUDA $CUDA_MAJOR_MINOR, gcc $GCC_MAJOR)"
        CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-$v)")
        break
      fi
    done
  fi
  # Build only for the GPUs present (much faster); override with CUDA_ARCHS="75;86".
  if [ -n "${CUDA_ARCHS:-}" ]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS")
  elif command -v nvidia-smi >/dev/null 2>&1; then
    ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -u | paste -sd';')"
    [ -n "$ARCHS" ] && CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$ARCHS")
  fi
else
  echo ">> No CUDA toolkit found: building CPU version"
  if pkg-config --exists openblas 2>/dev/null; then
    CMAKE_ARGS+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS)
  fi
fi

cmake -S "$SRC" -B "$SRC/build" "${CMAKE_ARGS[@]}"
cmake --build "$SRC/build" -j"$(nproc)" --target whisper-server whisper-cli

cp "$SRC/build/bin/whisper-server" "$SRC/build/bin/whisper-cli" "$OUT/"
echo ">> Installed: $OUT/whisper-server, $OUT/whisper-cli"
