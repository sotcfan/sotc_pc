#!/bin/bash
# Retrains the PGO profile used by build-clang (SOTC_PGO=use, see the top-level CMakeLists.txt).
#
#   tools/pgo_train.sh
#
# Builds the instrumented executable in build-pgogen (SOTC_PGO=generate), runs the gameplay
# benchmark twice (tools/bench_game.sh: nobody at the keyboard, loads the most recent save) and
# the opening for 100 s, merges the profiles into ~/.local/share/sotcport/pgo/sotcport.profdata
# and rebuilds build-clang with it. Retrain after larger changes to the GPU layer: functions
# changed since the profile was taken compile without it (still correct, just not tuned).
set -e
# The game dump to run (the folder with eboot.bin and the pak*.psarc files).
: "${SOTC_GAME:?set SOTC_GAME to your game dump folder (the one with eboot.bin)}"
root=$(cd "$(dirname "$0")/.." && pwd)
pgo=~/.local/share/sotcport/pgo
cd "$root"
CC=clang CXX=clang++ cmake -B build-pgogen -G Ninja -DSOTC_PGO=generate > /dev/null
systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 cmake --build build-pgogen
mkdir -p "$pgo"
rm -f "$pgo"/*.profraw
# The instrumented build writes its counters every 30 s (src/diag/pgo_dump.cpp).
for i in 1 2; do
    SOTC_BUILD=build-pgogen tools/bench_game.sh pgo_train$i
done
systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
    timeout 100 ./build-pgogen/sotcport "$SOTC_GAME" > /dev/null 2>&1 || true
llvm-profdata merge -o "$pgo/sotcport.profdata" "$pgo"/*.profraw
# Everything is recompiled: the build does not track the profile file, and objects compiled with
# different profiles cannot be linked together (ThinLTO "ProfileSummary" conflicts).
systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 \
    cmake --build build-clang --clean-first
echo "profile: $pgo/sotcport.profdata"
