#!/bin/bash
# Gameplay benchmark with nobody at the keyboard.
#
#   tools/bench_game.sh <name> [perf]
#
# Skips the opening cinematic, picks "Continue" in the menu (loads the most recent save) and
# stays wherever the save leaves Wander; measures FPS with MangoHud between 75 and 115 s and
# takes screenshots at 80, 95 and 110 s. With "perf", also records a profile (perf record) from
# 80 to 110 s. Results go to ~/.local/share/sotcport/bench/<name>. SOTC_PERF=1 enables
# the GPU layer's temporary counters (third_party/shadps4/shim/perf_temp.h) in the log.
# SOTC_BUILD=<dir> runs the executable of another build directory (default build-clang).
# SOTC_BENCH_ROUTE=walk: instead of staying at the save point, walks out of it and runs across
# the plain for two minutes (the 60 FPS goal is measured there), FPS measured from 80 to 190 s,
# with frame time percentiles and the share of frames slower than 16.7 ms.
#
# The key script (SOTC_KEY_SCRIPT, src/input/input.cpp) depends on loading times: if the
# game's pace changes, check the screenshots.

set -e
# The game dump to run (the folder with eboot.bin and the pak*.psarc files).
: "${SOTC_GAME:?set SOTC_GAME to your game dump folder (the one with eboot.bin)}"
name=${1:?usage: bench_game.sh <name> [perf]}
root=$(cd "$(dirname "$0")/.." && pwd)
out=~/.local/share/sotcport/bench/$name
screenshots=~/.local/share/sotcport/gpu/screenshots
mkdir -p "$out" "$screenshots"
rm -f "${out:?}"/*.csv
# Screenshots from earlier sessions are kept (the user keeps them as a record); only the ones
# taken during this run are copied into the results.
marker=$out/.started
touch "$marker"

# The user's MangoHud configuration, with logging on from the start and written here.
conf=$out/mangohud.conf
grep -vE "^(output_folder|log_duration|autostart_log)=" ~/.config/MangoHud/MangoHud.conf > "$conf" 2>/dev/null || true
printf "output_folder=%s\nlog_duration=200\nautostart_log=1\n" "$out" >> "$conf"

if [ "$2" = perf ]; then
    (
        while ! pid=$(pgrep -x sotcport); do sleep 0.2; done
        sleep 80
        perf record -F 999 -g -p "$pid" -o "$out/perf.data" -- sleep 30 > /dev/null 2>&1
    ) &
fi

# Skip the opening cinematic, "Continue", load the most recent save (~59 s).
keys="36:Escape,38:K,39:K,40:K,41.5:Return,50:Return,54:Return,58:Return"
shots="80,95,110"
duration=120
window_from=75
window_to=115
if [ "${SOTC_BENCH_ROUTE:-}" = walk ]; then
    # Out of the save point and across the plain: run, turn a little, run.
    keys="$keys,75:S:6,82:W:30,112:D:2,114:W:30,144:A:3,147:W:45"
    shots="100,130,160,190"
    duration=200
    window_from=80
    window_to=190
fi

cd "$root"
SOTC_KEY_SCRIPT="$keys" SOTC_SCREENSHOTS="$shots" MANGOHUD_CONFIGFILE=$conf \
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
    timeout $duration mangohud "./${SOTC_BUILD:-build-clang}/sotcport" "$SOTC_GAME" \
    > "$out/log.txt" 2>&1 || true
wait

i=0
for f in $(find "$screenshots" -name '*.png' -newer "$marker" | sort); do
    magick "$f" -resize 25% "$out/screenshot$i.jpg"
    i=$((i + 1))
done

python3 - "$out" "$window_from" "$window_to" <<'PY'
import csv, glob, sys
rows = list(csv.reader(open(glob.glob(sys.argv[1] + "/*.csv")[0])))[3:]
t0 = int(rows[0][15])
lo, hi = float(sys.argv[2]), float(sys.argv[3])
window = [r for r in rows if lo <= (int(r[15]) - t0) / 1e9 <= hi]
fps = sorted(float(r[0]) for r in window)
frametimes = sorted((float(r[1]) for r in window), reverse=True)
def worst(share):
    return frametimes[int(len(frametimes) * share)]
slow = sum(1 for t in frametimes if t > 16.7) / len(frametimes) * 100
print(f"fps {lo:.0f}-{hi:.0f} s: mean {sum(fps) / len(fps):.1f}  min {fps[0]:.1f}  "
      f"worst 5% frame time {worst(0.05):.1f} ms  worst 1% {worst(0.01):.1f} ms  "
      f"frames over 16.7 ms {slow:.1f}%")
PY
