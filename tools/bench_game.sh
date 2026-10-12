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

cd "$root"
SOTC_KEY_SCRIPT="36:Escape,38:K,39:K,40:K,41.5:Return,50:Return,54:Return,58:Return" \
SOTC_SCREENSHOTS="80,95,110" MANGOHUD_CONFIGFILE=$conf \
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
    timeout 120 mangohud "./${SOTC_BUILD:-build-clang}/sotcport" "$SOTC_GAME" \
    > "$out/log.txt" 2>&1 || true
wait

i=0
for f in $(find "$screenshots" -name '*.png' -newer "$marker" | sort); do
    magick "$f" -resize 25% "$out/screenshot$i.jpg"
    i=$((i + 1))
done

python3 - "$out" <<'PY'
import csv, glob, sys
rows = list(csv.reader(open(glob.glob(sys.argv[1] + "/*.csv")[0])))[3:]
t0 = int(rows[0][15])
window = [r for r in rows if 75 <= (int(r[15]) - t0) / 1e9 <= 115]
fps = sorted(float(r[0]) for r in window)
frametimes = sorted((float(r[1]) for r in window), reverse=True)
print(f"fps 75-115 s: mean {sum(fps) / len(fps):.1f}  min {fps[0]:.1f}  "
      f"worst 5% frame time {frametimes[len(frametimes) // 20]:.1f} ms")
PY
