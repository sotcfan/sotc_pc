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
# The game runs with a data folder of its own (SOTC_BENCH_DATA, default
# ~/.local/share/sotcport-bench): with the player's folder, "Continue" followed the player's
# progress (so the scene changed between runs) and every run rewrote the player's profile save.
# Before each run the saves are restored from $SOTC_BENCH_DATA/saves.snapshot (made once from the
# player's saves), so every run starts at the same place (the GPU layer's folder, with the shader
# cache and the screenshots, stays the player's: it does not follow --data). SOTC_BENCH_SLOT=<n> makes save slot n (SAVEDATA0ShadowProgress<n>) the
# most recent one, the one "Continue" loads (the game sorts the saves by the time of
# sce_sys/param.bin).
# SOTC_BENCH_ROUTE=forest: the 60 FPS benchmark (see below). SOTC_BENCH_ROUTE=walk: instead of
# staying at the save point, walks out of it and runs across
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

# Skip the opening cinematic, "Continue", load the most recent save (~59 s), with the benchmark's
# own bindings (controls.ini, written below).
keys="36:Escape,38:K,39:K,40:K,41.5:Space,50:Space,54:Space,58:Space"
# SOTC_BENCH_MENU_DELAY=<s> presses those keys that many seconds later, for a slow start (a weak
# GPU, or no shader cache): the opening cinematic and the title screen wait for them. Add the
# same delay to SOTC_BENCH_SHOTS and SOTC_BENCH_DURATION.
if [ -n "${SOTC_BENCH_MENU_DELAY:-}" ]; then
    keys=$(echo "$keys" | tr ',' '\n' | awk -F: -v d="$SOTC_BENCH_MENU_DELAY" \
        '{ $1 = $1 + d; out = out (NR > 1 ? "," : "") $1 ":" $2 } END { print out }')
fi
shots="80,95,110"
duration=120
window_from=75
window_to=115
if [ "${SOTC_BENCH_ROUTE:-}" = forest ]; then
    # The player's save at the shrine of the Forest of Giants (2026-10-14, in the benchmark's
    # saves.snapshot, slot 0): stand up (a tap), walk away from the shrine for 3.5 s, then turn
    # the camera all the way round for 70 s, through the heaviest part of the game.
    keys="$keys,75:S:0.3,77:S:3.5,82:Right:70"
    shots="90,100,110,120,130,140,150"
    duration=160
    window_from=85
    window_to=150
elif [ "${SOTC_BENCH_ROUTE:-}" = walk ]; then
    # Out of the save point and across the plain: run, turn a little, run.
    keys="$keys,75:S:6,82:W:30,112:D:2,114:W:30,144:A:3,147:W:45"
    shots="100,130,160,190"
    duration=200
    window_from=80
    window_to=190
fi

# The configuration the game is played with: the two-stage draw pipeline and asynchronous
# readbacks (OPTIONS.md), uncapped unless SOTC_FPS_LIMIT is given. Set any of them in the
# environment to run otherwise.
export SOTC_DRAW_PIPE=${SOTC_DRAW_PIPE-1} SOTC_ASYNC_READBACK=${SOTC_ASYNC_READBACK-1}
# GPU timestamps around every command buffer (OPTIONS.md): the summary gives how busy the GPU
# was, the measure of the native renderer scope (2026-10-20: the GPU at 100%). No FPS cost
# measured (95.3 FPS with and without on the forest route).
export SOTC_GPU_TIMING=${SOTC_GPU_TIMING-1}
if [ -z "${SOTC_FPS_LIMIT:-}" ]; then
    export SOTC_UNCAPPED=${SOTC_UNCAPPED-1}
fi

# SOTC_BENCH_SHOTS=<s,s,...> takes the screenshots at other times.
shots=${SOTC_BENCH_SHOTS:-$shots}
# With SOTC_INPUT_REPLAY=<file> (src/input/input.cpp) a recorded play drives the game from the
# save load on (the keys above still go through the menus; leave SOTC_BENCH_ROUTE unset);
# SOTC_BENCH_DURATION=<s> and SOTC_BENCH_WINDOW=<from>-<to> change how long it runs and which
# seconds are measured.
# SOTC_BENCH_GAME_ARGS="<args>" passes them to the game as its command line (after "--"), for
# example the engine's console variables: "+r_enableCascadeShadows=0".
# SOTC_BENCH_KEYS=<script> adds keys to the script (for example "75:F7" for the diagnostic
# sequence, src/input/input.cpp).
keys="$keys${SOTC_BENCH_KEYS:+,$SOTC_BENCH_KEYS}"
keys=${keys#,}
duration=${SOTC_BENCH_DURATION:-$duration}
if [ -n "${SOTC_BENCH_WINDOW:-}" ]; then
    window_from=${SOTC_BENCH_WINDOW%-*}
    window_to=${SOTC_BENCH_WINDOW#*-}
fi

# The benchmark's own data folder (see above).
data=${SOTC_BENCH_DATA:-$HOME/.local/share/sotcport-bench}
player_data=$HOME/.local/share/sotcport
mkdir -p "$data"
if [ ! -d "$data/saves.snapshot" ]; then
    cp -a "$player_data/saves" "$data/saves.snapshot"
fi
# The key script needs keys of its own, whatever the player binds in their controls.ini (the
# player's folder is not this one, but a new controls.ini gets the defaults, and those follow the
# player's taste: K stopped being down on 2026-10-16 and the menu went to Settings instead of
# Continue). So the benchmark writes its bindings every run.
cat > "$data/controls.ini" <<'INI'
# Written by tools/bench_game.sh on every run (the key scripts depend on it).
W = LeftStickUp
S = LeftStickDown
A = LeftStickLeft
D = LeftStickRight
Up = RightStickUp
Down = RightStickDown
Left = RightStickLeft
Right = RightStickRight
Space = Cross
Return = Cross
Backspace = Circle
I = DpadUp
K = DpadDown
J = DpadLeft
L = DpadRight
Escape = Options
Tab = TouchPad
Z = L2
X = R2
INI
rm -rf "${data:?}/saves"
cp -a "$data/saves.snapshot" "$data/saves"
if [ -n "${SOTC_BENCH_SLOT:-}" ]; then
    touch "$data/saves/1/SAVEDATA0ShadowProgress$SOTC_BENCH_SLOT/sce_sys/param.bin"
fi

cd "$root"
SOTC_KEY_SCRIPT="$keys" SOTC_SCREENSHOTS="$shots" MANGOHUD_CONFIGFILE=$conf \
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
    timeout $duration mangohud "./${SOTC_BUILD:-build-clang}/sotcport" "$SOTC_GAME" --data "$data" \
    ${SOTC_BENCH_GAME_ARGS:+-- $SOTC_BENCH_GAME_ARGS} > "$out/log.txt" 2>&1 || true
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
# GPU busy time from SOTC_GPU_TIMING: each scheduler (the game's command buffers, presentation)
# logs "GPU timing: busy <x> ms/s" every 5 s, a few ms apart; the reports of one period are
# summed, the periods inside the window averaged. The log's clock starts with the game, close
# enough to MangoHud's.
import re
reports = []
for line in open(sys.argv[1] + "/log.txt", errors="replace"):
    m = re.match(r"\[\s*([\d.]+)\].*GPU timing: busy ([\d.]+) ms/s", line)
    if m and lo <= float(m[1]) <= hi:
        reports.append((float(m[1]), float(m[2])))
periods = []
for t, busy in sorted(reports):
    if periods and t - periods[-1][0] < 1.0:
        periods[-1][1] += busy
    else:
        periods.append([t, busy])
if periods:
    busy = sum(p[1] for p in periods) / len(periods)
    print(f"GPU busy {lo:.0f}-{hi:.0f} s: {busy:.0f} ms/s ({busy / 10:.0f}%), "
          f"{busy / (sum(fps) / len(fps)):.2f} ms per frame")
PY
