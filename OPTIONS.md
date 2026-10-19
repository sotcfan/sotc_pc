# sotcport options

Everything sotcport can be told at startup: command line options and environment variables.

## Syntax

```sh
VARIABLE=value OTHER=value ./build-clang/sotcport /path/to/your/dump [options] [-- game arguments]
```

Environment variables go before the command, separated by spaces. Switches turn on with `=1`;
the ones that are on by default turn off with `=0` (or with their `NO_` variant, as noted).
Anything after `--` is passed to the game as its own command line.

A typical play session (steady 30 FPS, two-stage draw pipeline), with a memory cap so a runaway
process cannot take the machine down:

```sh
SOTC_FPS_LIMIT=30 SOTC_DRAW_PIPE=1 \
  systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
  ./build-clang/sotcport ~/path/to/CUSA08809
```

## Command line options

| Option | What it does |
|---|---|
| `--log <level>` | Log detail: `trace`, `debug`, `info` (default), `warning`, `error`. |
| `--data <folder>` | Where saves and the game's writable data live (default `~/.local/share/sotcport`). |
| `--trace <file>` | Records, in order, the first call to each Sony function the game imports. |
| `--dump-after <s>` | After `s` seconds every game thread prints where it is (`kill -USR1 <pid>` does the same at any time). |

## Building

| CMake option | What it does |
|---|---|
| `-DSOTC_LTO=ON` | Link-time optimization (ThinLTO; needs clang). |
| `-DSOTC_PGO=generate` | Instrumented build that records a profile while the game runs (`tools/pgo_train.sh`). |
| `-DSOTC_PGO=use` | Optimizes with the recorded profile. Retrain after big changes to the GPU layer; the final build needs `--clean-first`. |

## Frame rate

**Play at a steady 30 FPS** (`SOTC_FPS_LIMIT=30`); 60 is experimental.

- The game's physics (Wander's cape, Agro) steps at a fixed 1/60 s, so it only behaves at a steady
  30 or 60 FPS. Other rates, even perfectly steady ones (40, 45), make the cape whip around and
  Agro overshoot.
- At 60 FPS the physics is fine, but the game's logic is not: an original PS4 never runs it at 60,
  and some of it breaks (the fifth colossus flies straight up out of its arena). `SOTC_NEO_MODE=1`
  (the PS4 Pro, which has an official 60 FPS mode) may fix that; inconclusive.
- 60 FPS is also not always reached: in heavy scenes the GPU layer cannot keep up.

| Variable | What it does |
|---|---|
| `SOTC_FPS_LIMIT=<n>` | Frame limiter with even frame pacing. With a list, it starts at the first rate, drops to the next when frames keep missing the beat, and climbs back after a while with room to spare. A single value (`=30`) holds that rate. Use 30 (recommended) or 60 (experimental) only. For 60, add `SOTC_NEO_MODE=1`: the game's official 60 FPS mode is the PS4 Pro's (whether that fixes all of its logic at 60 is not confirmed yet). |
| `SOTC_UNCAPPED=1` | No frame rate cap: flips are presented at once and the 60 Hz vblank only drives the game's events. For measuring; the physics misbehaves. |
| `SOTC_NEO_MODE=1` | Tells the game it runs on a PS4 Pro, where it officially supports 60 FPS. On an original PS4 the game logic breaks at 60 (the fifth colossus flies out of its arena). Experimental. |
| `SOTC_FLIP_RATE=0` | Ignores the game's own request for 30 FPS (`sceVideoOutSetFlipRate`). Older way to uncap; `SOTC_FPS_LIMIT` paces frames better. |

## GPU layer: performance switches

| Variable | Default | What it does |
|---|---|---|
| `SOTC_DRAW_PIPE=1` | off | Two-stage draw pipeline: the command processor thread decodes while a second thread (`sotc:DrawRec`) records Vulkan draws. |
| `SOTC_ASYNC_READBACK=1` | off | When the game reads results the GPU wrote, the game thread waits for the GPU instead of the command processor thread. |
| `SOTC_ASYNC_SHADERS=1` | off | Builds new graphics pipelines on background threads (`sotc:ShaderBuild`), so a new area stutters less. A draw whose pipeline is not ready yet is skipped for that frame. |
| `SOTC_VK_RECORD_THREAD=0` | on | Turns off recording Vulkan commands on a separate thread. |
| `SOTC_FLUSH_EVERY=<n>` | 256 | With the draw pipeline, submits GPU work early every `n` draws and dispatches, so the GPU starts sooner. |
| `SOTC_SLOW_PENDING_OPS=1` | off | Goes back to checking pending GPU operations with a system call per draw (A/B measurements). |
| `SOTC_NO_SHADOW_COPY=1` | on | The game copies tables of constants and descriptors with a small compute shader, and later draws read them: the command processor thread then had to wait for the GPU (~0.5 ms, up to ~1,500 times a second). The CPU computes those copies as the commands arrive, so nothing waits, and the GPU's memory gets the CPU's bytes in order instead of running the copy (+20% in the forest, 2026-10-19). It was off from 2026-10-14 to 2026-10-19: objects blinked after the bridge cutscene when the GPU's own copy diverged from the CPU's. This turns it off. `SOTC_VERIFY_SHADOW_COPY=<n>` runs every n-th copy on the GPU too and compares (diagnostics). |
| `SOTC_NO_SRT_PATCHES=1` | on | Constants a shader reads that the GPU itself computed are copied on the GPU, right before the draw, instead of being read back on the CPU. This turns it off. |
| `SOTC_NO_BIND_MEMO=1` | on | Most draws bind exactly the buffers, textures and samplers of the draw before them: their bindings are reused and only the per-draw constants are redone. This turns it off. |
| `SOTC_NO_TARGET_CACHE=1` | on | Render targets are looked up again only when their registers change. This turns it off. |
| `SOTC_NO_TEXTURE_CACHE_LOOKUP=1` | on | Textures are looked up in the texture cache again only when the cache's set of images changes. This turns it off. |
| `SOTC_ASYNC_SUBMIT=1` | off | The Vulkan recording thread ends and submits the command buffers itself, so the thread asking for a submission does not wait for it. Experimental. |

## GPU layer: correctness switches (being tested)

| Variable | Default | What it does |
|---|---|---|
| `SOTC_NO_REPROTECT=1` | on | A new mapping, or a protection change by the game, over memory a GPU cache watches gets the write protection back (otherwise the game's later writes there go unnoticed). This turns it off. |
| `SOTC_LOD_STATS=1` | off | Answers the game's texture streaming requests (`IT_GET_LOD_STATS`): textures drawn since the last request are reported as needing their finest level, the others as unused. Without it every texture the game asks about reads as needing its finest level; suspected cause of textures switching detail levels at the last colossus (being tested). |

## Diagnostics

| Variable | What it does |
|---|---|
| `SOTC_INPUT_RECORD=<file>` | Writes down the controller state the game reads (and the F9 presses), counted in the game's controller reads. |
| `SOTC_INPUT_REPLAY=<file>` | Plays such a recording back instead of the keyboard and controllers, menus included, taking the screenshots at the recorded F9 presses: a stretch of play reproduced as many times as a test needs. |
| `SOTC_LOG_PAD_CALLERS=1` | Logs every place in the game's code that reads the controller, once each, with the buffer it reads into: the starting point for hooking the game's own input code (split actions, `src/input/game_hooks.cpp`). |
| `SOTC_WATCH_DATA=<address>[:<length>[:w]]` | Hardware watchpoint (Linux): logs every 2 s which instructions of the game read or write that memory (`length` 1, 2, 4 or 8, default 4; `w` for writes only). For finding the code that uses a value (`src/diag/data_watch.h`). |
| `SOTC_PROBE_CAMERA=1 \| <slots> \| stick \| orient \| menu` | Temporary: the probes that found the camera and menu code for the mouse (`src/diag/camera_probe.cpp`). |
| `SOTC_MOUSE_CAMERA_TEST="<from>:<to>:<yaw>:<pitch>"` | Turns the free camera by itself between those seconds of the run, `yaw`/`pitch` radians per camera update, as the mouse would (right and down positive), and logs the angles: for testing the native mouse camera with a key script (`src/input/mouse_camera.cpp`). |
| `SOTC_MOUSE_SCRIPT="80:move:400:500,82:click:400:545"` | Moves the menu cursor, clicks, presses (`down`) and releases (`up`) the left button, clicks the right button (`right`) or turns the wheel (`wheelup`, `wheeldown`) at those seconds (the `SOTC_KEY_SCRIPT` clock), in the game's UI coordinates (1920 x 1080): for testing the mouse in menus with a script (`src/input/menu_mouse.cpp`). |
| F7 (key) | Diagnostic sequence: for ~4 s each, turns off one GPU layer optimization after another (shadow copies, SRT patches, bind memo, target cache, texture lookup cache, all of them, then LOD statistics on) and takes 12 screenshots per step, with `DIAG PHASE` lines in the log. For finding which one causes a defect that only shows in one place of the game; keep the camera still meanwhile. |
| `SOTC_VERIFY_IMAGES=1` | Every 64th texture bound, checks that its memory still holds what its last upload read; logs the stale ones (memory changed without the texture cache noticing): which pages changed, the game's mappings there and its recent map/unmap events. Slows the game down. |
| `SOTC_PERF=1` | Every 5 s, GPU layer counters in the log: time per kind of work, draws and dispatches per second, readbacks, write faults, draw pipeline stalls, frame pacing, VRAM in use. |
| `SOTC_SPIKES=<ms>` | Logs every frame slower than `<ms>` with what its threads waited for (with `SOTC_PERF=1`). |
| `SOTC_SCREENSHOTS="60,90"` | Saves screenshots at those seconds (F9 saves one at any time) to `~/.local/share/sotcport/gpu/screenshots`. |
| `SOTC_KEY_SCRIPT="36:Escape,41.5:Space,75:W:20"` | Presses keys by itself: `second:Key[:seconds held]`, comma separated. F9/F10/F11 also trigger their tools. |
| `SOTC_RENDERDOC=""` | Loads RenderDoc; F10 captures the frame. Needs `SDL_VIDEO_DRIVER=x11` (RenderDoc does not support Wayland). X11 is slower; use it only for captures. |
| `SOTC_DUMP_SHADERS=1` | Writes each shader's GCN, IR and SPIR-V to `gpu/shader/dumps`. Use with `SOTC_NO_SHADER_CACHE=1`, or shaders already in the cache are not translated again. |
| `SOTC_NO_SHADER_CACHE=1` | Ignores the on-disk shader cache. |
| `SOTC_WATCH_COPIES=1` | Counts the game's memory copies by memory type and call site (log, every 10 s). Slows the game down; short measurements only. |
| `SOTC_PACKET_STATS=1` | Statistics of the GPU command packets the game sends. |
| `SOTC_SRT_FETCH_PAGES=1` | The shader resource walker downloads whole pages the GPU wrote, instead of exact bytes (rules out reads the byte check would miss). |
| `SOTC_TRACE_FLATBUF=<hash>` | Logs the flattened user data and the samplers the GPU layer gives that shader (hex hash, e.g. `bd2fa723`) at every draw or dispatch. |
| `SOTC_GPU_TIMING=1` | Every 5 s, how long the GPU was busy running sotcport's command buffers (timestamps at their start and end), how long it waited between them, how many there were and the longest. |
| `SOTC_FRAME_TIMELINE=<n>` | Every n-th frame, logs a timeline in milliseconds since the previous one: game submissions (`S`, with their size), flips (`F`), presentation (`P`), the command processor waking (`a`) and running out of work (`i`, `d`), stage B running out of packets and getting some (`e`, `b`), stage A waiting for stage B (`w` with the reason, `W`), guest readbacks (`R`, `r`). |
| `SOTC_READBACK_ON_A=1` | Guest readbacks go through the command processor after draining the draw pipeline, as before (A/B measurements; by default stage B records the copy at once). |
| `SOTC_MEMORY_TYPES=1` | Every 10 s, how much the GPU layer reads and binds of each kind of game memory (Onion, Garlic, flexible) and how the game maps memory (sizes, alignments). |
| `SOTC_TRACE_SRT_READBACKS=1` | Logs a sample of the times the command processor thread had to wait for data the GPU wrote (shader, which read, address, value before and after). |
| `SOTC_STREAM_THRESHOLD=<bytes>` | Read-only buffers up to that size are copied straight from game memory at each draw (default 8 KiB) instead of coming from the GPU layer's tracked copies (rules out a missed CPU write). |

## Native renderer (work in progress)

| Variable | What it does |
|---|---|
| `SOTC_RECORD_GNM=1` | Records the game's calls to its GPU library and checks the state model against the GPU registers on every draw. Report: `~/.local/share/sotcport/gnm_recorder.txt` (every 20 s). Much slower (with validation the logos alone take over a minute): the window title then says "diagnostics on: much slower". |
| `SOTC_RECORD_FRAMES="100,200"` | With `SOTC_RECORD_GNM=1`: writes every call of the frames at those seconds to `gnm_frames/`. |
| `SOTC_VALIDATE_NATIVE=all` | With `SOTC_RECORD_GNM=1`: runs compute dispatches again on the native renderer's own Vulkan device and compares the results with the GPU layer's. A list of shader hashes (`=1902028a,2ece299b`) checks only those. |
| `SOTC_VALIDATE_NATIVE_DRAWS=all` | With `SOTC_RECORD_GNM=1`: the same for draws (vertex + pixel shader), reported per shader pair. A list of hashes (vertex or pixel shader) checks only the draws that use them. |
| `SOTC_VALIDATE_NATIVE_COUNT=<n>` | Checks per shader (default 3). A check that matched while changing nothing does not count (up to 30 of them, at most one every 2 s): an empty scene proves little. |
| `SOTC_VALIDATE_NATIVE_START=<s>` | No checks during the first `s` seconds: the logos run at normal speed, so `SOTC_KEY_SCRIPT` times stay those of a normal session. To check gameplay: `SOTC_VALIDATE_NATIVE_START=70 SOTC_KEY_SCRIPT="36:Escape,38:K,39:K,40:K,41.5:Space,50:Space,54:Space,58:Space,75:W:40"` (skips the opening cinematic, continues the save, walks). |
| `SOTC_VALIDATE_NATIVE_DUMP=1` | Saves each check to `~/.local/share/sotcport/native/`: SPIR-V, bindings, samplers, flattened user data and the addresses it was read from, the GPU layer's images over each binding, the images and buffers before and after (native and GPU layer). It also compares the inputs (images and buffers the shaders only read) with the GPU layer's, and runs the check twice to tell order-dependent shaders apart. Slow: use with a short list of hashes. |
| `SOTC_NATIVE_VK_VALIDATION=1` | Turns on the Khronos validation layer for the native renderer's device. |
| `SOTC_TRACE_GNM=<addresses>` | Intercepts the listed GPU library functions and writes the command packets they produce to `gnm_trace.txt`. |
| `SOTC_SURVEY_GNM=1` | With `SOTC_RECORD_GNM=1`: counts calls to the other GPU library functions the game uses. |
| `SOTC_DUMP_RESOURCES=".csdr"` | Saves every resource whose name ends with that suffix as the game decompresses it (e.g. its shaders). |
| `SOTC_RECORDER_TRACE` | Extra tracing inside the recorder (development). |

## A/B switches (development; leave alone for play)

Each turns off an optimization that is on by default, to check whether it causes a problem:

| Variable | Turns off |
|---|---|
| `SOTC_NO_UPLOAD_GUARD=1` | The guard that keeps a texture dirty when the game rewrites it during an upload (the dark stones / glowing door fix). |
| `SOTC_NO_UPLOAD_PRECHECK=1` | The lock-free check that skips buffer uploads with nothing modified. |
| `SOTC_NO_HOT_WINDOWS=1` | Batched downloads of recently read GPU-written memory. |
| `SOTC_NO_IMAGE_DESC_CACHE=1` | The cache of texture descriptions per T#. |
| `SOTC_NO_FINDIMAGE_SHORTCUT=1` | The texture lookup shortcut (an exact match can only be on the first page). |

Experiments that were measured and left off (they did not help): `SOTC_FAULT_WINDOW=<KiB>`
(unprotect a window around each write fault), `SOTC_PIPE_FENCE_IN_CMDBUF=1`,
`SOTC_PIPE_COMMANDS_ON_B=1`.

Draw pipeline fine tuning and measurements: `SOTC_PIPE_SPIN_US=<µs>` (how long the recording
thread spins before sleeping), `SOTC_PIPE_SYNC_FENCES=1`, `SOTC_PIPE_SYNC_COMPUTE=1` (every compute
packet drains the pipeline), `SOTC_PIPE_NO_FENCE_WAITS=1`, `SOTC_PIPE_VERIFY=1`,
`SOTC_UNMAP_ON_GUEST_THREAD=1`.

## Other

| Variable | What it does |
|---|---|
| `SOTC_ALLOW_UNKNOWN_EBOOT=1` | Runs an `eboot.bin` of a version sotcport was not made for (expect crashes). |
| `SOTC_GAME=<dump>`, `SOTC_BUILD=<build folder>` | For `tools/bench_game.sh` (gameplay benchmark), not for the game. |
