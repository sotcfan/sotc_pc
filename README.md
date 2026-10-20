# sotcport

A native PC port of **Shadow of the Colossus** (the 2018 PS4 remake, CUSA08809) in the
"bring your own game" model: no emulator, and **no game files included**. You run your own,
legally obtained copy of the game, dumped from your own console.

The PS4 game code is already x86-64. sotcport loads the original `eboot.bin` into its own
process at the console's address and replaces every Sony library function the game calls
(files, memory, threads, audio, input, saves...) with its own implementation written for the
PC. GPU commands are translated to Vulkan by a modified copy of
[shadPS4](https://github.com/shadps4-emu/shadPS4)'s video core.

> **Status:** in development. The game is playable from the title screen through gameplay at
> 30 FPS on the test machine; expect bugs. Linux is the tested system; Windows support is new
> and experimental.

## Requirements

On every system:

- A CPU with SSE4a (every AMD Ryzen has it). The game uses SSE4a instructions that Intel CPUs
  lack; Intel support is planned.
- A Vulkan 1.3 GPU (tested on an AMD Radeon RX 7900 XT). NVIDIA cards are untested: some
  effects that depend on 64-wide shader waves (character lighting, water) may look wrong there.
- **Your own decrypted dump of the game** (the folder with `eboot.bin`, `sce_module/` and the
  `pak*.psarc` files). sotcport never writes to that folder.

Linux: x86-64, Clang 18+, CMake 3.28+, Ninja; system packages: SDL3, glslang, fmt 12, abseil,
Boost, xxhash, zlib.

Windows (experimental, not yet tested on real hardware): Windows 10 1903 or later / Windows 11,
64-bit, [MSYS2](https://www.msys2.org) (its CLANG64 environment), and a page file of a few GB (the
game's memory is committed from it).

### Supported game version

The port hooks functions inside the game's executable, so it only runs the version it was made
for and refuses others at startup (`SOTC_ALLOW_UNKNOWN_EBOOT=1` tries anyway; expect crashes).
Check yours with `sha256sum eboot.bin`:

| Version | `eboot.bin` SHA-256 |
|---|---|
| CUSA08809 (Europe), 01.00 | `21982727c6f97fd415b3af5e99c5e071619c4dcc2248787a701a31a45ab0a4e0` |

## Building and running on Linux

```sh
CC=clang CXX=clang++ cmake -B build -G Ninja
cmake --build build
./build/sotcport /path/to/your/dump
```

Optional: `-DSOTC_LTO=ON` (link-time optimization) and `-DSOTC_PGO=generate|use`
(profile-guided optimization; `tools/pgo_train.sh` trains a profile). Together they make the
GPU command thread about 10% faster.

Saves and settings go to `~/.local/share/sotcport` (`--data <folder>` changes it).

### Launcher

`launcher/sotcport-launcher.sh` opens a window that picks your dump, sets the options and starts
the game, with its log. It needs PySide6 (Arch/CachyOS: `pacman -S pyside6`; Debian/Ubuntu:
`apt install python3-pyside6.qtwidgets`; elsewhere `pip install PySide6`);
`launcher/install-desktop.sh` adds it to the application menu.

- **Display:** fullscreen (borderless, Alt+Enter toggles it), monitor, window size, VSync (present
  mode), PS4 Pro mode, the frame rate limit and the game's resolution: by default the monitor's
  own (the game draws and outputs at it, no upscaling; on a PS4 Pro it drew at 1600x900 or
  2560x1440 and enlarged that to 1080p or 4K). Upscaling: off by default, or FSR 3.1 (Quality, Balanced,
  Performance, Ultra Performance, or Native as anti-aliasing only) with its sharpening.
- **Graphics:** presets (Console, High, Maximum) and the game engine's own settings (sun shadows
  and their detail, ambient occlusion, anti-aliasing, motion blur, depth of field, bloom, film
  grain, vignette, distant detail of vegetation and props, particles, anisotropic filtering),
  given to the engine as console variables on its command line, plus the texture size in VRAM.
  "Restore defaults" sets everything back (folders and languages stay).
- **Game language:** menus and subtitles, by default the system's (Brazilian Portuguese, US
  English, Canadian French and Latin American Spanish too, which the European dump hides).
- **Performance, controls (`controls.ini`), advanced** (extra variables and engine arguments).

Its settings are kept in `~/.config/sotcport-launcher/settings.json`.

## Building and running on Windows

The game calls sotcport's functions with the System V calling convention, which only clang
supports on Windows, so the build uses the clang toolchain from [MSYS2](https://www.msys2.org)
(its CLANG64 packages). Every command below is for **PowerShell**.

1. Install MSYS2 in its default folder, `C:\msys64`.
2. Install the compiler and the libraries:

   ```powershell
   C:\msys64\usr\bin\pacman.exe -Syu --noconfirm
   C:\msys64\usr\bin\pacman.exe -S --needed --noconfirm `
     mingw-w64-clang-x86_64-toolchain mingw-w64-clang-x86_64-cmake `
     mingw-w64-clang-x86_64-ninja mingw-w64-clang-x86_64-pkgconf mingw-w64-clang-x86_64-python `
     mingw-w64-clang-x86_64-vulkan-headers mingw-w64-clang-x86_64-vulkan-loader `
     mingw-w64-clang-x86_64-glslang mingw-w64-clang-x86_64-sdl3 mingw-w64-clang-x86_64-fmt `
     mingw-w64-clang-x86_64-boost mingw-w64-clang-x86_64-abseil-cpp `
     mingw-w64-clang-x86_64-xxhash mingw-w64-clang-x86_64-zlib
   ```

   (If the first command says it updated core packages and must close, run it again.)
3. In every PowerShell window used to build or run sotcport, put those tools first on the PATH:

   ```powershell
   $env:PATH = "C:\msys64\clang64\bin;$env:PATH"
   ```

4. Build, from the sotcport folder:

   ```powershell
   cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
   cmake --build build
   ```

   `-DSOTC_LTO=ON` works as on Linux; PGO is Linux-only.

5. Run (with the PATH of step 3, which also has the DLLs sotcport needs):

   ```powershell
   $env:SOTC_FPS_LIMIT = "30"; $env:SOTC_DRAW_PIPE = "1"
   .\build\sotcport.exe "C:\path\to\your\dump" *> sotc_log.txt
   ```

   The log (`sotc_log.txt`) is what to attach to a bug report; the console stays empty while it
   is written. Do not end the dump path with a backslash: `"C:\dump\"` reaches the program as
   `C:\dump"`.

Saves, settings and the shader cache go to `%LOCALAPPDATA%\sotcport` (`--data <folder>` changes
it). Not available on Windows: thread dumps on request (use `--dump-after`), F11 sampling and PGO.

## Options and controls (both systems)

- `--log debug`: more detail in the log.
- `--data <folder>`: where saves and settings go.
- `--trace <file>`: records, in order, the first call to each Sony function.

Controls: a gamepad (SDL3, buttons by position) or keyboard and mouse. Click in the window to
capture the mouse: it turns the game's free camera directly, as far as it moves (elsewhere it
pushes the right stick); F1 releases it. In menus the mouse is a cursor: pointing at an item
selects it, the left button confirms (or steps a value on its arrows), the right button and Esc go
back and the wheel moves the selection; on the world map dragging moves the map and the wheel
zooms.
Keyboard and mouse bindings are in `controls.ini` in the data folder, written with the defaults on
the first run (WASD move, Left Alt walk, mouse/arrows camera, Space ✕, left button □, right
button R2...); delete it to get the defaults back. F9 writes a marker into the log and saves a
screenshot.
Trophies have no PlayStation service behind them: each one is recorded the first time the game
unlocks it, in `trophies.txt` in the data folder (trophy id, date and time; 38 trophies, 0 is the
platinum, given with the last of the others). Their names are encrypted in the game's files; the
icons in `sce_sys/trophy/trophy00.trp` (`TROP000.PNG`...) match the ids.

Environment variables (display, frame limiter, performance switches, diagnostics) and the
engine's console variables: see [OPTIONS.md](OPTIONS.md); the launcher sets them for you. Without
it, `SOTC_NEO_MODE=1 SOTC_FPS_LIMIT=60 SOTC_DRAW_PIPE=1 SOTC_ASYNC_READBACK=1 SOTC_ASYNC_SHADERS=1`
is the launcher's default: the PS4 Pro mode at 60 FPS. Without `SOTC_NEO_MODE=1` keep
`SOTC_FPS_LIMIT=30`: some of the game's logic breaks above 30 on an original PS4 (see OPTIONS.md,
"Frame rate").

## Layout

| Folder | Contents |
|---|---|
| `src/loader/` | SELF loading, memory mapping, relocations, import linking, crash reports |
| `src/hle/` | our implementations of the Sony libraries, one folder per library |
| `src/input/` | keyboard and gamepads as a PS4 controller |
| `src/gpu/` | the interface to the GPU layer |
| `src/diag/` | diagnostics (freeze watch, packet statistics, profiling helpers) |
| `third_party/shadps4/` | shadPS4's video core and shader recompiler, modified (changes marked `sotcport:`) |
| `third_party/externals/` | libraries built with the project (see [THIRD_PARTY.md](THIRD_PARTY.md)) |
| `tools/` | benchmark, PGO training, import listing |
| `launcher/` | the launcher (Python, Qt/PySide6) |

## Legal

sotcport contains no code, data or assets from the game and does not circumvent any
protection: it needs a dump made by the user from their own console. Shadow of the Colossus
is a trademark of Sony Interactive Entertainment. This project is not affiliated with or
endorsed by Sony Interactive Entertainment, Bluepoint Games or the shadPS4 project.

## License

GPL-2.0-or-later (see [LICENSE](LICENCE.md), like the shadPS4 code it builds on. Third-party components
keep their own licenses, listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [shadPS4](https://github.com/shadps4-emu/shadPS4): the video core and shader recompiler.
- bbport (the native Bloodborne port): the design this project shares, several of its GPU
  layer optimizations and the layout of its launcher; its Windows fork by Supermedo: the Windows
  memory and TLS approach.
- [vgmstream](https://github.com/vgmstream/vgmstream): the HEVAG coefficient table.
- [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9): the ATRAC9 decoder.
