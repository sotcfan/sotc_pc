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

## Building and running on Windows

The game calls sotcport's functions with the System V calling convention, which only clang
supports on Windows, so the build uses MSYS2's **CLANG64** environment.

1. Install MSYS2 and open the **MSYS2 CLANG64** terminal.
2. Install the tools and libraries:

   ```sh
   pacman -S --needed git \
     mingw-w64-clang-x86_64-{toolchain,cmake,ninja,pkgconf,python} \
     mingw-w64-clang-x86_64-{vulkan-headers,vulkan-loader,glslang} \
     mingw-w64-clang-x86_64-{sdl3,fmt,boost,abseil-cpp,xxhash,zlib}
   ```

3. Build (from the sotcport folder):

   ```sh
   cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
   cmake --build build
   ```

   `-DSOTC_LTO=ON` works as on Linux; PGO is Linux-only.

4. Run it from the same CLANG64 terminal (it has the DLLs on its PATH):

   ```powershell
   .\build\sotcport.exe "C:\path\to\your\dump\"
   ```

   To run it outside MSYS2, copy the DLLs it needs next to `sotcport.exe`:
   `ldd build/sotcport.exe | grep -i clang64 | awk '{print $3}' | xargs -I{} cp {} build/`

Saves, settings and the shader cache go to `%LOCALAPPDATA%\sotcport` (`--data <folder>` changes
it). Set environment variables with `export NAME=value` in the CLANG64 terminal (or
`set NAME=value` in `cmd`). Not available on Windows: `kill -USR1` thread dumps (use
`--dump-after`), F11 sampling and PGO.

If something goes wrong, keep the log for a bug report:

```powershell
.\build\sotcport.exe "C:\path\to\your\dump" 2>&1 | tee sotc_log.txt
```

## Options and controls (both systems)

- `--log debug`: more detail in the log.
- `--data <folder>`: where saves and settings go.
- `--trace <file>`: records, in order, the first call to each Sony function.

Controls: a gamepad (SDL3, buttons by position) or the keyboard: WASD left stick, arrow keys
right stick, Space/Enter ✕, Backspace/F ○, E △, Q □, Shift R1, Ctrl L1, C R2, Z L2, B R3,
V L3, I J K L directional buttons, Esc Options, Tab touchpad. F9 writes a marker into the log
and saves a screenshot.

Environment variables (frame limiter, performance switches, diagnostics): see
[OPTIONS.md](OPTIONS.md). For play, `SOTC_FPS_LIMIT=30 SOTC_DRAW_PIPE=1` is a good start: a
steady 30 FPS. `SOTC_FPS_LIMIT=60` is still experimental: the GPU layer does not always keep up, and
some of the game's logic breaks at 60 (see OPTIONS.md, "Frame rate").

## Layout

| Folder | Contents |
|---|---|
| `src/loader/` | SELF loading, memory mapping, relocations, import linking, crash reports |
| `src/hle/` | our implementations of the Sony libraries, one folder per library |
| `src/input/` | keyboard and gamepads as a PS4 controller |
| `src/gpu/` | the interface to the GPU layer |
| `src/diag/` | diagnostics (freeze watch, packet statistics, profiling helpers) |
| `third_party/shadps4/` | shadPS4's video core and shader recompiler, modified (changes marked `sotcport:`) |
| `third_party/externals/` | libraries built with the project (see `THIRD_PARTY.md`) |
| `tools/` | benchmark, PGO training, import listing |

## Legal

sotcport contains no code, data or assets from the game and does not circumvent any
protection: it needs a dump made by the user from their own console. Shadow of the Colossus
is a trademark of Sony Interactive Entertainment. This project is not affiliated with or
endorsed by Sony Interactive Entertainment, Bluepoint Games or the shadPS4 project.

## License

GPL-2.0-or-later (see `LICENSE`), like the shadPS4 code it builds on. Third-party components
keep their own licenses, listed in `THIRD_PARTY.md`.

## Credits

- [shadPS4](https://github.com/shadps4-emu/shadPS4): the video core and shader recompiler.
- bbport (the native Bloodborne port): the design this project shares and several of its GPU
  layer optimizations; its Windows fork by Supermedo: the Windows memory and TLS approach.
- [vgmstream](https://github.com/vgmstream/vgmstream): the HEVAG coefficient table.
- [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9): the ATRAC9 decoder.
