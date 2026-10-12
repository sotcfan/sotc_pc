# Third-party components

sotcport is GPL-2.0-or-later. It builds on the components below, which keep their own licenses;
each license text is in the component's folder.

| Component | Where | License | Use |
|---|---|---|---|
| [shadPS4](https://github.com/shadps4-emu/shadPS4) (video core, shader recompiler, parts of `common`) | `third_party/shadps4/` | GPL-2.0-or-later (`LICENSE`) | GPU command translation to Vulkan; modified, changes marked `sotcport:` |
| [sirit](https://github.com/shadps4-emu/sirit) | `third_party/externals/sirit/` | BSD-3-Clause (`LICENSE.txt`) | SPIR-V emitter |
| [Zydis](https://github.com/zyantific/zydis) | `third_party/externals/zydis/` | MIT (`LICENSE`) | x86 instruction decoder (loader, function hooks) |
| [xbyak](https://github.com/herumi/xbyak) | `third_party/externals/xbyak/` | BSD-3-Clause (`COPYRIGHT`) | x86 code generation |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | `third_party/externals/vma/` | MIT (`LICENSE.txt`) | GPU memory |
| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | `third_party/externals/vulkan-headers/` | Apache-2.0 or MIT (`LICENSE.md`, `LICENSES/`) | Vulkan API headers |
| [magic_enum](https://github.com/Neargye/magic_enum) | `third_party/externals/magic_enum/` | MIT (`LICENSE`) | enum names |
| [robin-map](https://github.com/Tessil/robin-map) | `third_party/externals/robin-map/` | MIT (`LICENSE`) | hash maps |
| [half](https://half.sourceforge.net/) | `third_party/externals/half/` | MIT (`LICENSE.txt`) | half-precision floats |
| [miniz](https://github.com/richgel999/miniz) | `third_party/externals/miniz/` | MIT (`LICENSE`) | zlib-compatible compression |
| [stb_image, stb_image_write](https://github.com/nothings/stb) | `third_party/externals/stb/` | MIT or public domain (end of each header) | screenshots |
| [Tracy](https://github.com/wolfpld/tracy) | `third_party/externals/tracy/` | BSD-3-Clause (`LICENSE`) | profiler hooks (inactive in normal builds) |
| [RenderDoc in-application API](https://renderdoc.org/) | `third_party/externals/renderdoc/` | MIT (in `renderdoc_app.h`) | frame captures for debugging |
| [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9) | `third_party/externals/LibAtrac9/` | MIT (`LICENSE`, origin in `ORIGIN.txt`) | ATRAC9 audio decoder |
| [vgmstream](https://github.com/vgmstream/vgmstream) (HEVAG coefficient table only) | `third_party/externals/vgmstream_hevag/` | ISC-style (`COPYING`, origin in `ORIGIN.txt`) | HEVAG audio decoder |

System libraries linked at build time (not shipped in this repository): SDL3 (zlib), glslang
(BSD-3-Clause and others), fmt (MIT), abseil (Apache-2.0), Boost (BSL-1.0), xxHash
(BSD-2-Clause), zlib (zlib).

The design of this port and several optimizations of the GPU layer follow bbport, the native
Bloodborne port (GPL-2.0-or-later).
