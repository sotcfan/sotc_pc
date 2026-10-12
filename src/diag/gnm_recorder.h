#pragma once

// The game's renderer recorder (PLANO_RENDERIZADOR_NATIVO.md, etapa 2, "gravador que só
// observa"): watches the ~95 functions of Sony's Gnm/Gnmx code (embedded in the executable)
// through which the game talks to the GPU, without changing what they do.
//
// SOTC_RECORD_GNM=1 turns it on. It then checks the frontier: every dword each of these
// functions writes into a command buffer is marked with the function's number, and every PM4
// packet the GPU executes is looked up in those marks. The report
// (~/.local/share/sotcport/gnm_recorder.txt, every 20 seconds) says how much of what the
// GPU ran was written by a known function, per queue, per function and per packet type, with
// samples of what was not. Nothing uncovered = the list of functions to replace is complete.
//
// It also follows the state the game sets (shaders, textures, render targets) and lists, per
// shader, the variants it ran with: the kinds of textures in the slots it uses and, for pixel
// shaders, the render target formats — what the translation of a shader depends on besides its
// code (gnm_shaders.txt to read, gnm_variants.tsv for tools).
//
// SOTC_RECORD_FRAMES="60,90": also writes the calls of three flip intervals (the engine builds
// more than one frame at a time, so one interval does not hold a whole frame) after each of
// those seconds, one line per call made by the game (arguments decoded:
// shader hashes, textures, buffers, render targets), to
// ~/.local/share/sotcport/gnm_frames/frame_<n>.txt.

namespace diag {

// Called between Linker::LoadExecutable and Linker::Link. Does nothing without the variable.
void InstallGnmRecorder();

} // namespace diag
