#pragma once

// Diagnostic tool for the native renderer's shader work (PLANO_RENDERIZADOR_NATIVO.md, etapa 1):
// saves resources as the game decompresses them. Most shaders (.csdr) are Oodle-compressed
// inside the PSARC archives and the uncompressed size lives in the engine's own resource
// records, so the simplest way to get them is to let the game decompress them.
//
// SOTC_DUMP_RESOURCES=<extensions>: comma-separated list (e.g. ".csdr,.cslt"). Each matching
// resource is written, decompressed, to ~/.local/share/sotcport/resources/<game path>
// the first time the game loads it.

namespace diag {

// Called between Linker::LoadExecutable and Linker::Link. Does nothing without the variable.
void InstallResourceDump();

} // namespace diag
