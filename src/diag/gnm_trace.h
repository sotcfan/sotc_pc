#pragma once

// Diagnostic tool for mapping the game's renderer (PLANO_RENDERIZADOR_NATIVO.md, etapa 0):
// intercepts functions of Sony's Gnm/Gnmx code inside the executable and records, for each one,
// its arguments and the GPU command packets it writes.
//
// SOTC_TRACE_GNM=<file>: the file lists the functions, one per line, as a hexadecimal
// address in the first column (other columns and lines starting with a letter are ignored).
// The report goes to ~/.local/share/sotcport/gnm_trace.txt every 20 seconds.

namespace diag {

// Called between Linker::LoadExecutable and Linker::Link. Does nothing without the variable.
void InstallGnmTracer();

} // namespace diag
