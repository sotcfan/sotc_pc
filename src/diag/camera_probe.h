#pragma once

// TEMP (2026-10-16): finding the game's camera code (see camera_probe.cpp).

namespace diag {

// SOTC_PROBE_CAMERA. Between Linker::LoadExecutable and Linker::Link.
void InstallCameraProbe();

} // namespace diag
