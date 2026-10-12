#pragma once

// PM4 packet statistics (groundwork for a two-stage draw pipeline, as in bbport): SOTC_PACKET_STATS=1 counts
// the packets the GPU command processor runs, per queue and opcode, and estimates how often a
// two-stage draw pipeline would have to drain (wait for the second stage to run dry). Written to
// the log every 5 s.

namespace diag {

// Called before the game starts. Does nothing without the variable.
void InstallPacketStats();

} // namespace diag
