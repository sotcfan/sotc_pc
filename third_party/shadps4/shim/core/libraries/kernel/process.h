#pragma once

// The few libkernel functions shadPS4's GPU code calls directly; implemented in gpu_glue.cpp.

#include "common/types.h"

namespace Libraries::Kernel {

s32 PS4_SYSV_ABI sceKernelIsNeoMode();
u64 PS4_SYSV_ABI sceKernelGetTscFrequency();
u64 PS4_SYSV_ABI sceKernelReadTsc();
s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* version);

} // namespace Libraries::Kernel
