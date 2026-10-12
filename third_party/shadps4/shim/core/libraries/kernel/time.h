#pragma once

// The libkernel clock functions shadPS4's VideoOut calls; implemented in gpu_glue.cpp.

#include "common/types.h"

namespace Libraries::Kernel {

u64 PS4_SYSV_ABI sceKernelReadTsc();
u64 PS4_SYSV_ABI sceKernelGetProcessTime();

} // namespace Libraries::Kernel
