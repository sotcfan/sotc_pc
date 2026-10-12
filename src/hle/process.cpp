#include "hle/process.h"

#include "common/log.h"
#include "hle/kernel/kernel.h"
#include "hle/libc/libc.h"

namespace hle {

void InitializeProcess(const loader::ProcParam* param) {
    kernel::InitializeMainThread();
    if (param) {
        LOG_INFO("Process", "SDK {:#x}", param->sdk_version);
    }
    libc::Initialize(param && param->size > offsetof(loader::ProcParam, libc_param)
                         ? param->libc_param
                         : nullptr);
}

} // namespace hle
