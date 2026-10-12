#include "hle/kernel/errno.h"

#include <cerrno>

#include "hle/registry.h"

namespace hle::kernel {

namespace {

thread_local int t_errno = 0;

// Returns the address of the calling thread's errno.
PS4_SYSV_ABI int* error() {
    return &t_errno;
}

} // namespace

int ToBsdErrno(int host_errno) {
    switch (host_errno) {
    case 0: return 0;
    case EPERM: return BSD_EPERM;
    case ENOENT: return BSD_ENOENT;
    case ESRCH: return BSD_ESRCH;
    case EINTR: return BSD_EINTR;
    case EIO: return BSD_EIO;
    case EBADF: return BSD_EBADF;
    case EDEADLK: return BSD_EDEADLK;
    case ENOMEM: return BSD_ENOMEM;
    case EACCES: return BSD_EACCES;
    case EFAULT: return BSD_EFAULT;
    case EBUSY: return BSD_EBUSY;
    case EEXIST: return BSD_EEXIST;
    case ENOTDIR: return BSD_ENOTDIR;
    case EISDIR: return BSD_EISDIR;
    case EINVAL: return BSD_EINVAL;
    case EMFILE: return BSD_EMFILE;
    case ENOSPC: return BSD_ENOSPC;
    case ESPIPE: return BSD_ESPIPE;
    case EROFS: return BSD_EROFS;
    case ERANGE: return BSD_ERANGE;
    case EAGAIN: return BSD_EAGAIN;
    case ETIMEDOUT: return BSD_ETIMEDOUT;
    case ENAMETOOLONG: return BSD_ENAMETOOLONG;
    case ENOTEMPTY: return BSD_ENOTEMPTY;
    case ENOSYS: return BSD_ENOSYS;
    default: return BSD_EIO;
    }
}

void SetErrno(int bsd_errno) {
    t_errno = bsd_errno;
}

HLE_FUNCTION("libkernel", "__error", error);

} // namespace hle::kernel
