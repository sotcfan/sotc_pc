#pragma once

#include "common/types.h"

namespace hle::kernel {

// The PS4 kernel is derived from FreeBSD, so errno values follow FreeBSD, not Linux.
// (Prefixed because the plain names are macros from <cerrno> holding the Linux values.)
constexpr int BSD_EPERM = 1;
constexpr int BSD_ENOENT = 2;
constexpr int BSD_ESRCH = 3;
constexpr int BSD_EINTR = 4;
constexpr int BSD_EIO = 5;
constexpr int BSD_EBADF = 9;
constexpr int BSD_EDEADLK = 11;
constexpr int BSD_ENOMEM = 12;
constexpr int BSD_EACCES = 13;
constexpr int BSD_EFAULT = 14;
constexpr int BSD_EBUSY = 16;
constexpr int BSD_EEXIST = 17;
constexpr int BSD_ENOTDIR = 20;
constexpr int BSD_EISDIR = 21;
constexpr int BSD_EINVAL = 22;
constexpr int BSD_EMFILE = 24;
constexpr int BSD_ENOSPC = 28;
constexpr int BSD_ESPIPE = 29;
constexpr int BSD_EROFS = 30;
constexpr int BSD_ERANGE = 34;
constexpr int BSD_EAGAIN = 35;
constexpr int BSD_ETIMEDOUT = 60;
constexpr int BSD_ENAMETOOLONG = 63;
constexpr int BSD_ENOTEMPTY = 66;
constexpr int BSD_ENOSYS = 78;

// Linux errno -> FreeBSD errno.
int ToBsdErrno(int host_errno);

// Sets the errno the game reads through __error().
void SetErrno(int bsd_errno);

// Sony kernel functions return negative codes 0x8002xxxx, where xxxx is the FreeBSD errno.
constexpr s32 ErrorCode(int bsd_errno) {
    return static_cast<s32>(0x80020000u | static_cast<u32>(bsd_errno));
}

} // namespace hle::kernel
