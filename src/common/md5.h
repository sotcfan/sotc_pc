#pragma once

#include <array>
#include <string_view>

#include "common/types.h"

// MD5 (RFC 1321). Only used to find files in PSARC archives, never for security.
std::array<u8, 16> Md5(std::string_view data);
