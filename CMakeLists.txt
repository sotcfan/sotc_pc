cmake_minimum_required(VERSION 3.28)
project(sotcport LANGUAGES C CXX ASM)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(FATAL_ERROR "The game code is x86-64: build on an x86-64 PC.")
endif()
if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux|Windows")
    message(FATAL_ERROR "Linux and Windows are supported.")
endif()
if(WIN32 AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # The game's code uses the System V calling convention; our functions it calls are marked
    # __attribute__((sysv_abi)), which needs clang (MSYS2 CLANG64: see the README).
    message(FATAL_ERROR "Windows builds need clang (MSYS2 CLANG64)")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ZLIB REQUIRED)

# Whole-program optimization (as bbport does). Set before add_subdirectory(third_party) so the GPU layer and the libraries
# built here get it too. Clang only (the project is built with clang since milestone 2).
#   SOTC_LTO=ON          ThinLTO, linked with lld.
#   SOTC_PGO=generate    instrumented build: writes <SOTC_PGO_DIR>/sotcport-<pid>.profraw every
#                          30 s while the game runs (src/diag/pgo_dump.cpp); play a few minutes,
#                          then: llvm-profdata merge -o <dir>/sotcport.profdata <dir>/*.profraw
#   SOTC_PGO=use         optimizes with <SOTC_PGO_DIR>/sotcport.profdata (functions changed
#                          since the profile was taken just compile without it).
# Measured on 2026-10-11 (tools/bench_game.sh, 3 alternating runs each): GPU command thread cost
# per draw+dispatch 3.97 -> 3.94 us with ThinLTO alone, 4.06 -> 3.64 us (-10%) with LTO + PGO.
# build-clang uses both; retrain the profile after larger changes to the GPU layer.
option(SOTC_LTO "Link-time optimization (ThinLTO)" OFF)
set(SOTC_PGO "off" CACHE STRING "Profile-guided optimization: off, generate or use")
set(SOTC_PGO_DIR "$ENV{HOME}/.local/share/sotcport/pgo" CACHE PATH "Profile directory")
if(SOTC_LTO OR NOT SOTC_PGO STREQUAL "off")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "SOTC_LTO/SOTC_PGO need clang")
    endif()
endif()
if(SOTC_LTO)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-flto=thin>)
    add_link_options(-flto=thin -fuse-ld=lld)
endif()
if(SOTC_PGO STREQUAL "generate")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fprofile-instr-generate>)
    add_link_options(-fprofile-instr-generate)
    add_compile_definitions(SOTC_PGO_GENERATE SOTC_PGO_DIR="${SOTC_PGO_DIR}")
elseif(SOTC_PGO STREQUAL "use")
    set(SOTC_PGO_PROFILE "${SOTC_PGO_DIR}/sotcport.profdata")
    if(NOT EXISTS "${SOTC_PGO_PROFILE}")
        message(WARNING "SOTC_PGO=use: ${SOTC_PGO_PROFILE} not found, building without PGO")
    else()
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fprofile-instr-use=${SOTC_PGO_PROFILE}>
                        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-profile-instr-out-of-date>
                        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-profile-instr-unprofiled>
                        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-backend-plugin>)
    endif()
elseif(NOT SOTC_PGO STREQUAL "off")
    message(FATAL_ERROR "SOTC_PGO must be off, generate or use")
endif()

if(WIN32)
    # For our code and the GPU layer alike: <windows.h> without min/max macros and rarely used
    # parts; Windows 10 APIs (placeholders: VirtualAlloc2, MapViewOfFile3).
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)
endif()
add_subdirectory(third_party)

# Everything but main() goes into an object library shared by the game and the tools.
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.S)
list(REMOVE_ITEM CORE_SOURCES ${PROJECT_SOURCE_DIR}/src/main.cpp)
add_library(sotc_core OBJECT ${CORE_SOURCES})
target_include_directories(sotc_core PUBLIC src third_party/externals/vgmstream_hevag
    third_party/externals/xbyak)
# Vulkan headers for the native renderer's own device (src/render/vulkan).
target_include_directories(sotc_core SYSTEM PUBLIC third_party/externals/vulkan-headers/include)
target_compile_options(sotc_core PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -fno-strict-aliasing -fno-omit-frame-pointer>)
target_link_libraries(sotc_core PUBLIC pthread ZLIB::ZLIB shadps4_video LibAtrac9)
if(WIN32)
    # onecore: VirtualAlloc2/MapViewOfFile3/UnmapViewOfFile2; the rest for SDL, sockets, etc.
    target_link_libraries(sotc_core PUBLIC onecore bcrypt ws2_32)
endif()

add_executable(sotcport src/main.cpp)
target_link_libraries(sotcport PRIVATE sotc_core)
if(WIN32)
    # The game's address range (8 GiB - 1 TiB, and the mirror at 2 TiB) must stay free until the
    # memory manager reserves it: no high-entropy ASLR for heaps, stacks and DLLs. The main thread
    # also gets a 16 MiB stack (as the Windows fork of bbport does).
    target_link_options(sotcport PRIVATE -Wl,--disable-high-entropy-va
        "SHELL:-Xlinker --stack -Xlinker 16777216")
endif()
