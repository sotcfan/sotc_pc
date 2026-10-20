# Third-party code. Everything under shadps4/ is a frozen copy of parts of the shadPS4
# emulator (GPL-2.0, see shadps4/README.md): its GPU layer, used TEMPORARILY until the game's
# renderer has a native interface of its own.

# sirit: SPIR-V emitter used by the shader recompiler.
set(SIRIT_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(externals/sirit EXCLUDE_FROM_ALL)

# Zydis: x86 disassembler, used by the shader recompiler's fault handler.
set(ZYDIS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_DOXYGEN OFF CACHE BOOL "" FORCE)
add_subdirectory(externals/zydis EXCLUDE_FROM_ALL)

add_library(miniz STATIC
    externals/miniz/miniz.c externals/miniz/miniz_tdef.c
    externals/miniz/miniz_tinfl.c externals/miniz/miniz_zip.c)
target_include_directories(miniz PUBLIC externals/miniz)

# ATRAC9 decoder (the game's sounds), MIT. See externals/LibAtrac9/ORIGEM.txt.
file(GLOB LIBATRAC9_SOURCES externals/LibAtrac9/src/*.c)
add_library(LibAtrac9 STATIC ${LIBATRAC9_SOURCES})
target_include_directories(LibAtrac9 PUBLIC externals/LibAtrac9/src)
set_target_properties(LibAtrac9 PROPERTIES C_STANDARD 99)

find_package(fmt 12 CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_program(GLSLANG_COMPILER glslang REQUIRED)

# Host shaders (GLSL compiled to SPIR-V headers at build time), as in shadPS4.
set(SHADPS4_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shadps4/src)
set(HOST_SHADER_DIR ${SHADPS4_SRC}/video_core/host_shaders)
set(HOST_SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/host_shaders/include/video_core/host_shaders)
set(HOST_SHADER_HEADERS "")
function(add_host_shader FILENAME SHADER_NAME)
    set(HEADER ${HOST_SHADER_OUT}/${SHADER_NAME}.h)
    string(TOUPPER ${SHADER_NAME} SYMBOL)
    set(DEFINES ${ARGN})
    list(TRANSFORM DEFINES PREPEND "-D")
    add_custom_command(
        OUTPUT ${HEADER}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${HOST_SHADER_OUT}
        COMMAND ${GLSLANG_COMPILER} --target-env vulkan1.3 -o ${HEADER} --vn ${SYMBOL}
                ${DEFINES} ${HOST_SHADER_DIR}/${FILENAME}
        COMMAND ${CMAKE_COMMAND} "-DHEADER_FILE=${HEADER}"
                -P ${HOST_SHADER_DIR}/AddShaderHeaderInclude.cmake
        DEPENDS ${HOST_SHADER_DIR}/${FILENAME} ${HOST_SHADER_DIR}/AddShaderHeaderInclude.cmake
        VERBATIM)
    set(HOST_SHADER_HEADERS ${HOST_SHADER_HEADERS} ${HEADER} PARENT_SCOPE)
endfunction()
add_host_shader(color_to_ms_depth.frag color_to_ms_depth_frag)
add_host_shader(ms_image_blit.frag ms_image_blit_frag)
add_host_shader(ms_image_blit.frag ms_image_blit_msaa_frag SRC_MSAA=1)
add_host_shader(fault_buffer_process.comp fault_buffer_process_comp)
add_host_shader(fs_tri.vert fs_tri_vert)
add_host_shader(fsr.comp fsr_easu_comp SAMPLE_EASU=1)
add_host_shader(fsr.comp fsr_rcas_comp SAMPLE_RCAS=1)
add_host_shader(post_process.frag post_process_frag)
foreach(BPP IN ITEMS 8 16 32 64 96 128)
    add_host_shader(tiling.comp tiling_micro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=0)
    add_host_shader(tiling.comp tiling_macro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=1)
endforeach()
add_custom_target(shadps4_host_shaders DEPENDS ${HOST_SHADER_HEADERS})

file(GLOB_RECURSE SHADPS4_SOURCES CONFIGURE_DEPENDS
    ${SHADPS4_SRC}/common/*.cpp
    ${SHADPS4_SRC}/video_core/*.cpp
    ${SHADPS4_SRC}/shader_recompiler/*.cpp
    ${SHADPS4_SRC}/core/libraries/gnmdriver/*.cpp
    ${SHADPS4_SRC}/core/libraries/videoout/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shadps4/shim/*.cpp)
# Parts of shadPS4's common code that belong to the emulator, not to the GPU layer.
list(FILTER SHADPS4_SOURCES EXCLUDE REGEX
    "common/(discord_rpc_handler|key_manager|memory_patcher|ntapi)\\.cpp$")
# Replaced by shim/logging_bridge.cpp.
list(FILTER SHADPS4_SOURCES EXCLUDE REGEX "common/logging/log\\.cpp$")

add_library(shadps4_video STATIC ${SHADPS4_SOURCES})
add_dependencies(shadps4_video shadps4_host_shaders)

# sotcport: the shader cache on disk is keyed to the sources that decide what a cached
# shader holds (shader_cache_version.cmake), so changing them starts a new cache and rebuilding
# anything else keeps it.
file(GLOB_RECURSE SHADER_CACHE_INPUTS CONFIGURE_DEPENDS
    ${SHADPS4_SRC}/shader_recompiler/*.cpp
    ${SHADPS4_SRC}/shader_recompiler/*.h
    ${SHADPS4_SRC}/video_core/renderer_vulkan/vk_pipeline_*
    ${SHADPS4_SRC}/video_core/renderer_vulkan/vk_graphics_pipeline.*
    ${SHADPS4_SRC}/video_core/renderer_vulkan/vk_compute_pipeline.*
    ${CMAKE_CURRENT_SOURCE_DIR}/shadps4/shim/core/emulator_settings.h)
list(SORT SHADER_CACHE_INPUTS)
list(JOIN SHADER_CACHE_INPUTS "\n" SHADER_CACHE_INPUT_LINES)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/shader_cache_inputs.txt "${SHADER_CACHE_INPUT_LINES}\n")
set(SHADER_CACHE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/shader_cache_version/shader_cache_version.h)
add_custom_command(
    OUTPUT ${SHADER_CACHE_HEADER}
    COMMAND ${CMAKE_COMMAND} -DLIST=${CMAKE_CURRENT_BINARY_DIR}/shader_cache_inputs.txt
            -DOUTPUT=${SHADER_CACHE_HEADER}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/shader_cache_version.cmake
    DEPENDS ${SHADER_CACHE_INPUTS} ${CMAKE_CURRENT_SOURCE_DIR}/shader_cache_version.cmake
    VERBATIM)
add_custom_target(shadps4_shader_cache_version DEPENDS ${SHADER_CACHE_HEADER})
add_dependencies(shadps4_video shadps4_shader_cache_version)
# The shim directory comes first: it replaces emulator headers (memory manager, settings,
# window...) with adapters to our own code.
set(SHADPS4_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/shadps4/shim
    ${SHADPS4_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/shadps4
    ${CMAKE_CURRENT_BINARY_DIR}/host_shaders/include
    ${CMAKE_CURRENT_BINARY_DIR}/shader_cache_version
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/vulkan-headers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/vma/include
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/magic_enum/include
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/robin-map/include
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/tracy/public
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/xbyak
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/half/include
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/renderdoc
    # sotcport's own headers used by the shims (hle/, loader/, gpu/), after shadPS4's so
    # that "common/..." means shadPS4's here.
    ${PROJECT_SOURCE_DIR}/src)
# Also used by tools that call into the shader recompiler (tools/shader_survey).
set(SHADPS4_INCLUDE_DIRS ${SHADPS4_INCLUDE_DIRS} PARENT_SCOPE)
target_include_directories(shadps4_video PRIVATE ${SHADPS4_INCLUDE_DIRS})
target_compile_definitions(shadps4_video PRIVATE BOOST_ASIO_STANDALONE)
target_compile_options(shadps4_video PRIVATE -w -fno-strict-aliasing -fno-omit-frame-pointer)
target_link_libraries(shadps4_video PUBLIC
    sirit Zydis miniz fmt::fmt absl::flat_hash_map absl::flat_hash_set
    Boost::headers SDL3::SDL3 xxhash ${CMAKE_DL_LIBS})

# sotcport: FSR 3.1 temporal upscaling (FireBurn/FSR-Vulkan, MIT; only the upscaler, see
# externals/fsr-vulkan/ORIGIN.txt), used by renderer_vulkan/vk_fsr.cpp. Its CMake looks Vulkan up
# with FindVulkan: the headers are the project's own (the loader library is the system's).
set(FFX_VK_PORTABLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_PROBE OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_FSR3_3_1_5_SCAFFOLD OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_FSR4_V07_VULKAN OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(Vulkan_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals/vulkan-headers/include
    CACHE PATH "Vulkan headers (the project's own)" FORCE)
add_subdirectory(externals/fsr-vulkan EXCLUDE_FROM_ALL)
target_link_libraries(shadps4_video PUBLIC ffx-vulkan::portable)
