// SPDX-License-Identifier: GPL-2.0-or-later
// sotcport: background compilation of graphics pipelines (SOTC_ASYNC_SHADERS=1).
//
// The first time a graphics pipeline is needed, creating it in the driver takes ~5 ms (shader
// compilation by the driver; measured without the driver's and our on-disk caches, which is what
// a first playthrough sees). Done on the GPU command thread, a few of those in a frame are a
// visible hitch. With this option the pipeline is created on a worker thread instead and the
// draws that need it are skipped until it is ready (objects may appear a few frames late the
// first time). Compute pipelines are still created at once: skipping a dispatch could leave
// data the game reads later unwritten.
//
// This file is not part of the shader cache version (it decides when pipelines are built, not
// what the cache holds): see shader_cache_version.cmake.

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "common/unique_function.h"

namespace Vulkan {

class AsyncPipelineCompiler {
public:
    /// True when SOTC_ASYNC_SHADERS=1.
    static bool Enabled();

    explicit AsyncPipelineCompiler(unsigned num_threads);
    ~AsyncPipelineCompiler();

    /// Runs `job` on a worker thread.
    void Submit(Common::UniqueFunction<void> job);

private:
    void Run(std::stop_token stop);

    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<Common::UniqueFunction<void>> jobs;
    std::vector<std::jthread> threads;
};

} // namespace Vulkan
