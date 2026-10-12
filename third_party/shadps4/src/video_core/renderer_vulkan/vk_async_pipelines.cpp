// SPDX-License-Identifier: GPL-2.0-or-later
// sotcport: see vk_async_pipelines.h.

#include <cstdlib>

#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_async_pipelines.h"

namespace Vulkan {

bool AsyncPipelineCompiler::Enabled() {
    static const bool enabled = std::getenv("SOTC_ASYNC_SHADERS") != nullptr;
    return enabled;
}

AsyncPipelineCompiler::AsyncPipelineCompiler(unsigned num_threads) {
    for (unsigned i = 0; i < num_threads; i++) {
        threads.emplace_back([this](std::stop_token stop) { Run(stop); });
    }
}

AsyncPipelineCompiler::~AsyncPipelineCompiler() {
    for (auto& thread : threads) {
        thread.request_stop();
    }
    cv.notify_all();
    threads.clear(); // joins
}

void AsyncPipelineCompiler::Submit(Common::UniqueFunction<void> job) {
    {
        std::scoped_lock lock{mutex};
        jobs.push_back(std::move(job));
    }
    cv.notify_one();
}

void AsyncPipelineCompiler::Run(std::stop_token stop) {
    Common::SetCurrentThreadName("sotc:ShaderBuild");
    while (true) {
        Common::UniqueFunction<void> job;
        {
            std::unique_lock lock{mutex};
            if (!cv.wait(lock, stop, [this] { return !jobs.empty(); })) {
                return; // stop requested
            }
            job = std::move(jobs.front());
            jobs.pop_front();
        }
        job();
    }
}

} // namespace Vulkan
