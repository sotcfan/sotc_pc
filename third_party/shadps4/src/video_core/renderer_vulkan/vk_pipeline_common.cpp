// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include <boost/container/static_vector.hpp>

#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;

void Pipeline::BindResources(DescriptorWrites& set_writes,
                             const Shader::PushData& push_data) const {
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;
    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    const vk::PipelineLayout layout = *pipeline_layout;

    // sotcport: recorded through Scheduler::Record (threaded recording); the push data
    // travels by value.
    scheduler.Record([layout, stage_flags, push_data](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
    });

    // Bind descriptor set.
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        if (!scheduler.IsRecordingDeferred()) {
            scheduler.CommandBuffer().pushDescriptorSetKHR(bind_point, layout, 0, set_writes);
            return;
        }
        // The writes point into the rasterizer's descriptor info arrays, reused by the next
        // draw: copy writes and infos into the recording block and point the copies at each
        // other.
        size_t bytes = set_writes.size() * sizeof(vk::WriteDescriptorSet) + 64;
        for (const auto& write : set_writes) {
            if (write.pBufferInfo) {
                bytes += write.descriptorCount * sizeof(vk::DescriptorBufferInfo) + 16;
            } else if (write.pImageInfo) {
                bytes += write.descriptorCount * sizeof(vk::DescriptorImageInfo) + 16;
            } else if (write.pTexelBufferView) {
                bytes += write.descriptorCount * sizeof(vk::BufferView) + 16;
            }
        }
        scheduler.ReserveRecordData(bytes);
        auto* writes = static_cast<vk::WriteDescriptorSet*>(scheduler.RecordAllocate(
            set_writes.size() * sizeof(vk::WriteDescriptorSet), alignof(vk::WriteDescriptorSet)));
        for (size_t i = 0; i < set_writes.size(); i++) {
            vk::WriteDescriptorSet write = set_writes[i];
            const u32 n = write.descriptorCount;
            if (write.pBufferInfo) {
                auto* infos = static_cast<vk::DescriptorBufferInfo*>(scheduler.RecordAllocate(
                    n * sizeof(vk::DescriptorBufferInfo), alignof(vk::DescriptorBufferInfo)));
                std::copy_n(write.pBufferInfo, n, infos);
                write.pBufferInfo = infos;
            } else if (write.pImageInfo) {
                auto* infos = static_cast<vk::DescriptorImageInfo*>(scheduler.RecordAllocate(
                    n * sizeof(vk::DescriptorImageInfo), alignof(vk::DescriptorImageInfo)));
                std::copy_n(write.pImageInfo, n, infos);
                write.pImageInfo = infos;
            } else if (write.pTexelBufferView) {
                auto* views = static_cast<vk::BufferView*>(scheduler.RecordAllocate(
                    n * sizeof(vk::BufferView), alignof(vk::BufferView)));
                std::copy_n(write.pTexelBufferView, n, views);
                write.pTexelBufferView = views;
            }
            new (&writes[i]) vk::WriteDescriptorSet{write};
        }
        const u32 count = static_cast<u32>(set_writes.size());
        scheduler.Record([bind_point, layout, writes, count](vk::CommandBuffer cmdbuf) {
            cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, count, writes);
        });
        return;
    }

    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    scheduler.Record([bind_point, layout, desc_set](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindDescriptorSets(bind_point, layout, 0, desc_set, {});
    });
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->hw_stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
