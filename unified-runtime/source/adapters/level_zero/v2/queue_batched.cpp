//===--------------- queue_batched.cpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue_batched.hpp"
#include "command_buffer.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "ur.hpp"

#include "../common/latency_tracker.hpp"
#include "../helpers/kernel_helpers.hpp"
#include "../image_common.hpp"

#include "../program.hpp"
#include "../ur_interface_loader.hpp"
#include <cstddef>

namespace v2 {

ur_queue_batched_t::ur_queue_batched_t(ur_context_handle_t hContext,
    ur_device_handle_t hDevice,
                uint32_t ordinal,
                ze_command_queue_priority_t priority,
                std::optional<int32_t> index,
                event_flags_t eventFlags,
                ur_queue_flags_t flags, 
                ur_exp_command_buffer_handle_t cmdBuffer) 
                : hContext(hContext), hDevice(hDevice),
      commandListManager(
          hContext, hDevice,
          hContext->getCommandListCache().getImmediateCommandList(
              hDevice->ZeDevice,
              {true, ordinal, true /* always enable copy offload */},
              ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, priority, index)),
      flags(flags),
      eventPool(hContext->getEventPoolCache(PoolCacheType::Immediate)
                    .borrow(hDevice->Id.value(), eventFlags)),
                    commandBuffer(std::move(cmdBuffer))
                {}


} // namespace v2