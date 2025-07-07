//===--------------- queue_batched.hpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#pragma once

#include "../common.hpp"
#include "../device.hpp"

#include "common/ur_ref_count.hpp"
#include "context.hpp"
#include "event.hpp"
#include "event_pool_cache.hpp"
#include "memory.hpp"
#include "queue_api.hpp"

#include "ur/ur.hpp"

#include "command_list_manager.hpp"
#include "lockable.hpp"
#include "command_buffer.hpp"
#include "queue_immediate_in_order.hpp"
#include "ur_api.h"

namespace v2 {

struct ur_queue_batched_t : ur_object, ur_queue_t_ {
private:
    ur_context_handle_t hContext;
    ur_device_handle_t hDevice;
    lockable<ur_command_list_manager> commandListManager;
    ur_queue_flags_t flags;
    v2::raii::cache_borrowed_event_pool eventPool;
    ur_exp_command_buffer_handle_t commandBuffer;

public:
    ur_queue_batched_t(ur_context_handle_t, ur_device_handle_t,
                        uint32_t ordinal,
                        ze_command_queue_priority_t priority,
                        std::optional<int32_t> index,
                        event_flags_t eventFlags,
                        ur_queue_flags_t flags, 
                        ur_exp_command_buffer_handle_t cmdBuffer);

};

} // namespace v2