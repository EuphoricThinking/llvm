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
#include "ur_api.h"
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
                    // commandBuffer(std::move(cmdBuffer))
                    commandBuffer(cmdBuffer)
                {}

ur_result_t ur_queue_batched_t::enqueueKernelLaunch(
      ur_kernel_handle_t hKernel, uint32_t workDim,
      const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
      const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
      const ur_kernel_launch_property_t *launchPropList,
      uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
      ur_event_handle_t *phEvent) {
      //   return ur::level_zero::urCommandBufferAppendKernelLaunchExp(commandBuffer, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, 0 /* numKernelAlternatives */, nullptr /* phKernelAlternatives */, 0 /* numSyncPointsInWaitList */, nullptr /* syncPointWaitList */, 0 /*numEventsInWaitList*/, nullptr /* *eventWaitList */, nullptr /* retSyncPoint */, nullptr /* event */, nullptr /* command - not updatable buffer */);
      // }
      //  return commandListManager.lock()->appendKernelLaunch(
      //   hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
      //   numPropsInLaunchPropList, launchPropList, numEventsInWaitList,
      //   phEventWaitList,
      //   createEventIfRequested(eventPool.get(), phEvent, this));

      auto commandListLocked = commandBuffer->commandListManager.lock();

      UR_CALL(commandListLocked->appendKernelLaunch(
      hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, numPropsInLaunchPropList,
      launchPropList, numEventsInWaitList, phEventWaitList,
      createEventIfRequested(eventPool.get(), phEvent, this)));



      return UR_RESULT_SUCCESS;
      }


ur_result_t ur_queue_batched_t::queueFinish() {
  try {
    // urCommandBufferFinalizeExp(
    //     commandBuffer);
    // finalize before enqueueing the command buffer
    UR_CALL(commandBuffer->finalizeCommandBuffer());

    // enqueue command buffer
    auto lockedCommandListManager = commandListManager.lock();
    lockedCommandListManager->appendCommandBufferExp(
    commandBuffer, 0, nullptr,
    createEventAndRetain(eventPool.get(), nullptr, this));

    // finish queue
    ZE2UR_CALL(zeCommandListHostSynchronize,
             (lockedCommandListManager->getZeCommandList(), UINT64_MAX));

  hContext->getAsyncPool()->cleanupPoolsForQueue(this);
  hContext->forEachUsmPool([this](ur_usm_pool_handle_t hPool) {
    hPool->cleanupPoolsForQueue(this);
    return true;
  });

  UR_CALL(lockedCommandListManager->releaseSubmittedKernels());

  // release cmdbuff
  // urCommandBufferReleaseExp(commandBuffer);
    if (commandBuffer->RefCount.release()) {
      if (auto executionEvent = commandBuffer->getExecutionEventUnlocked()) {
      ZE2UR_CALL_THROWS(zeEventHostSynchronize,
               (executionEvent->getZeEvent(), UINT64_MAX));
      }
    delete commandBuffer;
    }

    // create cmdbuff
    ur_exp_command_buffer_handle_t cmdBuffer = nullptr;
    
    ur_exp_command_buffer_desc_t cmdBufferDesc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
        nullptr,     // pNext
        false,       // isUpdatable
        true, // isInOrder
        // (flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0,     // isInOrder
        (flags & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0 // enableProfiling
    };

     ur::level_zero::urCommandBufferCreateExp(
        hContext, hDevice, &cmdBufferDesc, &cmdBuffer);

        commandBuffer = std::move(cmdBuffer);



  return UR_RESULT_SUCCESS;
}
catch (...) {
  return exceptionToResult(std::current_exception());
}
}

ur_queue_batched_t::~ur_queue_batched_t() {
try {
    // // urCommandBufferReleaseExp(commandBuffer);
    // if (commandBuffer->RefCount.release()) {
    //   if (auto executionEvent = commandBuffer->getExecutionEventUnlocked()) {
    //   ZE2UR_CALL_THROWS(zeEventHostSynchronize,
    //            (executionEvent->getZeEvent(), UINT64_MAX));
    //   }
    // delete commandBuffer;
    // }

    UR_CALL_THROWS(queueFinish());
    delete commandBuffer;
  } catch (...) {
    // Ignore errors during destruction
  }
}

ur_result_t
ur_queue_batched_t::enqueueMemBufferRead(ur_mem_handle_t hBuffer, bool blockingRead,
                                   size_t offset, size_t size, void *pDst,
                                   uint32_t numEventsInWaitList,
                                   const ur_event_handle_t *phEventWaitList,
                                   ur_event_handle_t *phEvent) {
  try {
auto commandListLocked = commandBuffer->commandListManager.lock();
  // auto eventsWaitList = commandBuffer->getWaitListFromSyncPoints(
  //     pSyncPointWaitList, numSyncPointsInWaitList);
  UR_CALL(commandListLocked->appendMemBufferRead(
      hBuffer, false, offset, size, pDst, numEventsInWaitList,
      phEventWaitList, createEventIfRequested(eventPool.get(), phEvent, this)));

  //   if (blockingRead) {
  //     this->queueFinish();
  // }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}
}

ur_result_t
ur_queue_batched_t::enqueueMemBufferWrite(ur_mem_handle_t hBuffer, bool blockingWrite,
                                    size_t offset, size_t size,
                                    const void *pSrc,
                                    uint32_t numEventsInWaitList,
                                    const ur_event_handle_t *phEventWaitList,
                                    ur_event_handle_t *phEvent) 
                                      try {

  // the same issue as in urCommandBufferAppendKernelLaunchExp
  // sync mechanic can be ignored, because all lists are in-order
  // Responsibility of UMD to offload to copy engine
  auto commandListLocked = commandBuffer->commandListManager.lock();
  // auto eventsWaitList = hCommandBuffer->getWaitListFromSyncPoints(
  //     pSyncPointWaitList, numSyncPointsInWaitList);

  UR_CALL(commandListLocked->appendMemBufferWrite(
      hBuffer, blockingWrite, offset, size, pSrc, numEventsInWaitList,
      phEventWaitList, createEventIfRequested(eventPool.get(), phEvent, this)));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}


ur_result_t
ur_queue_batched_t::queueGetInfo(ur_queue_info_t propName,
                                            size_t propSize, void *pPropValue,
                                            size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  // TODO: consider support for queue properties and size
  switch ((uint32_t)propName) { // cast to avoid warnings on EXT enum values
  case UR_QUEUE_INFO_CONTEXT:
    return ReturnValue(hContext);
  case UR_QUEUE_INFO_DEVICE:
    return ReturnValue(hDevice);
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{RefCount.getCount()});
  case UR_QUEUE_INFO_FLAGS:
    return ReturnValue(flags);
  case UR_QUEUE_INFO_SIZE:
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_QUEUE_INFO_EMPTY: {
    auto status = ZE_CALL_NOCHECK(
        zeCommandListHostSynchronize,
        (commandListManager.get_no_lock()->getZeCommandList(), 0));
    if (status == ZE_RESULT_SUCCESS) {
      return ReturnValue(true);
    } else if (status == ZE_RESULT_NOT_READY) {
      return ReturnValue(false);
    } else {
      return ze2urResult(status);
    }
  }
  default:
    UR_LOG(ERR,
           "Unsupported ParamName in urQueueGetInfo: "
           "ParamName=ParamName={}(0x{})",
           propName, logger::toHex(propName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}


ur_result_t ur_queue_batched_t::queueGetNativeHandle(
    ur_queue_native_desc_t * /*pDesc*/, ur_native_handle_t *phNativeQueue) {
  *phNativeQueue = reinterpret_cast<ur_native_handle_t>(
      commandListManager.get_no_lock()->getZeCommandList());
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::queueFlush() {
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_queue_batched_t::enqueueEventsWaitWithBarrier(
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  TRACK_SCOPE_LATENCY(
      "ur_queue_batched_t::enqueueEventsWaitWithBarrier");
  // For in-order queue we don't need a real barrier, just wait for
  // requested events in potentially different queues and add a "barrier"
  // event signal because it is already guaranteed that previous commands
  // in this queue are completed when the signal is started. However, we do
  // need to use barrier if profiling is enabled: see
  // zeCommandListAppendWaitOnEvents
  if ((flags & UR_QUEUE_FLAG_PROFILING_ENABLE) != 0) {
    return commandListManager.lock()->appendEventsWaitWithBarrier(
        numEventsInWaitList, phEventWaitList,
        createEventIfRequested(eventPool.get(), phEvent, this));
  } else {
    return commandListManager.lock()->appendEventsWait(
        numEventsInWaitList, phEventWaitList,
        createEventIfRequested(eventPool.get(), phEvent, this));
  }
}

} // namespace v2