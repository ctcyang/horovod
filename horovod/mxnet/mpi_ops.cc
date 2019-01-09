// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <chrono>
#include <memory>
#include <thread>
#include <atomic>

#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "mpi_ops.h"
#include "ready_event.h"
#include "tensor_util.h"

namespace horovod {
namespace mxnet {

static HandleManager handle_manager;

namespace {

std::string GetOpName(std::string prefix, const char* name, int handle) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

} // namespace

void DoAllreduce(NDArray* tensor, NDArray* output, const char* name,
                 int handle, Callback cb) {
  ThrowIfError(common::CheckInitialized());
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = std::make_shared<MXTensor<NDArray>>(output);

  handle_manager.AttachCallback(handle, cb);
  auto enqueue_result =
      EnqueueTensorAllreduce(hvd_context, hvd_tensor, hvd_output, nullptr,
                             GetOpName("allreduce", name, handle), device,
                             [handle](const Status& status) {
                               handle_manager.ExecuteCallback(handle);
                               handle_manager.MarkDone(handle, status);
                             });
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA
void DoAllreduceCudaOnCPU(NDArray* tensor, NDArray* output, const char* name,
                         int handle, Callback cb) {
  ThrowIfError(common::CheckInitialized());
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, tensor->dtype());
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(tensor);

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  handle_manager.AttachCallback(handle, cb);
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      GetOpName("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
        handle_manager.ExecuteCallback(handle);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);
}
#endif

void DoAllgather(NDArray* tensor, NDArray* output, const char* name,
                int handle, Callback cb) {
  ThrowIfError(common::CheckInitialized());
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);

  handle_manager.AttachCallback(handle, cb);
  auto enqueue_result =
      EnqueueTensorAllgather(hvd_context, hvd_tensor, nullptr,
                             GetOpName("allgather", name, handle), device,
                             [handle](const Status& status) {
                               handle_manager.ExecuteCallback(handle);
                               handle_manager.MarkDone(handle, status);
                             });
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA
void DoAllgatherCudaOnCPU(NDArray* tensor, NDArray* output, const char* name,
                          int handle, Callback cb) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_tensor = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, tensor->dtype());
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(tensor);

  auto hvd_cpu_output = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, output->dtype());
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  handle_manager.AttachCallback(handle, cb);
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      GetOpName("allgather", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_output, output](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_output->tensor(), output);
        handle_manager.ExecuteCallback(handle);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);
}
#endif

void DoBroadcast(NDArray* tensor, NDArray* output, int root_rank,
                const char* name, int handle, Callback cb) {
  ThrowIfError(common::CheckInitialized());
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      TensorUtil::Copy(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<MXTensor<NDArray>>(output);
  }

  handle_manager.AttachCallback(handle, cb);
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_tensor, hvd_output, root_rank, nullptr,
      GetOpName("broadcast", name, handle), device,
      [handle](const Status& status) {
        handle_manager.ExecuteCallback(handle);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA
void DoBroadcastCudaOnCPU(
    std::shared_ptr<MXTemporaryBuffer<NDArray>>& hvd_cpu_buffer, int root_rank,
    const char* name, int handle, Callback cb) {
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());
  auto ready_event =
      std::make_shared<MXReadyEvent<NDArray>>(hvd_cpu_buffer->tensor());

  handle_manager.AttachCallback(handle, cb);
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      GetOpName("broadcast", name, handle), CPU_DEVICE_ID,
      [handle](const Status& status) {
        handle_manager.ExecuteCallback(handle);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

extern "C" int horovod_mxnet_allreduce_async(NDArray* input, NDArray* output,
                                             const char* name, bool average,
                                             int* handle) {
  MX_API_BEGIN();
  *handle = handle_manager.AllocateHandle();
  auto allreduce_async_fn = [input, output,
                             name, handle](RunContext rctx,
                                           Callback cb) mutable {
    DoAllreduce(input, output, name, *handle, cb);
  };
#if HAVE_CUDA
  auto allreduce_async_cpu_fn = [input, output,
                                 name, handle](RunContext rctx,
                                               Callback cb) mutable {
    DoAllreduceCudaOnCPU(input, output, name, *handle, cb);
  };
#endif

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allreduce_async_cpu_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllreduce");
    // In-place
  } else {
    Engine::Get()->PushAsync(allreduce_async_cpu_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
  }
#else
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allreduce_async_fn, input->ctx(), {input->var()},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
    // In-place
  } else {
    Engine::Get()->PushAsync(allreduce_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
  }
#endif

  if (average) {
    *output /= horovod_size();
  }
  MX_API_END();
}

extern "C" int horovod_mxnet_allgather_async(NDArray* input, NDArray* output,
                                             const char* name,
                                             int* handle) {
  MX_API_BEGIN();
  *handle = handle_manager.AllocateHandle();
  auto allgather_async_fn = [input, output,
                             name, handle](RunContext rctx,
                                           Callback cb) mutable {
    DoAllgather(input, output, name, *handle, cb);
  };
#if HAVE_CUDA
  auto allgather_async_cpu_fn =
      [input, output, name, handle](RunContext rctx,
                                    Callback cb) mutable {
    DoAllgatherCudaOnCPU(input, output, name, *handle, cb);
  };
#endif

#if HAVE_CUDA && HOROVOD_GPU_ALLGATHER != 'M'
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allgather_async_cpu_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllgather");
    // In-place
  } else {
    Engine::Get()->PushAsync(allgather_async_cpu_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllgather");
  }
#else
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allgather_async_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllgather");
    // In-place
  } else {
    Engine::Get()->PushAsync(allgather_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllgather");
  }
#endif
  MX_API_END();
}

extern "C" int horovod_mxnet_broadcast_async(NDArray* input, NDArray* output,
                                             int root_rank, const char* name,
                                             int* handle) {
  MX_API_BEGIN();
  *handle = handle_manager.AllocateHandle();
  auto broadcast_async_fn = [input, output, name,
                             root_rank, handle](RunContext rctx,
                                                Callback cb) mutable {
    DoBroadcast(input, output, root_rank, name, *handle, cb);
  };

#if HAVE_CUDA && HOROVOD_GPU_BROADCAST != 'M'
  // Not in-place
  ThrowIfError(common::CheckInitialized());
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, input->dtype());
  TensorUtil::AsyncCopyCudaToCPU(input, hvd_cpu_buffer->tensor());
  auto broadcast_async_cpu_fn = [hvd_cpu_buffer, name, root_rank, *handle]
                                (RunContext rctx, Callback cb) mutable {
    DoBroadcastCudaOnCPU(hvd_cpu_buffer, root_rank, name, *handle, cb);
  };

  Engine::Get()->PushAsync(broadcast_async_cpu_fn, input->ctx(), {},
                           {output->var()}, FnProperty::kNormal, 0,
                           "HorovodBroadcast");

  TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
#else
  Engine::Get()->PushAsync(broadcast_async_fn, input->ctx(), {},
                           {output->var()}, FnProperty::kNormal, 0,
                           "HorovodBroadcast");
#endif
  MX_API_END();
}

extern "C" int horovod_mxnet_poll(int handle) {
  return handle_manager.PollHandle(handle) ? 1 : 0;
}

extern "C" int horovod_mxnet_wait_and_clear(int handle) {
  MX_API_BEGIN();
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
  MX_API_END();
}

} // namespace mxnet
} // namespace horovod
