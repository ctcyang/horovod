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

std::string GetOpName(std::string prefix, char* name) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }
  return std::string();
}

std::string GetOpNameHandle(std::string prefix, const std::string& name, int handle) {
  if (name.length() != 0) {
    return name;
  }
  return prefix + ".noname." + std::to_string(handle);
}

} // namespace

int DoAllreduce(NDArray* tensor, NDArray* output, int average, const std::string& name, Callback cb) {
  ThrowIfError(common::CheckInitialized());
 
  auto handle = handle_manager.AllocateHandle(cb);
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = std::make_shared<MXTensor<NDArray>>(output);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, nullptr,
      GetOpNameHandle("allreduce", name, handle), device,
      [handle, average, output](const Status& status) {
        handle_manager.MarkDone(handle, status);
        handle_manager.ExecuteCallback(handle);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
int DoAllreduceCudaOnCPU(NDArray* tensor, NDArray* output, int average, std::string& name, Callback cb) {
  ThrowIfError(common::CheckInitialized());
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer =
      std::make_shared<MXTemporaryBuffer<NDArray>>(CPU_DEVICE_ID, tensor->dtype());
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(tensor);

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto handle = handle_manager.AllocateHandle(cb);
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      GetOpNameHandle("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, average, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
        handle_manager.MarkDone(handle, status);
        handle_manager.ExecuteCallback(handle);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

int DoAllgather(NDArray* tensor, NDArray* output, std::string& name, Callback cb) {
  ThrowIfError(common::CheckInitialized());
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);

  auto handle = handle_manager.AllocateHandle(cb);
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_tensor, nullptr, GetOpNameHandle("allgather", name, handle),
      device, [handle](const Status& status) {
        handle_manager.MarkDone(handle, status);
        handle_manager.ExecuteCallback(handle);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
int DoAllgatherCudaOnCPU(NDArray* tensor, NDArray* output, std::string& name, Callback cb) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_tensor =
      std::make_shared<MXTemporaryBuffer<NDArray>>(CPU_DEVICE_ID, tensor->dtype());
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(tensor);

  auto hvd_cpu_output =
      std::make_shared<MXTemporaryBuffer<NDArray>>(CPU_DEVICE_ID, output->dtype());
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  auto handle = handle_manager.AllocateHandle(cb);
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      GetOpNameHandle("allgather", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_output, output](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_output->tensor(), output);
        handle_manager.MarkDone(handle, status);
        handle_manager.ExecuteCallback(handle);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

int DoBroadcast(NDArray* tensor, NDArray* output, int root_rank, std::string& name, Callback cb) {
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

  auto handle = handle_manager.AllocateHandle(cb);
  auto enqueue_result =
      EnqueueTensorBroadcast(hvd_context, hvd_tensor, hvd_output, root_rank,
                             nullptr, GetOpNameHandle("broadcast", name, handle),
                             device, [handle](const Status& status) {
                               handle_manager.MarkDone(handle, status);
                               handle_manager.ExecuteCallback(handle);
                             });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
int DoBroadcastCudaOnCPU(std::shared_ptr<MXTemporaryBuffer<NDArray>>& hvd_cpu_buffer, int root_rank, std::string& name, Callback cb) {
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(hvd_cpu_buffer->tensor());

  auto handle = handle_manager.AllocateHandle(cb);
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      GetOpNameHandle("broadcast", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_buffer](const Status& status) {
        handle_manager.MarkDone(handle, status);
        handle_manager.ExecuteCallback(handle);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

// Do AllReduce on GPU only if src and dst are on GPU
// Otherwise do AllReduce on CPU
extern "C" int horovod_mxnet_allreduce_async(
    NDArray* input, NDArray* output, int average, char* name) {

  std::string new_name = GetOpName("allreduce", name);
  auto allreduce_async_fn = [input, output, new_name, average](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
      DoAllreduce(input, output, average, new_name, cb);
  };
#if HAVE_CUDA
  auto allreduce_async_cpu_fn = [input, output, new_name, average](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
      DoAllreduceCudaOnCPU(input, output, average, new_name, cb);
  };
#endif

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(
      allreduce_async_cpu_fn,
      input->ctx(),
      {input->var()},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodAllreduce");
  // In-place
  } else {
    Engine::Get()->PushAsync(
      allreduce_async_cpu_fn,
      input->ctx(),
      {},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodAllreduce");
  }
#else
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(
      allreduce_async_fn,
      input->ctx(),
      {input->var()},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodAllreduce");
  // In-place
  } else {
    Engine::Get()->PushAsync(
      allreduce_async_fn,
      input->ctx(),
      {},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodAllreduce");
  }
#endif
  return 0;
}

extern "C" int horovod_mxnet_allgather_async(
    NDArray* input, NDArray* output, char* name) {

  std::string new_name = GetOpName("allgather", name);
  auto allgather_async_fn = [input, output, new_name](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
    DoAllgather(input, output, new_name, cb);
  };
#if HAVE_CUDA
  auto allgather_async_cpu_fn = [input, output, new_name](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
    DoAllgatherCudaOnCPU(input, output, new_name, cb);
  };
#endif

#if HAVE_CUDA && HOROVOD_GPU_ALLGATHER != 'M'
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(
      allgather_async_cpu_fn,
      Context::CPU(0),
      {input->var()},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodAllgather");
  // In-place
  } else {
    Engine::Get()->PushAsync(
      allgather_async_cpu_fn,
      Context::CPU(0),
      {},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodAllgather");
  }
#else
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(
      allgather_async_fn,
      Context::CPU(0),
      {input->var()},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodAllgather");
  // In-place
  } else {
    Engine::Get()->PushAsync(
      allgather_async_fn,
      Context::CPU(0),
      {},
      {output->var()},
      FnProperty::kNormal,
      0,
      "HorovodAllgather");
  }
#endif
  return 0;
}

extern "C" int horovod_mxnet_broadcast_async(
    NDArray* input, NDArray* output, int root_rank, char* name) {
   
  std::string new_name = GetOpName("broadcast", name);
  auto broadcast_async_fn = [input, output, new_name, root_rank](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
    DoBroadcast(input, output, root_rank, new_name, cb);
  };

#if HAVE_CUDA && HOROVOD_GPU_BROADCAST != 'M'
  // Not in-place
  ThrowIfError(common::CheckInitialized());
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer =
      std::make_shared<MXTemporaryBuffer<NDArray>>(CPU_DEVICE_ID, input->dtype());
  TensorUtil::AsyncCopyCudaToCPU(input, hvd_cpu_buffer->tensor());
  auto broadcast_async_cpu_fn = [hvd_cpu_buffer, new_name, root_rank](
      RunContext rctx, Engine::CallbackOnComplete cb) mutable {
    DoBroadcastCudaOnCPU(hvd_cpu_buffer, root_rank, new_name, cb);
  };

  Engine::Get()->PushAsync(
    broadcast_async_cpu_fn,
    Context::CPU(0),
    {},
    {output->var()},
    FnProperty::kNormal,
    0,
    "HorovodBroadcast");

  TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
#else
  Engine::Get()->PushAsync(
    broadcast_async_fn,
    Context::CPU(0),
    {},
    {output->var()},
    FnProperty::kNormal,
    0,
    "HorovodBroadcast");
#endif
  return 0;
}

extern "C" int horovod_mxnet_poll(int handle) {
  return handle_manager.PollHandle(handle) ? 1 : 0;
}

extern "C" void horovod_mxnet_wait_and_clear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

} // namespace mxnet
} // namespace horovod
