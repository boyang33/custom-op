/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensor_split_by_offsets.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

using GPUDevice = Eigen::GpuDevice;
using tensor_split_by_offsets_config::kBlockSize;
using tensor_split_by_offsets_config::kMaxBlocks;
using tensor_split_by_offsets_config::kStreamThreshold;
using tensor_split_by_offsets_config::kNumStreams;

namespace {

/**
 * GPU kernel：单个输出的拆分操作
 */
template <typename T>
__global__ void TensorSplitKernel(const T* input_data,
                                  const int64_t start_row,
                                  const int64_t row_count,
                                  const int64_t row_size,
                                  T* __restrict__ output_data) {
  const int64_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_output_elements = row_count * row_size;

  if (element_index >= total_output_elements) {
    return;
  }

  const int64_t output_row_index = element_index / row_size;
  const int64_t column_index = element_index % row_size;
  const int64_t input_row_index = start_row + output_row_index;
  const int64_t input_index = input_row_index * row_size + column_index;

  output_data[element_index] = input_data[input_index];
}

}  // namespace

/**
 * GPU二级Functor：处理单个输出的拆分操作
 * 
 * 与一级Functor的区别：
 * 1. 接受stream参数，支持多流并发
 * 2. 处理单个任务，而非批量任务
 * 3. 不需要context参数
 */
template <typename T>
struct TensorSplitByOffsetsGPUSingleTaskFunctor {
  void operator()(const GPUDevice& device,
                  cudaStream_t stream,
                  const T* input_data,
                  int64_t start_row,
                  int64_t row_count,
                  int64_t row_size,
                  T* output_data) {
    if (row_count == 0 || row_size == 0) {
      return;
    }

    if (input_data == nullptr || output_data == nullptr) {
      LOG(ERROR) << "TensorSplitByOffset: Null pointer detected in GPU single task functor";
      return;
    }

    // 计算需要处理的总元素数
    const int64_t total_elements = row_count * row_size;
    const int64_t num_blocks = std::min(kMaxBlocks, (total_elements + kBlockSize - 1) / kBlockSize);

    if (num_blocks <= 0) {
      return;
    }

    // 在指定的stream上启动kernel
    TensorSplitKernel<T><<<num_blocks, kBlockSize, 0, stream>>>(
        input_data, start_row, row_count, row_size, output_data);

    const cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
      LOG(ERROR) << "TensorSplitByOffset: TensorSplitKernel launch failed: "
                 << cudaGetErrorString(kernel_error);
    }
  }
};

/**
 * GPU设备上的TensorSplitByOffset functor实现
 *
 * 主要特性：
 * 1. 批量处理多个输出，支持多流并发
 * 2. 使用二级Functor处理单个任务
 * 3. 当输出数量超过阈值时，启用多流优化
 * 
 * 参数使用：
 *   - context: 未使用（GPU不需要CPU线程池）
 *   - device: 使用，用于获取默认stream和管理CUDA流
 */
template <typename T>
struct TensorSplitByOffsetsFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context,
                  const GPUDevice& device,
                  const std::vector<SplitTask>& tasks) {
    if (tasks.empty()) {
      return;
    }

    const int num_tasks = static_cast<int>(tasks.size());

    // 如果任务数量超过阈值，使用多流优化
    if (num_tasks > kStreamThreshold) {
      // 创建多个CUDA流以实现并发执行
      cudaStream_t streams[kNumStreams];
      
      // 第一个stream复用device的主stream
      streams[0] = device.stream();
      
      // 创建额外的stream
      for (int s = 1; s < kNumStreams; ++s) {
        cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
      }

      // 使用二级Functor处理每个任务
      TensorSplitByOffsetsGPUSingleTaskFunctor<T> single_task_functor;
      
      // 将任务分配到不同的stream上，以轮询方式
      for (int i = 0; i < num_tasks; ++i) {
        const SplitTask& task = tasks[i];
        const int stream_id = i % kNumStreams;
        
        single_task_functor(
            device,
            streams[stream_id],
            static_cast<const T*>(task.input_data),
            task.start_row,
            task.row_count,
            task.row_size,
            static_cast<T*>(task.output_data));
      }

      // 同步并销毁创建的stream
      for (int s = 1; s < kNumStreams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
      }
    } else {
      // 任务数量少，使用默认stream顺序执行
      TensorSplitByOffsetsGPUSingleTaskFunctor<T> single_task_functor;
      
      for (const auto& task : tasks) {
        single_task_functor(
            device,
            device.stream(),
            static_cast<const T*>(task.input_data),
            task.start_row,
            task.row_count,
            task.row_size,
            static_cast<T*>(task.output_data));
      }
    }
  }
};

// 显式实例化支持的类型
template struct TensorSplitByOffsetsFunctor<GPUDevice, float>;
template struct TensorSplitByOffsetsFunctor<GPUDevice, double>;
template struct TensorSplitByOffsetsFunctor<GPUDevice, int32>;
template struct TensorSplitByOffsetsFunctor<GPUDevice, int64>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
