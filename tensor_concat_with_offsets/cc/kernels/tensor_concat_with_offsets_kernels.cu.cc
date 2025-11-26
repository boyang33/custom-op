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

#include "tensor_concat_with_offsets.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

using GPUDevice = Eigen::GpuDevice;
using tensor_concat_with_offsets_config::kBlockSize;
using tensor_concat_with_offsets_config::kMaxBlocks;
using tensor_concat_with_offsets_config::kStreamThreshold;
using tensor_concat_with_offsets_config::kNumStreams;

namespace {

/**
 * GPU kernel: 将单个输入tensor复制到输出tensor的指定位置
 *
 * 策略：每个线程处理一个元素的复制
 *
 * @param input_data 输入tensor的数据指针
 * @param output_offset 在输出tensor中的起始偏移量（行索引）
 * @param length 要复制的行数
 * @param row_size 每行的元素数量
 * @param output_data 输出tensor的数据指针
 */
template <typename T>
__global__ void TensorConcatKernel(const T* __restrict__ input_data,
                                   const int64_t output_offset,
                                   const int64_t length,
                                   const int64_t row_size,
                                   T* __restrict__ output_data) {
  // 计算当前线程处理的元素索引
  const int64_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = length * row_size;

  if (element_index >= total_elements) {
    return;
  }

  // 计算在输入tensor中的位置
  const int64_t input_row = element_index / row_size;
  const int64_t col = element_index % row_size;

  // 计算在输出tensor中的位置
  const int64_t output_row = output_offset + input_row;
  const int64_t output_index = output_row * row_size + col;

  // 执行复制
  output_data[output_index] = input_data[element_index];
}

}  // namespace

/**
 * GPU 设备特化：TensorConcatWithOffsets Functor 实现
 *
 * 策略：
 *   1. 使用 CUDA kernel 并行复制每个输入 tensor
 *   2. 大量输入（>8）时使用多 stream 优化，提升并发度
 *   3. 少量输入时串行启动 kernel，避免 stream 创建开销
 *
 * 参数使用：
 *   - context: 未使用（仅为统一接口）
 *   - device: 使用，用于获取 CUDA stream
 *
 * 多 stream 优化：
 *   - 阈值：num_inputs > kStreamThreshold（默认 8）
 *   - 并行度：kNumStreams（默认 4 个 stream）
 *   - 策略：轮流分配输入到不同 stream，最大化 GPU 利用率
 */
template <typename T>
struct TensorConcatWithOffsetsFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context,
                  const GPUDevice& device,
                  const std::vector<const T*>& input_data_ptrs,
                  const int64_t* offsets_data,
                  int32 num_inputs,
                  int64_t row_size,
                  T* output_data) {
    if (output_data == nullptr) {
      LOG(ERROR) << "TensorConcatWithOffsets GPU: Null output pointer";
      return;
    }

    // 优化：对于大量输入，使用多stream并行启动kernel
    if (num_inputs > kStreamThreshold) {
      // 多stream并行模式
      cudaStream_t streams[kNumStreams];
      
      // 创建streams（复用device的stream作为stream[0]）
      streams[0] = device.stream();
      for (int s = 1; s < kNumStreams; ++s) {
        cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
      }
      
      // 并行启动kernels
      for (int32 i = 0; i < num_inputs; ++i) {
        const T* input_data = input_data_ptrs[i];
        const int64_t offset = offsets_data[i * 2];
        const int64_t length = offsets_data[i * 2 + 1];
        
        if (length == 0 || input_data == nullptr) {
          continue;
        }
        
        const int64_t total_elements = length * row_size;
        const int64_t num_blocks = 
            std::min(kMaxBlocks, (total_elements + kBlockSize - 1) / kBlockSize);
        
        if (num_blocks <= 0) {
          continue;
        }
        
        // 轮流使用不同的stream
        int stream_id = i % kNumStreams;
        TensorConcatKernel<T><<<num_blocks, kBlockSize, 0, streams[stream_id]>>>(
            input_data, offset, length, row_size, output_data);
      }
      
      // 同步所有streams（除了stream[0]，它会被TF自动同步）
      for (int s = 1; s < kNumStreams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
      }
      
      // 检查错误（在同步后）
      const cudaError_t sync_error = cudaGetLastError();
      if (sync_error != cudaSuccess) {
        LOG(ERROR) << "TensorConcatWithOffsets GPU: Kernel execution failed: "
                   << cudaGetErrorString(sync_error);
      }
      
    } else {
      // 少量输入：使用原始串行模式（避免stream创建开销）
      for (int32 i = 0; i < num_inputs; ++i) {
        const T* input_data = input_data_ptrs[i];
        const int64_t offset = offsets_data[i * 2];
        const int64_t length = offsets_data[i * 2 + 1];
        
        if (length == 0 || input_data == nullptr) {
          continue;
        }
        
        const int64_t total_elements = length * row_size;
        const int64_t num_blocks = 
            std::min(kMaxBlocks, (total_elements + kBlockSize - 1) / kBlockSize);
        
        if (num_blocks <= 0) {
          continue;
        }
        
        TensorConcatKernel<T><<<num_blocks, kBlockSize, 0, device.stream()>>>(
            input_data, offset, length, row_size, output_data);
        
        const cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess) {
          LOG(ERROR) << "TensorConcatWithOffsets: TensorConcatKernel launch failed for input "
                     << i << ": " << cudaGetErrorString(kernel_error);
        }
      }
    }
  }
};

// 显式实例化支持的类型
template struct TensorConcatWithOffsetsFunctor<GPUDevice, float>;
template struct TensorConcatWithOffsetsFunctor<GPUDevice, double>;
template struct TensorConcatWithOffsetsFunctor<GPUDevice, int32>;
template struct TensorConcatWithOffsetsFunctor<GPUDevice, int64>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
