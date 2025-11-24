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

template <typename T>
struct TensorConcatWithOffsetsFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& device,
                  const std::vector<const T*>& input_data_ptrs,
                  const int64_t* offsets_data,
                  int32 num_inputs,
                  int64_t row_size,
                  T* output_data) {
    if (output_data == nullptr) {
      LOG(ERROR) << "TensorConcatWithOffsets GPU: Null output pointer";
      return;
    }

    // 处理每个输入tensor
    for (int32 i = 0; i < num_inputs; ++i) {
      const T* input_data = input_data_ptrs[i];
      const int64_t offset = offsets_data[i * 2];
      const int64_t length = offsets_data[i * 2 + 1];

      // 跳过空tensor
      if (length == 0 || input_data == nullptr) {
        continue;
      }

      // 计算需要处理的总元素数
      const int64_t total_elements = length * row_size;
      const int64_t num_blocks = 
          std::min(kMaxBlocks, (total_elements + kBlockSize - 1) / kBlockSize);

      if (num_blocks <= 0) {
        continue;
      }

      // 启动GPU kernel
      TensorConcatKernel<T><<<num_blocks, kBlockSize, 0, device.stream()>>>(
          input_data, offset, length, row_size, output_data);

      // 检查kernel启动错误
      const cudaError_t kernel_error = cudaGetLastError();
      if (kernel_error != cudaSuccess) {
        LOG(ERROR) << "TensorConcatWithOffsets: TensorConcatKernel launch failed for input "
                   << i << ": " << cudaGetErrorString(kernel_error);
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
