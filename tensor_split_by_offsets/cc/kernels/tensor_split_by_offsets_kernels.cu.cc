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

namespace {

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

template <typename T>
struct TensorSplitByOffsetsFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& device,
                  const T* input_data,
                  int64_t start_row,
                  int64_t row_count,
                  int64_t row_size,
                  T* output_data) {
    if (row_count == 0 || row_size == 0) {
      return;
    }

    if (input_data == nullptr || output_data == nullptr) {
      LOG(ERROR) << "TensorSplitByOffset: Null pointer detected in GPU functor";
      return;
    }

    // 计算需要处理的总元素数
    const int64_t total_elements = row_count * row_size;
    const int64_t num_blocks = std::min(kMaxBlocks, (total_elements + kBlockSize - 1) / kBlockSize);

    if (num_blocks <= 0) {
      return;
    }

    // 启动 GPU kernel
    TensorSplitKernel<T><<<num_blocks, kBlockSize, 0, device.stream()>>>(
        input_data, start_row, row_count, row_size, output_data);

    const cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
      LOG(ERROR) << "TensorSplitByOffset: TensorSplitKernel launch failed: "
                 << cudaGetErrorString(kernel_error);
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
