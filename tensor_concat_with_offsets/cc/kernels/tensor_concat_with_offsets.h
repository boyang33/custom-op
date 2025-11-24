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

#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_CONCAT_WITH_OFFSETS_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_CONCAT_WITH_OFFSETS_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace functor {

// ============================================================================
// TensorConcatWithOffsets算子配置常量
// ============================================================================

namespace tensor_concat_with_offsets_config {

// GPU硬件配置
constexpr int32 kBlockSize = 256;      // 每个block的线程数
constexpr int64_t kMaxBlocks = 65535;  // GPU硬件限制的最大block数

}  // namespace tensor_concat_with_offsets_config

/**
 * TensorConcatWithOffsets算子的核心functor接口
 *
 * 功能：将多个输入tensor沿第0维根据预计算的offsets合并到输出tensor
 *
 * 模板参数：
 *   Device: 计算设备类型（CPUDevice或GPUDevice）
 *   T: 数据类型（float, double, int32, int64等）
 *
 * 算法特点：
 *   - 支持任意维度的tensor（≥1维）
 *   - 沿第0维进行合并，其他维度保持不变
 *   - CPU版本使用std::memcpy
 *   - GPU版本使用高效的并行kernel
 */
template <typename Device, typename T>
struct TensorConcatWithOffsetsFunctor {
  /**
   * 执行tensor合并操作
   *
   * @param d 计算设备句柄
   * @param input_data_ptrs 输入tensor的数据指针列表
   * @param offsets_data 预计算的offsets数据（[N, 2]，每行为[offset, length]）
   * @param num_inputs 输入tensor的数量
   * @param row_size 每行的元素数量（除第0维外所有维度的乘积）
   * @param output_data 输出tensor的数据指针
   */
  void operator()(const Device& d,
                  const std::vector<const T*>& input_data_ptrs,
                  const int64_t* offsets_data,
                  int32 num_inputs,
                  int64_t row_size,
                  T* output_data);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_CONCAT_WITH_OFFSETS_H_
