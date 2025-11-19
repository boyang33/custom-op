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

#ifndef KERNEL_TENSOR_SPLIT_BY_OFFSETS_H_
#define KERNEL_TENSOR_SPLIT_BY_OFFSETS_H_

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace functor {

// ============================================================================
// TensorSplitByOffset算子配置常量
// ============================================================================

namespace tensor_split_by_offsets_config {

// CPU优化配置
constexpr int64_t kZeroCopyThreshold = 0;  // 零拷贝阈值：切片包含的元素数量

// GPU硬件配置
constexpr int32 kBlockSize = 256;      // 每个block的线程数
constexpr int64_t kMaxBlocks = 65535;  // GPU硬件限制的最大block数

}  // namespace tensor_split_by_offsets_config

/**
 * TensorSplitByOffset算子的核心functor接口
 *
 * 功能：将多维输入tensor沿第0维按指定偏移量拆分到单个输出tensor
 *
 * 模板参数：
 *   Device: 计算设备类型（CPUDevice或GPUDevice）
 *   T: 数据类型（float, double, int32, int64等）
 *
 * 算法特点：
 *   - 支持任意维度的tensor（≥1维）
 *   - 沿第0维进行拆分，其他维度保持不变
 *   - CPU版本使用std::memcpy
 *   - GPU版本使用高效的并行kernel
 */
template <typename Device, typename T>
struct TensorSplitByOffsetsFunctor {
  /**
   * 执行单个输出的tensor拆分操作
   *
   * @param d 计算设备句柄
   * @param input_data 输入tensor的数据指针
   * @param start_row 起始行索引
   * @param row_count 拷贝的行数
   * @param row_size 每行的元素数量（除第0维外所有维度的乘积）
   * @param output_data 输出tensor的数据指针
   */
  void operator()(const Device& d,
                  const T* input_data,
                  int64_t start_row,
                  int64_t row_count,
                  int64_t row_size,
                  T* output_data);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // KERNEL_SPLIT_BY_OFFSET_H_
