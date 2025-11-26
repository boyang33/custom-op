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

// 前向声明
class OpKernelContext;

namespace functor {

// ============================================================================
// TensorSplitByOffset算子配置常量
// ============================================================================

namespace tensor_split_by_offsets_config {

// CPU优化配置
constexpr int64_t kZeroCopyThreshold = 0;  // 零拷贝阈值：切片包含的元素数量
constexpr int64_t kDefaultParallelThreshold = 1024 * 1024;  // CPU并行化阈值：1MB
constexpr int kDefaultMaxParallelism = 0;  // 最大并行度，0表示使用系统默认
constexpr int kMinOutputsForParallel = 4;  // 最小输出数量才考虑并行化

// GPU优化配置
constexpr int32 kBlockSize = 256;      // 每个block的线程数
constexpr int64_t kMaxBlocks = 65535;  // GPU硬件限制的最大block数
constexpr int kStreamThreshold = 8;    // GPU多流阈值：输出数量
constexpr int kNumStreams = 4;         // GPU多流数量

}  // namespace tensor_split_by_offsets_config

/**
 * 批量拆分任务描述结构
 * 用于描述一个需要执行的拆分操作
 */
struct SplitTask {
  int output_index;        // 输出tensor的索引
  const void* input_data;  // 输入数据指针（泛型指针）
  int64_t start_row;       // 起始行索引
  int64_t row_count;       // 拷贝的行数
  int64_t row_size;        // 每行的元素数量
  void* output_data;       // 输出数据指针（泛型指针）
};

/**
 * TensorSplitByOffset算子的核心functor接口
 *
 * 功能：将多维输入tensor沿第0维按指定偏移量批量拆分到多个输出tensor
 *
 * 模板参数：
 *   Device: 计算设备类型（CPUDevice或GPUDevice）
 *   T: 数据类型（float, double, int32, int64等）
 *
 * 算法特点：
 *   - 支持任意维度的tensor（≥1维）
 *   - 沿第0维进行拆分，其他维度保持不变
 *   - 批量处理多个输出，支持并行化
 *   - CPU版本支持多线程并行复制
 *   - GPU版本支持多流并发执行
 */
template <typename Device, typename T>
struct TensorSplitByOffsetsFunctor {
  /**
   * 批量执行多个输出的tensor拆分操作
   *
   * @param context 操作上下文（用于CPU并行化）
   * @param d 计算设备句柄
   * @param tasks 拆分任务列表
   */
  void operator()(OpKernelContext* context,
                  const Device& d,
                  const std::vector<SplitTask>& tasks);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // KERNEL_SPLIT_BY_OFFSET_H_
