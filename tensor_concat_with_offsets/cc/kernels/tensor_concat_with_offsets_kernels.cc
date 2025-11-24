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
#endif  // GOOGLE_CUDA

#include "tensor_concat_with_offsets.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/ops_util.h"

#include <chrono>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <climits>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

/**
 * 性能诊断工具
 */
namespace perf_diagnostics {

// 环境变量控制开关
// 注意：每次都读取环境变量，支持运行时动态修改
static bool IsDebugEnabled() {
  const char* env = std::getenv("TF_CONCAT_DEBUG");
  return env != nullptr && std::string(env) == "1";
}

// 计时器
class Timer {
 public:
  Timer() : start_(std::chrono::high_resolution_clock::now()) {}
  
  double ElapsedMicros() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
  }
  
  double ElapsedMillis() const {
    return ElapsedMicros() / 1000.0;
  }
  
 private:
  std::chrono::high_resolution_clock::time_point start_;
};

// 统计信息收集器
struct ConcatStats {
  int num_inputs = 0;
  int64_t total_elements = 0;
  int64_t total_bytes = 0;
  int64_t min_chunk_size = INT64_MAX;
  int64_t max_chunk_size = 0;
  int64_t avg_chunk_size = 0;
  int small_chunks = 0;   // < 1KB
  int medium_chunks = 0;  // 1KB - 64KB
  int large_chunks = 0;   // > 64KB
  int empty_chunks = 0;
  
  double validation_time_ms = 0;
  double offset_calc_time_ms = 0;
  double alloc_output_time_ms = 0;
  double alloc_offsets_time_ms = 0;
  double copy_time_ms = 0;
  double total_time_ms = 0;
  
  void AnalyzeChunkSizes(const std::vector<int64_t>& chunk_sizes, size_t element_size) {
    for (int64_t size : chunk_sizes) {
      int64_t bytes = size * element_size;
      
      if (size == 0) {
        empty_chunks++;
      } else if (bytes < 1024) {
        small_chunks++;
      } else if (bytes < 65536) {
        medium_chunks++;
      } else {
        large_chunks++;
      }
      
      if (size > 0) {
        min_chunk_size = std::min(min_chunk_size, size);
        max_chunk_size = std::max(max_chunk_size, size);
      }
    }
    
    if (num_inputs > empty_chunks) {
      avg_chunk_size = total_elements / (num_inputs - empty_chunks);
    }
  }
  
  void Print() const {
    LOG(INFO) << "===== TensorConcatWithOffsets Performance Diagnostics =====";
    LOG(INFO) << "Input Configuration:";
    LOG(INFO) << "  - Num inputs: " << num_inputs;
    LOG(INFO) << "  - Total elements: " << total_elements;
    LOG(INFO) << "  - Total bytes: " << total_bytes << " (" 
              << (total_bytes / 1024.0 / 1024.0) << " MB)";
    LOG(INFO) << "  - Empty chunks: " << empty_chunks;
    
    LOG(INFO) << "Chunk Size Distribution:";
    LOG(INFO) << "  - Small chunks (<1KB): " << small_chunks;
    LOG(INFO) << "  - Medium chunks (1-64KB): " << medium_chunks;
    LOG(INFO) << "  - Large chunks (>64KB): " << large_chunks;
    LOG(INFO) << "  - Min chunk size: " << min_chunk_size << " elements";
    LOG(INFO) << "  - Max chunk size: " << max_chunk_size << " elements";
    LOG(INFO) << "  - Avg chunk size: " << avg_chunk_size << " elements";
    
    LOG(INFO) << "Timing Breakdown:";
    LOG(INFO) << "  - Validation: " << validation_time_ms << " ms ("
              << (validation_time_ms / total_time_ms * 100) << "%)";
    LOG(INFO) << "  - Offset calculation: " << offset_calc_time_ms << " ms ("
              << (offset_calc_time_ms / total_time_ms * 100) << "%)";
    LOG(INFO) << "  - Offsets allocation: " << alloc_offsets_time_ms << " ms ("
              << (alloc_offsets_time_ms / total_time_ms * 100) << "%)";
    LOG(INFO) << "  - Output allocation: " << alloc_output_time_ms << " ms ("
              << (alloc_output_time_ms / total_time_ms * 100) << "%)";
    LOG(INFO) << "  - Data copy: " << copy_time_ms << " ms ("
              << (copy_time_ms / total_time_ms * 100) << "%)";
    LOG(INFO) << "  - TOTAL: " << total_time_ms << " ms";
    
    LOG(INFO) << "Performance Metrics:";
    if (copy_time_ms > 0) {
      double bandwidth_gbps = (total_bytes / 1024.0 / 1024.0 / 1024.0) / 
                              (copy_time_ms / 1000.0);
      LOG(INFO) << "  - Copy bandwidth: " << bandwidth_gbps << " GB/s";
      LOG(INFO) << "  - Avg copy time per chunk: " 
                << (copy_time_ms / num_inputs) << " ms";
    }
    
    LOG(INFO) << "Optimization Recommendations:";
    
    // 推荐1: 内存分配优化
    if (alloc_output_time_ms > total_time_ms * 0.3) {
      LOG(WARNING) << "  ⚠ Memory allocation takes " 
                   << (alloc_output_time_ms / total_time_ms * 100) 
                   << "% of total time!";
      LOG(WARNING) << "    → Consider: Tensor pooling / pre-allocation";
      LOG(WARNING) << "    → Consider: Reduce allocation frequency";
    }
    
    // 推荐2: 小块批量优化
    if (small_chunks > num_inputs * 0.5 && num_inputs > 10) {
      LOG(WARNING) << "  ⚠ High ratio of small chunks (" << small_chunks 
                   << "/" << num_inputs << ")";
      LOG(WARNING) << "    → Consider: Batch memcpy for small chunks";
      LOG(WARNING) << "    → Consider: Buffer pooling strategy";
    }
    
    // 推荐3: 大块使用GPU
    if (large_chunks > 5 && total_bytes > 10 * 1024 * 1024) {
      LOG(WARNING) << "  ⚠ Large data volume (" 
                   << (total_bytes / 1024.0 / 1024.0) << " MB) with " 
                   << large_chunks << " large chunks";
      LOG(WARNING) << "    → Consider: GPU acceleration";
    }
    
    // 推荐4: 异步执行
    if (total_time_ms > 10.0) {
      LOG(WARNING) << "  ⚠ Total execution time > 10ms";
      LOG(WARNING) << "    → Consider: Async execution";
      LOG(WARNING) << "    → Consider: Pipeline parallelism";
    }
    
    // 推荐5: 空块过滤
    if (empty_chunks > num_inputs * 0.2) {
      LOG(WARNING) << "  ⚠ High ratio of empty chunks (" << empty_chunks 
                   << "/" << num_inputs << ")";
      LOG(WARNING) << "    → Consider: Pre-filter empty inputs";
    }
    
    LOG(INFO) << "========================================================";
  }
};

}  // namespace perf_diagnostics

/**
 * 内存对齐工具函数
 */
namespace alignment_utils {

/**
 * 计算对齐后的偏移量
 * @param current_offset 当前偏移量
 * @param alignment 对齐字节数
 * @param element_size 元素大小（字节）
 * @return 对齐后的偏移量（以元素为单位）
 */
inline int64_t AlignOffset(int64_t current_offset, int32 alignment, int64_t element_size) {
  // 将元素偏移转换为字节偏移
  int64_t byte_offset = current_offset * element_size;

  // 计算对齐后的字节偏移，向上取整
  int64_t aligned_byte_offset = ((byte_offset + alignment - 1) / alignment) * alignment;

  // 转换回元素偏移
  return aligned_byte_offset / element_size;
}

}  // namespace alignment_utils

/**
 * CPU设备上的TensorConcatWithOffsets functor实现
 *
 * 主要职责：
 * 1. 根据预计算的offsets执行高效的内存复制
 * 2. 使用std::memcpy优化数据传输
 */
template <typename T>
struct TensorConcatWithOffsetsFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  const std::vector<const T*>& input_data_ptrs,
                  const int64_t* offsets_data,
                  int32 num_inputs,
                  int64_t row_size,
                  T* output_data) {
    if (output_data == nullptr) {
      LOG(ERROR) << "TensorConcatWithOffsets CPU: Null output pointer";
      return;
    }

    // 处理每个输入tensor的数据复制
    for (int32 i = 0; i < num_inputs; ++i) {
      const T* input_data = input_data_ptrs[i];
      const int64_t offset = offsets_data[i * 2];
      const int64_t length = offsets_data[i * 2 + 1];

      if (length > 0 && input_data != nullptr) {
        T* dst = output_data + offset * row_size;
        const int64_t total_elements = length * row_size;
        const size_t copy_size = static_cast<size_t>(total_elements) * sizeof(T);
        std::memcpy(dst, input_data, copy_size);
      }
    }
  }
};

template <typename Device, typename T>
class TensorConcatWithOffsetsOp : public OpKernel {
 public:
  explicit TensorConcatWithOffsetsOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &num_inputs_));
    OP_REQUIRES_OK(context, context->GetAttr("alignment", &alignment_));
    OP_REQUIRES_OK(context, context->GetAttr("use_alignment", &use_alignment_));

    OP_REQUIRES(context, num_inputs_ > 0,
                errors::InvalidArgument("N must be > 0, got ", num_inputs_));
    OP_REQUIRES(
        context, alignment_ > 0 && (alignment_ & (alignment_ - 1)) == 0,
        errors::InvalidArgument("alignment must be a positive power of 2, got ", alignment_));

    // 确保对齐值合理（不超过4KB，避免过度内存浪费）
    OP_REQUIRES(context, alignment_ <= 4096,
                errors::InvalidArgument("alignment too large (max 4096), got ", alignment_));
  }

  void Compute(OpKernelContext* context) override {
    using namespace perf_diagnostics;
    
    // 全局计时
    Timer total_timer;
    const bool debug_enabled = IsDebugEnabled();
    
    // 统计信息
    ConcatStats stats;
    stats.num_inputs = num_inputs_;
    
    // 阶段1: 收集输入信息并验证
    Timer validation_timer;
    std::vector<const T*> input_data_ptrs(num_inputs_);
    std::vector<int64_t> input_lengths(num_inputs_);

    const Tensor& first_tensor = context->input(0);
    const TensorShape& reference_shape = first_tensor.shape();
    const int rank = reference_shape.dims();

    OP_REQUIRES(context, rank >= 1,
                errors::InvalidArgument("All input tensors must have at least 1 dimension"));

    // 计算每行元素数（第0维之外的所有元素）
    int64_t row_size = 1;
    for (int dim = 1; dim < rank; ++dim) {
      row_size *= reference_shape.dim_size(dim);
    }

    // 收集所有输入并验证形状
    int64_t total_original_elements = 0;
    std::vector<int64_t> chunk_elements;  // 用于统计
    
    for (int i = 0; i < num_inputs_; ++i) {
      const Tensor& input_tensor = context->input(i);
      const TensorShape& current_shape = input_tensor.shape();

      OP_REQUIRES(
          context, current_shape.dims() == rank,
          errors::InvalidArgument("All input tensors must have the same rank. ", "Expected rank ",
                                  rank, " but input ", i, " has rank ", current_shape.dims()));

      for (int dim = 1; dim < rank; ++dim) {
        OP_REQUIRES(
            context, current_shape.dim_size(dim) == reference_shape.dim_size(dim),
            errors::InvalidArgument("All input tensors must have the same shape ",
                                    "except for dimension 0. Dimension ", dim,
                                    " mismatch: expected ", reference_shape.dim_size(dim),
                                    " but input ", i, " has ", current_shape.dim_size(dim)));
      }

      const int64_t length = input_tensor.dim_size(0);
      input_lengths[i] = length;
      input_data_ptrs[i] = length > 0 ? input_tensor.flat<T>().data() : nullptr;
      total_original_elements += length;
      
      chunk_elements.push_back(length * row_size);
    }
    
    stats.validation_time_ms = validation_timer.ElapsedMillis();

    // 阶段2: 分配offsets tensor并预计算offsets
    Timer offset_alloc_timer;
    TensorShape offsets_shape({num_inputs_, 2});
    Tensor* offsets_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, offsets_shape, &offsets_tensor));
    stats.alloc_offsets_time_ms = offset_alloc_timer.ElapsedMillis();
    
    // TensorFlow的int64类型需要转换为int64_t
    int64_t* offsets_data = reinterpret_cast<int64_t*>(offsets_tensor->flat<int64>().data());

    // 阶段3: 预计算offsets
    Timer offset_calc_timer;
    const int64_t element_size = sizeof(T);
    int64_t current_offset = 0;

    for (int i = 0; i < num_inputs_; ++i) {
      int64_t actual_offset = current_offset;

      if (use_alignment_ && input_lengths[i] > 0) {
        actual_offset = alignment_utils::AlignOffset(current_offset, alignment_, element_size);
      }

      offsets_data[i * 2] = actual_offset;
      offsets_data[i * 2 + 1] = input_lengths[i];

      current_offset = actual_offset + input_lengths[i];
    }

    const int64_t total_elements = current_offset;
    stats.total_elements = total_elements * row_size;
    stats.total_bytes = stats.total_elements * element_size;
    stats.offset_calc_time_ms = offset_calc_timer.ElapsedMillis();

    // 阶段4: 分配合并后的输出tensor
    Timer alloc_output_timer;
    TensorShape merged_shape = reference_shape;
    merged_shape.set_dim(0, total_elements);
    Tensor* merged_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, merged_shape, &merged_tensor));
    stats.alloc_output_time_ms = alloc_output_timer.ElapsedMillis();
    
    if (debug_enabled && stats.alloc_output_time_ms > 1.0) {
      LOG(WARNING) << "TensorConcat: Large allocation time detected: " 
                   << stats.alloc_output_time_ms << " ms for " 
                   << (stats.total_bytes / 1024.0 / 1024.0) << " MB";
    }

    // 如果没有数据需要复制，直接返回
    if (total_elements == 0) {
      if (debug_enabled) {
        LOG(INFO) << "TensorConcat: Empty output, skipping copy";
      }
      return;
    }

    // 阶段5: 执行数据复制
    Timer copy_timer;
    T* output_data = merged_tensor->flat<T>().data();
    TensorConcatWithOffsetsFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), input_data_ptrs, offsets_data, num_inputs_, row_size,
            output_data);
    stats.copy_time_ms = copy_timer.ElapsedMillis();
    
    // 记录总时间
    stats.total_time_ms = total_timer.ElapsedMillis();
    
    // 分析和输出诊断信息
    if (debug_enabled) {
      stats.AnalyzeChunkSizes(chunk_elements, element_size);
      stats.Print();
      
      // 如果发现性能问题，输出额外警告
      if (stats.total_time_ms > 5.0) {
        LOG(ERROR) << "TensorConcat: Performance issue detected! Total time: " 
                   << stats.total_time_ms << " ms";
        LOG(ERROR) << "  Please check the diagnostics above for optimization recommendations.";
      }
    }
  }

 private:
  int num_inputs_;
  int32 alignment_;
  bool use_alignment_;
};

// CPU 注册
#define REGISTER_CPU_KERNEL(type)                                                   \
  REGISTER_KERNEL_BUILDER(                                                          \
      Name("TensorConcatWithOffsets").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorConcatWithOffsetsOp<CPUDevice, type>)

REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
REGISTER_CPU_KERNEL(int32);
REGISTER_CPU_KERNEL(int64);

#undef REGISTER_CPU_KERNEL

// GPU 注册
#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("TensorConcatWithOffsets") \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("offsets"),    \
                          TensorConcatWithOffsetsOp<GPUDevice, type>)

REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
REGISTER_GPU_KERNEL(int32);
REGISTER_GPU_KERNEL(int64);

#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace functor
}  // namespace tensorflow
