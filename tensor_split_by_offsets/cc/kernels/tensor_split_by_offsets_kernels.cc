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

#include "tensor_split_by_offsets.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/threadpool.h"

#include <cstdlib>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

// 使用统一的配置常量
using namespace tensor_split_by_offsets_config;

/**
 * CPU二级Functor：处理单个输出的拆分操作
 * 
 * 与一级Functor的区别：
 * 1. 处理单个任务，而非批量任务
 * 2. 不需要context参数
 * 3. 不涉及并行化逻辑
 * 
 * 注意：CPU版本不需要stream参数，因为CPU使用线程池而非流
 */
template <typename T>
struct TensorSplitByOffsetsCPUSingleTaskFunctor {
  void operator()(const T* input_data,
                  int64_t start_row,
                  int64_t row_count,
                  int64_t row_size,
                  T* output_data) {
    // 早期退出条件检查
    if (row_count == 0 || row_size == 0) {
      return;
    }

    if (input_data == nullptr || output_data == nullptr) {
      LOG(ERROR) << "TensorSplitByOffset CPU: Null pointer detected in single task functor";
      return;
    }

    // 计算输入数据的起始位置
    const int64_t input_start_idx = start_row * row_size;
    const int64_t total_elements = row_count * row_size;

    // 执行高效的内存复制
    const T* src = input_data + input_start_idx;
    T* dst = output_data;
    const size_t copy_size = static_cast<size_t>(total_elements) * sizeof(T);

    // 使用标准库的优化内存复制函数
    std::memcpy(dst, src, copy_size);
  }
};

/**
 * CPU设备上的TensorSplitByOffset functor实现
 *
 * 主要特性：
 * 1. 批量处理多个输出，支持并行化
 * 2. 使用二级Functor处理单个任务
 * 3. 环境变量控制并行化行为
 * 
 * 参数使用：
 *   - context: 使用，用于获取线程池进行并行化
 *   - d: 未使用（仅为统一接口）
 */
template <typename T>
struct TensorSplitByOffsetsFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context,
                  const CPUDevice& d,
                  const std::vector<SplitTask>& tasks) {
    if (tasks.empty()) {
      return;
    }

    // 检查是否需要并行化
    const char* threshold_env = std::getenv("TF_SPLIT_PARALLEL_THRESHOLD");
    const int64_t parallel_threshold = threshold_env ? 
        std::strtoll(threshold_env, nullptr, 10) : kDefaultParallelThreshold;

    const char* max_parallelism_env = std::getenv("TF_SPLIT_MAX_PARALLELISM");
    const int max_parallelism = max_parallelism_env ? 
        std::atoi(max_parallelism_env) : kDefaultMaxParallelism;

    // 计算总数据量
    int64_t total_elements = 0;
    for (const auto& task : tasks) {
      total_elements += task.row_count * task.row_size;
    }

    const int num_tasks = static_cast<int>(tasks.size());

    // 使用二级Functor处理单个任务
    TensorSplitByOffsetsCPUSingleTaskFunctor<T> single_task_functor;

    // 如果数据量足够大且任务数量足够多，使用并行处理
    if (total_elements >= parallel_threshold && num_tasks >= kMinOutputsForParallel) {
      auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
      int num_threads = worker_threads->num_threads;
      
      if (max_parallelism > 0) {
        num_threads = std::min(num_threads, max_parallelism);
      }

      // 并行处理每个任务
      auto shard_fn = [&](int64_t start_idx, int64_t limit_idx) {
        for (int64_t idx = start_idx; idx < limit_idx; ++idx) {
          const SplitTask& task = tasks[idx];
          
          // 调用二级Functor处理单个任务
          single_task_functor(
              static_cast<const T*>(task.input_data),
              task.start_row,
              task.row_count,
              task.row_size,
              static_cast<T*>(task.output_data));
        }
      };

      // 动态计算 cost_per_unit
      // 平均每个任务的元素数量，乘以每个元素的估算处理成本
      // 对于内存复制操作，每个元素的成本约为 10-100 个CPU周期
      const int64_t avg_elements_per_task = total_elements / num_tasks;
      const int64_t cost_per_element = 50;  // 每个元素的估算成本
      const int64_t cost_per_unit = std::max(static_cast<int64_t>(10000), 
                                               avg_elements_per_task * cost_per_element);

      worker_threads->workers->ParallelFor(
          num_tasks,
          cost_per_unit,
          shard_fn);
    } else {
      // 小数据量或任务少，使用单线程处理
      for (const auto& task : tasks) {
        // 调用二级Functor处理单个任务
        single_task_functor(
            static_cast<const T*>(task.input_data),
            task.start_row,
            task.row_count,
            task.row_size,
            static_cast<T*>(task.output_data));
      }
    }
  }
};

template <typename Device, typename T>
class TensorSplitByOffsetsOp : public OpKernel {
 public:
  explicit TensorSplitByOffsetsOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("N", &num_outputs_));
    OP_REQUIRES_OK(context, context->GetAttr("use_alignment", &use_alignment_));
    OP_REQUIRES_OK(context, context->GetAttr("alignment", &alignment_));

    // 验证alignment是2的幂
    OP_REQUIRES(
        context, alignment_ > 0 && (alignment_ & (alignment_ - 1)) == 0,
        errors::InvalidArgument("alignment must be a positive power of 2, got ", alignment_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& offsets_tensor = context->input(1);

    // 验证输入
    OP_REQUIRES(context, input_tensor.dims() >= 1,
                errors::InvalidArgument("Input tensor must have at least 1 dimension"));

    OP_REQUIRES(context, offsets_tensor.dims() == 2,
                errors::InvalidArgument("Offsets tensor must be 2-dimensional"));

    OP_REQUIRES(context, offsets_tensor.dim_size(1) == 2,
                errors::InvalidArgument("Offsets tensor must have shape [N, 2]"));

    // *** 关键验证：检查offsets的数据类型 ***
    OP_REQUIRES(context, offsets_tensor.dtype() == DT_INT64,
                errors::InvalidArgument("Offsets tensor must be int64 type, got ",
                                        DataTypeString(offsets_tensor.dtype())));

    // 验证offsets tensor的第一维与N属性一致
    OP_REQUIRES(context, offsets_tensor.dim_size(0) == num_outputs_,
                errors::InvalidArgument("Offsets tensor first dimension must equal N, got ",
                                        offsets_tensor.dim_size(0), " vs ", num_outputs_));

    const int num_outputs = num_outputs_;

    const int64_t input_dim0_size = input_tensor.dim_size(0);
    const T* input_data = input_tensor.flat<T>().data();
    // TensorFlow的int64类型需要显式转换为int64_t
    const int64_t* offsets_data =
        reinterpret_cast<const int64_t*>(offsets_tensor.flat<int64>().data());

    // 计算每个"行"（第0维的一个slice）包含的元素数量
    const TensorShape& input_shape = input_tensor.shape();
    int64_t row_size = 1;
    for (int dim = 1; dim < input_shape.dims(); ++dim) {
      row_size *= input_shape.dim_size(dim);
    }

    // 收集需要复制数据的任务
    std::vector<SplitTask> copy_tasks;
    copy_tasks.reserve(num_outputs);

    // 为每个输出分配tensor，并决定是否需要复制
    for (int i = 0; i < num_outputs; ++i) {
      const int64_t start = offsets_data[i * 2];
      const int64_t length = offsets_data[i * 2 + 1];

      // *** 增强验证：检查offset值的合理性 ***
      OP_REQUIRES(context, start >= 0,
                  errors::InvalidArgument("Offset ", i, " has negative start: ", start));

      OP_REQUIRES(context, length >= 0,
                  errors::InvalidArgument("Offset ", i, " has negative length: ", length));

      OP_REQUIRES(context, start <= input_dim0_size,
                  errors::InvalidArgument("Offset ", i, " start ", start,
                                          " exceeds input dimension 0 size: ", input_dim0_size));

      OP_REQUIRES(context, start + length <= input_dim0_size,
                  errors::InvalidArgument("Offset ", i, " range [", start, ", ", start + length,
                                          "] exceeds input dimension 0 size: ", input_dim0_size));

      // 检查是否有整数溢出
      OP_REQUIRES(context, start + length >= start,
                  errors::InvalidArgument("Offset ", i, " integer overflow: start=", start,
                                          ", length=", length));

      // 创建输出tensor的形状
      TensorShape output_shape = input_shape;
      output_shape.set_dim(0, length);

      // 处理空切片的情况
      if (length == 0) {
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(i, output_shape, &output_tensor));
        continue;
      }

      const T* slice_start = input_data + start * row_size;
      if (use_alignment_ && IsAligned(slice_start, alignment_) &&
          CanUseSlice(input_tensor, start, length)) {
        // 零拷贝优化：直接使用输入tensor的切片
        Tensor sliced_tensor = input_tensor.Slice(start, start + length);
        context->set_output(i, sliced_tensor);
      } else {
        // 需要复制数据：分配输出tensor并添加到任务列表
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(i, output_shape, &output_tensor));

        // 添加到复制任务列表
        SplitTask task;
        task.output_index = i;
        task.input_data = static_cast<const void*>(input_data);
        task.start_row = start;
        task.row_count = length;
        task.row_size = row_size;
        task.output_data = static_cast<void*>(output_tensor->flat<T>().data());
        copy_tasks.push_back(task);
      }
    }

    // 批量执行所有复制任务
    if (!copy_tasks.empty()) {
      TensorSplitByOffsetsFunctor<Device, T> functor;
      functor(context, context->eigen_device<Device>(), copy_tasks);
    }
  }

 private:
  /**
   * 检查内存指针是否满足对齐要求
   *
   * @param ptr 要检查的内存指针
   * @param alignment 对齐要求（字节数）
   * @return true如果指针满足指定的对齐条件
   */
  bool IsAligned(const void* ptr, int32 alignment) const {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
  }

  /**
   * 判断是否可以使用零拷贝切片优化
   *
   * 启发式规则：
   * 1. 切片长度足够大（避免小切片的开销）
   * 2. 内存布局适合零拷贝操作
   *
   * @param tensor 源tensor
   * @param start 切片起始位置
   * @param length 切片长度
   * @return true如果适合使用零拷贝
   */
  bool CanUseSlice(const Tensor& tensor, int64_t start, int64_t length) const {
    // 零拷贝优化：当切片足够大时使用零拷贝

    const int64_t slice_elements = length * tensor.NumElements() / tensor.dim_size(0);
    return slice_elements > kZeroCopyThreshold;
  }

  int num_outputs_;
  bool use_alignment_;
  int32 alignment_;
};

// CPU 注册
#define REGISTER_CPU_KERNEL(type)                                                \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("TensorSplitByOffsets").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorSplitByOffsetsOp<CPUDevice, type>)

REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
REGISTER_CPU_KERNEL(int32);
REGISTER_CPU_KERNEL(int64);

#undef REGISTER_CPU_KERNEL

// GPU 注册
#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("TensorSplitByOffsets")   \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("offsets"),    \
                          TensorSplitByOffsetsOp<GPUDevice, type>)

REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
REGISTER_GPU_KERNEL(int32);
REGISTER_GPU_KERNEL(int64);

#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace functor
}  // namespace tensorflow
