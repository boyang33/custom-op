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

#include "tensor_concat_with_offsets.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/ops_util.h"

namespace tensorflow {
namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

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
    // 收集输入信息并验证
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
    }

    // 分配offsets tensor并预计算offsets
    TensorShape offsets_shape({num_inputs_, 2});
    Tensor* offsets_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, offsets_shape, &offsets_tensor));
    // TensorFlow的int64类型需要转换为int64_t
    int64_t* offsets_data = reinterpret_cast<int64_t*>(offsets_tensor->flat<int64>().data());

    const int64_t element_size = sizeof(T);
    int64_t current_offset = 0;

    // 预计算所有offsets和总大小
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

    // 分配合并后的输出tensor
    TensorShape merged_shape = reference_shape;
    merged_shape.set_dim(0, total_elements);
    Tensor* merged_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, merged_shape, &merged_tensor));

    // 如果没有数据需要复制，直接返回
    if (total_elements == 0) {
      return;
    }

    // 执行数据复制
    T* output_data = merged_tensor->flat<T>().data();
    TensorConcatWithOffsetsFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), input_data_ptrs, offsets_data, num_inputs_, row_size,
            output_data);
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

}  // namespace functor
}  // namespace tensorflow
