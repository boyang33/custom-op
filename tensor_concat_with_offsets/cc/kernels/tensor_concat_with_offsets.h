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

namespace tensorflow {
namespace functor {

/**
 * TensorConcatWithOffsets functor的CPU实现
 *
 * 根据预计算的offsets执行高效的内存复制操作
 */
template <typename Device, typename T>
struct TensorConcatWithOffsetsFunctor {
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
