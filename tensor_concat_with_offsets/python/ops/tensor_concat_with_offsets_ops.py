# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TensorFlow TensorConcatWithOffsets 算子的Python接口

这个模块提供了内存对齐优化的tensor合并功能，专门设计用于与
TensorSegmentByOffsets配合实现高性能的零拷贝优化。
"""

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops

# 加载自定义算子库
_tensor_concat_with_offsets_so = tf.load_op_library(
    resource_loader.get_path_to_datafile("_tensor_concat_with_offsets_ops.so"))


def tensor_concat_with_offsets(inputs, alignment=64, use_alignment=True, use_pinned_memory=False, name=None):
    """将多个tensor合并为一个大tensor，并生成内存对齐优化的偏移量数组。

    支持多维tensor输入，所有输入tensor必须具有相同的rank，且除第0维外所有维度大小必须相同。
    合并操作沿第0维进行，类似于tf.concat(tensors, axis=0)但提供内存对齐优化。

    Args:
        inputs: tensor列表，所有tensor除第0维外其他维度必须相同
        alignment: 内存对齐字节数，默认64字节。仅在use_alignment=True时生效
        use_alignment: 是否启用内存对齐优化，默认True
        use_pinned_memory: 是否使用pinned memory分配输出，默认False。
                          当输出需要立即传输到GPU时，使用pinned memory可提升传输速度
        name: 操作名称（可选）

    Returns:
        tuple: (merged_tensor, offsets_array)
        - merged_tensor: 合并后的tensor
        - offsets_array: 偏移量数组，shape为[N, 2]，格式为[[start0, length0], [start1, length1], ...]

    Examples:
        # 1维tensor示例
        t1 = tf.constant([1, 2, 3])
        t2 = tf.constant([4, 5])
        merged, offsets = tensor_concat_with_offsets([t1, t2])
        # merged: [1, 2, 3, 4, 5] (在对齐模式下可能包含padding)
        # offsets: [[0, 3], [4, 2]] (假设有对齐padding)

        # 多维tensor示例
        t1 = tf.constant([[1, 2], [3, 4]])  # shape: [2, 2]
        t2 = tf.constant([[5, 6]])          # shape: [1, 2]  
        merged, offsets = tensor_concat_with_offsets([t1, t2])
        # merged: [[1, 2], [3, 4], [5, 6]]  # shape: [3, 2]
        # offsets: [[0, 2], [2, 1]]
        
        # pinned memory示例（用于CPU->GPU传输优化）
        merged, offsets = tensor_concat_with_offsets([t1, t2], use_pinned_memory=True)
        with tf.device('/gpu:0'):
            gpu_tensor = tf.identity(merged)  # 高速传输到GPU
    """

    # 输入验证
    if not inputs:
        raise ValueError("inputs cannot be empty")

    # 确保所有输入都是tensor
    inputs = [tf.convert_to_tensor(inp) for inp in inputs]

    # 验证数据类型一致性
    first_dtype = inputs[0].dtype
    for i, inp in enumerate(inputs):
        if inp.dtype != first_dtype:
            raise ValueError(f"All inputs must have the same dtype. "
                           f"Input 0 has dtype {first_dtype}, but input {i} has dtype {inp.dtype}")

    # 验证对齐参数
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")

    if alignment > 4096:
        raise ValueError(f"alignment too large (max 4096), got {alignment}")

    # 验证use_alignment参数
    if not isinstance(use_alignment, bool):
        raise ValueError(f"use_alignment must be a boolean, got {type(use_alignment)}")
        
    # 验证use_pinned_memory参数
    if not isinstance(use_pinned_memory, bool):
        raise ValueError(f"use_pinned_memory must be a boolean, got {type(use_pinned_memory)}")

    # 检查所有tensor是否至少为1维，并验证维度兼容性
    for i, tensor in enumerate(inputs):
        # 安全地获取ndims，处理动态形状的情况
        ndims = tensor.shape.ndims
        if ndims is not None and ndims < 1:
            raise ValueError(f"Input tensor {i} must have at least 1 dimension, got shape {tensor.shape}")

        # 验证除第一维外，其他维度都相同
        if i > 0:
            first_ndims = inputs[0].shape.ndims
            # 只在形状已知时进行静态检查
            if ndims is not None and first_ndims is not None:
                if ndims != first_ndims:
                    raise ValueError(f"All input tensors must have the same rank. "
                                    f"First tensor has rank {first_ndims}, "
                                    f"but tensor {i} has rank {ndims}")

                # 检查其他维度
                for dim in range(1, ndims):
                    dim_size = tensor.shape[dim]
                    first_dim_size = inputs[0].shape[dim]
                    # 只在维度大小已知时检查
                    if dim_size is not None and first_dim_size is not None:
                        if dim_size != first_dim_size:
                            raise ValueError(f"All input tensors must have the same shape except for dimension 0. "
                                            f"Dimension {dim} mismatch: tensor 0 has size {first_dim_size}, "
                                            f"but tensor {i} has size {dim_size}")

    # 执行算子操作
    return _tensor_concat_with_offsets_so.tensor_concat_with_offsets(
        inputs, alignment=alignment, use_alignment=use_alignment, use_pinned_memory=use_pinned_memory)


@tf.RegisterGradient("TensorConcatWithOffsets")
def _tensor_concat_with_offsets_grad(op, grad_merged, grad_offsets):
    """Gradient function for TensorConcatWithOffsets."""
    if grad_merged is None:
        return [None] * op.get_attr("N")

    offsets = op.outputs[1]
    num_inputs = op.get_attr("N")

    # 使用TensorSplitByOffsets来分配梯度
    from tensor_split_by_offsets.python.ops import tensor_split_by_offsets_ops

    input_grads = tensor_split_by_offsets_ops.tensor_split_by_offsets(
        grad_merged, offsets, use_alignment=True)

    return input_grads
