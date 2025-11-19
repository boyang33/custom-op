# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TensorSplitByOffset op for splitting tensors along the first dimension."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
import tensorflow as tf

_tensor_split_by_offsets_ops = load_library.load_op_library(
        resource_loader.get_path_to_datafile('_tensor_split_by_offsets_ops.so'))

def tensor_split_by_offsets(input_tensor, offsets, use_alignment=True, alignment=64, name=None):
    """将tensor沿第0维按指定偏移量拆分为多个tensor。
    
    此操作将输入tensor沿第0维基于提供的偏移量信息拆分为多个输出tensor。
    每个偏移量指定 [start, length] 用于输入tensor的一个切片。
    
    输入tensor可以有任意维数（≥1维）。拆分仅沿第0维进行，其他维度保持不变。
    
    特性：
        - 支持多维tensor处理
        - CPU版本支持零拷贝优化（可控制）
        - GPU版本采用高效的行并行处理
        - 完整的边界检查和错误处理
        - GPU版本使用HostMemory("offsets")确保offsets在CPU上可访问
    
    Args:
        input_tensor: 至少1维的输入 `Tensor`
        offsets: 形状为 [N, 2] 的2维 `Tensor`，每行包含 [start, length]。
            每对值指定沿第0维的切片起始位置和长度。
        use_alignment: 布尔值，是否启用对齐优化。默认为 True。
                    当为 True 时，算子会尝试使用零拷贝优化以提高性能和减少内存使用。
                    当为 False 时，强制使用内存复制，确保输出tensor与输入tensor完全独立。
        alignment: 整数，内存对齐要求（字节数）。默认为 64。
                用于零拷贝优化时检查内存指针是否满足对齐条件。
                必须是2的幂（如1, 2, 4, 8, 16, 32, 64, 128等）。
                较大的对齐值可能提高性能，但会降低零拷贝的成功率。
        name: 操作名称（可选）
        
    Returns:
        `Tensor` 对象列表，每个对应输入tensor沿第0维的一个切片。
        每个输出tensor的形状与输入tensor相同，除了第0维。
        
    Raises:
        ValueError: 如果input_tensor维度小于1或offsets格式无效
        
    Example:
        ```python
        import tensorflow as tf
        
        # 创建输入tensor [6, 3]
        input_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], 
                                [10, 11, 12], [13, 14, 15], [16, 17, 18]])
        
        # 定义偏移量：第一个输出取前2行，第二个输出取接下来3行，第三个输出取最后1行
        offsets = tf.constant([[0, 2], [2, 3], [5, 1]], dtype=tf.int32)
        
        # 执行拆分（使用默认的零拷贝优化）
        outputs = split_by_offset(input_tensor, offsets)
        
        # 或者显式控制零拷贝行为和内存对齐
        outputs_aligned = tensor_split_by_offsets(input_tensor, offsets, use_alignment=True, alignment=64)
        outputs_copy = tensor_split_by_offsets(input_tensor, offsets, use_alignment=False)
        outputs_custom_align = tensor_split_by_offsets(input_tensor, offsets, alignment=32)
        
        # outputs[0]: [[1, 2, 3], [4, 5, 6]]           # shape: [2, 3]
        # outputs[1]: [[7, 8, 9], [10, 11, 12], [13, 14, 15]]  # shape: [3, 3]  
        # outputs[2]: [[16, 17, 18]]                   # shape: [1, 3]
        ```
    """
    if not isinstance(input_tensor, ops.Tensor):
        input_tensor = ops.convert_to_tensor(input_tensor)
    
    if not isinstance(offsets, ops.Tensor):
        offsets = ops.convert_to_tensor(offsets, dtype=ops.dtypes.int64)
    else:
        # 确保offsets是int64类型
        if offsets.dtype != ops.dtypes.int64:
            offsets = tf.cast(offsets, ops.dtypes.int64)
    
    # === 输入验证 ===
    if input_tensor.shape.ndims is not None and input_tensor.shape.ndims < 1:
        raise ValueError(f"输入tensor必须至少有1个维度，得到形状 {input_tensor.shape}")
    
    if offsets.shape.ndims is not None and offsets.shape.ndims != 2:
        raise ValueError(f"偏移量必须是2维tensor，得到形状 {offsets.shape}")
    
    if (offsets.shape.dims is not None and 
            offsets.shape.dims[1].value is not None and 
            offsets.shape.dims[1].value != 2):
        raise ValueError(f"偏移量必须具有形状 [N, 2]，得到形状 {offsets.shape}")
    
    # 验证alignment参数
    if not isinstance(alignment, int) or alignment <= 0:
        raise ValueError(f"alignment必须是正整数，得到 {alignment}")
    
    if alignment & (alignment - 1) != 0:
        raise ValueError(f"alignment必须是2的幂，得到 {alignment}")
    
    # === 自动计算N值 ===
    # 获取offsets的第一维作为N值
    if (hasattr(offsets, 'shape') and 
            offsets.shape.dims is not None and 
            offsets.shape.dims[0].value is not None):
        # 静态形状已知
        N = int(offsets.shape.dims[0].value)
    else:
        # 动态形状，需要在运行时确定
        N = tf.shape(offsets)[0]
    
    return _tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets, N=N, use_alignment=use_alignment, alignment=alignment, name=name)


# 梯度定义
@ops.RegisterGradient("TensorSplitByOffsets")
def _split_by_offset_grad(op, *grad_outputs):
    """TensorSplitByOffset算子的梯度计算函数
    
    梯度计算原理：
        TensorSplitByOffset将输入tensor沿第0维拆分为多个输出tensor，
        反向传播时需要将各个输出的梯度按原始位置合并回输入tensor。
        这实质上是TensorMergeWithOffsets的逆操作。
    
    算法步骤：
        1. 过滤掉None梯度（对应没有梯度的输出）
        2. 创建与输入tensor相同形状的零梯度tensor
        3. 将各个输出梯度按offsets指定的位置写回输入梯度tensor
        4. 使用tf.py_function处理运行时动态的offsets信息
    
    Args:
        op: 前向传播的TensorSplitByOffset操作实例
        *grad_outputs: 来自后续层的梯度列表，对应每个输出tensor的梯度
        
    Returns:
        [input_grad, None]: 
            - input_grad: 对输入tensor的梯度
            - None: offsets不需要梯度（它们是索引，不是可训练参数）
    """
    # 获取原始输入
    input_tensor = op.inputs[0]
    offsets = op.inputs[1]
    
    # 过滤掉None梯度，准备合并
    valid_grads = []
    valid_offsets = []
    
    for i, grad_output in enumerate(grad_outputs):
        if grad_output is not None:
            valid_grads.append(grad_output)
            # 提取对应的offset信息 [start, length]
            valid_offsets.append(offsets[i])
    
    if not valid_grads:
        # 如果没有有效梯度，返回零梯度
        return [tf.zeros_like(input_tensor), None]
    
    # 使用tf.concat将所有有效的梯度tensor合并
    # 这相当于执行TensorMergeWithOffsets操作
    merged_grad = tf.concat(valid_grads, axis=0)
    
    # 创建与输入tensor相同形状的零tensor
    input_grad = tf.zeros_like(input_tensor)
    
    # 使用动态更新将合并的梯度放回正确位置
    # 由于offsets是运行时确定的，我们需要使用tf.py_function或其他动态方法
    
    def update_grad(input_grad_tensor, merged_grad_tensor, offsets_tensor):
        """动态更新梯度的辅助函数"""
        import numpy as np
        
        # 转换为numpy数组进行操作
        input_grad_np = input_grad_tensor.numpy()
        merged_grad_np = merged_grad_tensor.numpy()
        offsets_np = offsets_tensor.numpy()
        
        result = input_grad_np.copy()
        current_pos = 0
        
        # 按offsets顺序将梯度写回正确位置
        for i in range(len(valid_grads)):
            start = int(offsets_np[i, 0])
            length = int(offsets_np[i, 1])
            
            if length > 0:
                # 将当前输出的梯度复制到输入梯度的对应位置
                end_pos = current_pos + length
                result[start:start+length] = merged_grad_np[current_pos:end_pos]
                current_pos = end_pos
        
        return result.astype(input_grad_np.dtype)
    
    # 使用tf.py_function执行numpy操作
    if valid_offsets:
        valid_offsets_tensor = tf.stack(valid_offsets)
        input_grad = tf.py_function(
                update_grad,
                [input_grad, merged_grad, valid_offsets_tensor],
                input_grad.dtype
        )
        input_grad.set_shape(input_tensor.shape)
    
    # 返回梯度：input_tensor的梯度，offsets不需要梯度
    return [input_grad, None]

# 保持向后兼容性
tensor_split_by_offsets_ops = _tensor_split_by_offsets_ops
