#!/usr/bin/env python3
"""
TensorConcatWithOffsets算子使用示例

这个示例展示了如何使用tensor_concat_with_offsets算子来合并多个tensor，
并获得每个tensor在合并结果中的偏移量信息。

支持任意维度的tensor，只要除了第一个维度外，其他维度都相同。
"""

import tensorflow as tf
import numpy as np
from python.ops import tensor_concat_with_offsets_ops

def example_1d_tensors():
    """演示1维tensor的合并"""
    print("=== 1维tensor合并示例 ===")
    
    # 创建不同长度的1维tensor
    tensor1 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    tensor2 = tf.constant([4.0, 5.0], dtype=tf.float32)
    tensor3 = tf.constant([6.0, 7.0, 8.0, 9.0], dtype=tf.float32)
    
    print(f"输入tensor1: {tensor1}")
    print(f"输入tensor2: {tensor2}")
    print(f"输入tensor3: {tensor3}")
    
    # 使用算子合并
    input_tensors = [tensor1, tensor2, tensor3]
    merged_tensor, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(input_tensors, use_alignment=False)
    
    print(f"\n合并后的tensor: {merged_tensor}")
    print(f"偏移量信息: {offsets}")
    
    # 验证可以使用偏移量信息恢复原始tensor
    print("\n验证恢复原始tensor:")
    for i, (start, length) in enumerate(offsets):
        restored = tf.slice(merged_tensor, [start], [length])
        original = input_tensors[i]
        print(f"原始tensor{i+1}: {original}")
        print(f"恢复tensor{i+1}: {restored}")
        print(f"是否一致: {tf.reduce_all(tf.equal(original, restored))}")

def example_2d_tensors():
    """演示2维tensor的合并"""
    print("\n\n=== 2维tensor合并示例 ===")
    
    # 创建不同第一维大小的2维tensor
    tensor1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)  # shape: [2, 3]
    tensor2 = tf.constant([[7, 8, 9]], dtype=tf.int32)             # shape: [1, 3]
    tensor3 = tf.constant([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=tf.int32)  # shape: [3, 3]
    
    print(f"输入tensor1 shape: {tensor1.shape}")
    print(f"tensor1:\n{tensor1}")
    print(f"\n输入tensor2 shape: {tensor2.shape}")
    print(f"tensor2:\n{tensor2}")
    print(f"\n输入tensor3 shape: {tensor3.shape}")
    print(f"tensor3:\n{tensor3}")
    
    # 使用算子合并
    input_tensors = [tensor1, tensor2, tensor3]
    merged_tensor, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(input_tensors, use_alignment=False)
    
    print(f"\n合并后的tensor shape: {merged_tensor.shape}")
    print(f"合并后的tensor:\n{merged_tensor}")
    print(f"偏移量信息: {offsets}")
    
    # 验证可以使用偏移量信息恢复原始tensor
    print("\n验证恢复原始tensor:")
    for i, (start, length) in enumerate(offsets):
        # 对于多维tensor，需要在所有其他维度上切片完整范围
        slice_begin = [start] + [0] * (len(merged_tensor.shape) - 1)
        slice_size = [length] + [-1] * (len(merged_tensor.shape) - 1)
        restored = tf.slice(merged_tensor, slice_begin, slice_size)
        original = input_tensors[i]
        print(f"\n原始tensor{i+1} shape: {original.shape}")
        print(f"恢复tensor{i+1} shape: {restored.shape}")
        print(f"是否一致: {tf.reduce_all(tf.equal(original, restored))}")

def example_3d_tensors():
    """演示3维tensor的合并"""
    print("\n\n=== 3维tensor合并示例 ===")
    
    # 创建不同第一维大小的3维tensor
    tensor1 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)  # shape: [2, 2, 2]
    tensor2 = tf.constant([[[9, 10], [11, 12]]], dtype=tf.float32)                  # shape: [1, 2, 2]
    tensor3 = tf.constant([[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]], dtype=tf.float32)  # shape: [3, 2, 2]
    
    print(f"输入tensor1 shape: {tensor1.shape}")
    print(f"输入tensor2 shape: {tensor2.shape}")
    print(f"输入tensor3 shape: {tensor3.shape}")
    
    # 使用算子合并
    input_tensors = [tensor1, tensor2, tensor3]
    merged_tensor, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(input_tensors, use_alignment=False)
    
    print(f"\n合并后的tensor shape: {merged_tensor.shape}")
    print(f"偏移量信息: {offsets}")
    
    # 验证可以使用偏移量信息恢复原始tensor
    print("\n验证恢复原始tensor:")
    for i, (start, length) in enumerate(offsets):
        slice_begin = [start, 0, 0]
        slice_size = [length, -1, -1]
        restored = tf.slice(merged_tensor, slice_begin, slice_size)
        original = input_tensors[i]
        print(f"原始tensor{i+1} shape: {original.shape}")
        print(f"恢复tensor{i+1} shape: {restored.shape}")
        print(f"是否一致: {tf.reduce_all(tf.equal(original, restored))}")

def example_4d_tensors():
    """演示4维tensor的合并"""
    print("\n\n=== 4维tensor合并示例 ===")
    
    # 创建不同第一维大小的4维tensor
    np.random.seed(42)  # 为了可重现性
    tensor1 = tf.constant(np.random.rand(2, 3, 4, 2).astype(np.float32))  # shape: [2, 3, 4, 2]
    tensor2 = tf.constant(np.random.rand(1, 3, 4, 2).astype(np.float32))  # shape: [1, 3, 4, 2]
    tensor3 = tf.constant(np.random.rand(3, 3, 4, 2).astype(np.float32))  # shape: [3, 3, 4, 2]
    
    print(f"输入tensor1 shape: {tensor1.shape}")
    print(f"输入tensor2 shape: {tensor2.shape}")
    print(f"输入tensor3 shape: {tensor3.shape}")
    
    # 使用算子合并
    input_tensors = [tensor1, tensor2, tensor3]
    merged_tensor, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(input_tensors, use_alignment=False)
    
    print(f"\n合并后的tensor shape: {merged_tensor.shape}")
    print(f"偏移量信息: {offsets}")
    
    # 验证可以使用偏移量信息恢复原始tensor
    print("\n验证恢复原始tensor:")
    for i, (start, length) in enumerate(offsets):
        slice_begin = [start, 0, 0, 0]
        slice_size = [length, -1, -1, -1]
        restored = tf.slice(merged_tensor, slice_begin, slice_size)
        original = input_tensors[i]
        print(f"原始tensor{i+1} shape: {original.shape}")
        print(f"恢复tensor{i+1} shape: {restored.shape}")
        
        # 对于浮点数，使用近似比较
        mse = tf.reduce_mean(tf.square(original - restored))
        print(f"均方误差 (应该接近0): {mse}")

def example_empty_tensors():
    """演示包含空tensor的合并"""
    print("\n\n=== 包含空tensor的合并示例 ===")
    
    # 创建包含空tensor的输入
    tensor1 = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)              # shape: [2, 2]
    tensor2 = tf.constant(np.empty([0, 2], dtype=np.int32))              # shape: [0, 2] - 空tensor
    tensor3 = tf.constant([[5, 6]], dtype=tf.int32)                      # shape: [1, 2]
    
    print(f"输入tensor1 shape: {tensor1.shape}")
    print(f"tensor1:\n{tensor1}")
    print(f"\n输入tensor2 shape: {tensor2.shape} (空tensor)")
    print(f"\n输入tensor3 shape: {tensor3.shape}")
    print(f"tensor3:\n{tensor3}")
    
    # 使用算子合并
    input_tensors = [tensor1, tensor2, tensor3]
    merged_tensor, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(input_tensors, use_alignment=False)
    
    print(f"\n合并后的tensor shape: {merged_tensor.shape}")
    print(f"合并后的tensor:\n{merged_tensor}")
    print(f"偏移量信息: {offsets}")
    
    # 验证空tensor的偏移量信息
    print("\n验证空tensor处理:")
    for i, (start, length) in enumerate(offsets):
        print(f"tensor{i+1}: start={start}, length={length}")
        if length == 0:
            print(f"  -> 这是空tensor")
        else:
            slice_begin = [start, 0]
            slice_size = [length, -1]
            restored = tf.slice(merged_tensor, slice_begin, slice_size)
            print(f"  -> 恢复的tensor shape: {restored.shape}")

def example_performance_comparison():
    """性能比较示例"""
    print("\n\n=== 性能比较示例 ===")
    
    # 创建较大的tensor进行性能测试
    print("创建较大的tensor进行性能测试...")
    
    tensors = []
    for i in range(10):
        size = np.random.randint(100, 1000)
        tensor = tf.random.normal([size, 128], dtype=tf.float32)
        tensors.append(tensor)
    
    print(f"创建了{len(tensors)}个tensor，第二维度都是128")
    
    # 使用自定义算子
    print("\n使用tensor_concat_with_offsets算子:")
    import time
    start_time = time.time()
    merged_tensor, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(tensors, use_alignment=False)
    custom_time = time.time() - start_time
    print(f"合并后tensor shape: {merged_tensor.shape}")
    print(f"用时: {custom_time:.4f}秒")
    
    # 使用TensorFlow内置函数比较
    print("\n使用tf.concat比较:")
    start_time = time.time()
    concat_result = tf.concat(tensors, axis=0)
    tf_time = time.time() - start_time
    print(f"concat结果shape: {concat_result.shape}")
    print(f"用时: {tf_time:.4f}秒")
    
    # 验证结果一致性
    print(f"\n结果是否一致: {tf.reduce_all(tf.equal(merged_tensor, concat_result))}")

if __name__ == "__main__":
    print("TensorConcatWithOffsets算子多维度使用示例")
    print("=" * 60)
    
    # 运行各种示例
    example_1d_tensors()
    example_2d_tensors()
    example_3d_tensors()
    example_4d_tensors()
    example_empty_tensors()
    example_performance_comparison()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("\n注意事项:")
    print("1. 所有输入tensor必须具有相同的维度数")
    print("2. 除第一个维度外，其他维度的大小必须相同")
    print("3. 算子支持float32、int32、int64等数据类型")
    print("4. 空tensor（第一维为0）也可以正常处理")
    print("5. 可以使用偏移量信息完美恢复原始tensor")