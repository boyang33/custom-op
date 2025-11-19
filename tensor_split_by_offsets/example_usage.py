#!/usr/bin/env python3
"""
TensorSplitByOffsets算子使用示例

这个示例展示了如何使用tensor_split_by_offsets算子根据偏移量信息
将一个大tensor拆分成多个小tensor。

支持任意维度的tensor，拆分只在第一个维度上进行。
"""

import tensorflow as tf
import numpy as np
import time
from python.ops import tensor_split_by_offsets_ops

def verify_with_tf_concat(output_tensors, expected_merged=None):
        """
        使用tf.concat进行交叉验证，确保拆分结果可以正确重新合并
        
        Args:
                output_tensors: tensor_split_by_offsets的输出结果
                expected_merged: 期望的合并结果（可选）
        
        Returns:
                bool: 验证是否通过
        """
        # 使用tf.concat重新合并
        reconstructed = tf.concat(output_tensors, axis=0)
        
        if expected_merged is not None:
                # 与期望结果比较
                is_match = tf.reduce_all(tf.equal(expected_merged, reconstructed))
                return is_match
        else:
                # 返回重新合并的结果供外部比较
                return reconstructed

def example_1d_tensor():
        """演示1维tensor的拆分"""
        print("=== 1维tensor拆分示例 ===")
        
        # 创建输入tensor和偏移量信息
        input_tensor = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.int32)
        offsets = tf.constant([[0, 3], [3, 2], [5, 4]], dtype=tf.int32)
        
        print(f"输入tensor: {input_tensor}")
        print(f"偏移量信息: {offsets}")
        print("偏移量含义: [[start1, length1], [start2, length2], [start3, length3]]")
        
        # 使用算子拆分
        output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分结果:")
        for i, tensor in enumerate(output_tensors):
                print(f"输出tensor{i+1}: {tensor}")
        
        # 验证拆分结果
        print("验证拆分结果:")
        expected_results = [
                tf.constant([1, 2, 3], dtype=tf.int32),      # offset=0, length=3
                tf.constant([4, 5], dtype=tf.int32),         # offset=3, length=2
                tf.constant([6, 7, 8, 9], dtype=tf.int32)    # offset=5, length=4
        ]
        
        for i, (expected, actual) in enumerate(zip(expected_results, output_tensors)):
                is_equal = tf.reduce_all(tf.equal(expected, actual))
                print(f"tensor{i+1}验证: {is_equal}")
        
        # 使用tf.concat进行交叉验证
        print("使用tf.concat交叉验证:")
        reconstructed = verify_with_tf_concat(output_tensors)
        is_identical = tf.reduce_all(tf.equal(input_tensor, reconstructed))
        print(f"重新合并是否与原始输入一致: {is_identical}")

def example_2d_tensor():
        """演示2维tensor的拆分"""
        print("=== 2维tensor拆分示例 ===")
        
        # 创建2维输入tensor（相当于合并后的tensor）
        input_tensor = tf.constant([
                [1, 2, 3], [4, 5, 6],           # 前2行 (来自tensor1)
                [7, 8, 9],                      # 第3行 (来自tensor2)
                [10, 11, 12], [13, 14, 15], [16, 17, 18]  # 后3行 (来自tensor3)
        ], dtype=tf.int32)  # shape: [6, 3]
        
        offsets = tf.constant([[0, 2], [2, 1], [3, 3]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"输入tensor:\n{input_tensor}")
        print(f"偏移量信息: {offsets}")
        
        # 使用算子拆分
        output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分结果:")
        for i, tensor in enumerate(output_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")
                print(f"内容:\n{tensor}")
                print()
        
        # 验证可以通过tf.concat重新合并  
        print("使用tf.concat交叉验证:")
        is_identical = verify_with_tf_concat(output_tensors, input_tensor)
        print(f"重新合并后是否与原tensor一致: {is_identical}")

def example_3d_tensor():
        """演示3维tensor的拆分"""
        print("=== 3维tensor拆分示例 ===")
        
        # 创建3维输入tensor
        input_tensor = tf.constant([
                [[1, 2], [3, 4]], [[5, 6], [7, 8]],           # 前2个 (来自tensor1)
                [[9, 10], [11, 12]],                           # 第3个 (来自tensor2)
                [[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]  # 后3个 (来自tensor3)
        ], dtype=tf.float32)  # shape: [6, 2, 2]
        
        offsets = tf.constant([[0, 2], [2, 1], [3, 3]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"偏移量信息: {offsets}")
        
        # 使用算子拆分
        output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分结果:")
        for i, tensor in enumerate(output_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")
                print(f"第一个元素: {tensor[0]}")
                print()
        
        # 验证重新合并
        print("使用tf.concat交叉验证:")
        reconstructed = verify_with_tf_concat(output_tensors)
        mse = tf.reduce_mean(tf.square(input_tensor - reconstructed))
        print(f"重新合并后的均方误差 (应该接近0): {mse}")

def example_4d_tensor():
        """演示4维tensor的拆分"""
        print("\n=== 4维tensor拆分示例 ===")
        
        # 创建4维输入tensor
        np.random.seed(42)
        input_data = np.random.rand(6, 3, 4, 2).astype(np.float32)
        input_tensor = tf.constant(input_data)
        
        offsets = tf.constant([[0, 2], [2, 1], [3, 3]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"偏移量信息: {offsets}")
        
        # 使用算子拆分
        output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分结果:")
        for i, tensor in enumerate(output_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")
                print(f"数据范围: [{tf.reduce_min(tensor):.3f}, {tf.reduce_max(tensor):.3f}]")
        
        # 验证重新合并
        reconstructed = tf.concat(output_tensors, axis=0)
        mse = tf.reduce_mean(tf.square(input_tensor - reconstructed))
        print(f"重新合并后的均方误差 (应该接近0): {mse}")

def example_empty_slice():
        """演示包含空切片的拆分"""
        print("\n=== 包含空切片的拆分示例 ===")
        
        # 创建输入tensor
        input_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int32)
        offsets = tf.constant([[0, 2], [2, 0], [2, 1]], dtype=tf.int32)  # 中间一个是空切片
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"输入tensor:\n{input_tensor}")
        print(f"偏移量信息: {offsets}")
        print("注意: 第二个切片长度为0，将产生空tensor")
        
        # 使用算子拆分
        output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分结果:")
        for i, tensor in enumerate(output_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")
                if tensor.shape[0] == 0:
                        print("这是一个空tensor")
                else:
                        print(f"内容:\n{tensor}")
                print()

def example_single_slice():
        """演示单个切片的情况"""
        print("\n=== 单个切片示例 ===")
        
        # 创建输入tensor
        input_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.int32)
        offsets = tf.constant([[0, 3]], dtype=tf.int32)  # 只有一个输出
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"输入tensor:\n{input_tensor}")
        print(f"偏移量信息: {offsets}")
        
        # 使用算子拆分
        output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分结果:")
        print(f"输出tensor数量: {len(output_tensors)}")
        print(f"输出tensor shape: {output_tensors[0].shape}")
        print(f"内容:\n{output_tensors[0]}")
        
        # 验证是否与原tensor相同
        is_identical = tf.reduce_all(tf.equal(input_tensor, output_tensors[0]))
        print(f"输出是否与输入相同: {is_identical}")

def example_roundtrip_test():
        """演示与tensor_merge_with_offsets的往返测试"""
        print("\n=== 往返测试示例 ===")
        
        # 创建原始tensor列表
        original_tensors = [
                tf.constant([[1, 2], [3, 4]], dtype=tf.float32),      # shape: [2, 2]
                tf.constant([[5, 6]], dtype=tf.float32),              # shape: [1, 2]
                tf.constant([[7, 8], [9, 10], [11, 12]], dtype=tf.float32)  # shape: [3, 2]
        ]
        
        print("原始tensor列表:")
        for i, tensor in enumerate(original_tensors):
                print(f"tensor{i+1} shape: {tensor.shape}")
                print(f"内容:\n{tensor}")
                print()
        
        # 步骤1: 使用tf.concat模拟merge操作
        merged_tensor = tf.concat(original_tensors, axis=0)
        
        # 计算offsets信息
        offsets_data = []
        current_offset = 0
        for tensor in original_tensors:
                length = tensor.shape[0]
                offsets_data.append([current_offset, length])
                current_offset += length
        offsets = tf.constant(offsets_data, dtype=tf.int32)
        
        print(f"合并后tensor shape: {merged_tensor.shape}")
        print(f"偏移量信息: {offsets}")
        
        # 步骤2: 使用tensor_split_by_offsets拆分
        restored_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(merged_tensor, offsets)
        
        print(f"拆分恢复结果:")
        for i, tensor in enumerate(restored_tensors):
                print(f"恢复tensor{i+1} shape: {tensor.shape}")
        
        # 步骤3: 验证往返一致性
        print(f"往返一致性验证:")
        for i, (original, restored) in enumerate(zip(original_tensors, restored_tensors)):
                mse = tf.reduce_mean(tf.square(original - restored))
                print(f"tensor{i+1} 均方误差: {mse} (应该接近0)")
                is_identical = tf.reduce_all(tf.equal(original, restored))
                print(f"tensor{i+1} 是否完全一致: {is_identical}")

def example_performance_test():
        """性能测试示例：自定义算子 vs tf.split vs tf.slice"""
        print("\n=== 性能测试示例：自定义算子 vs tf.split vs tf.slice ===")
        
        # 创建较大的tensor
        print("创建大tensor进行性能测试...")
        large_tensor = tf.random.normal([10000, 128], dtype=tf.float32)
        
        # 创建多个随机偏移量，确保覆盖整个tensor
        total_rows = int(large_tensor.shape[0])
        offsets_data = []
        sizes = []
        current_offset = 0
        while current_offset < total_rows:
                remaining = total_rows - current_offset
                if remaining <= 150:
                        length = remaining
                else:
                        length = np.random.randint(50, 151)
                offsets_data.append([current_offset, length])
                sizes.append(length)
                current_offset += length
        offsets = tf.constant(offsets_data, dtype=tf.int32)
        sizes_tensor = tf.constant(sizes, dtype=tf.int32)
        
        print(f"大tensor shape: {large_tensor.shape}")
        print(f"拆分成 {len(offsets_data)} 个子tensor")
        
        # 预热，避免首次调用干扰
        _ = tensor_split_by_offsets_ops.tensor_split_by_offsets(large_tensor, offsets)
        _ = tf.split(large_tensor, sizes_tensor, axis=0)
        _ = [tf.slice(large_tensor, [start, 0], [length, -1]) for start, length in offsets_data]
        
        def benchmark(fn, label):
                iterations = 20
                start_time = time.time()
                for _ in range(iterations):
                        result = fn()
                        # 强制执行，以避免延迟求值影响时间统计
                        if isinstance(result, list):
                                _ = [tensor.numpy() for tensor in result]
                        else:
                                _ = [tensor.numpy() for tensor in result]
                elapsed = (time.time() - start_time) / iterations
                print(f"{label} 平均用时: {elapsed * 1000:.3f} ms")
                return result, elapsed
        
        # 自定义算子拆分
        print("\n开始性能测试...")
        custom_result, custom_time = benchmark(
                lambda: tensor_split_by_offsets_ops.tensor_split_by_offsets(large_tensor, offsets),
                "自定义 tensor_split_by_offsets"
        )
        
        # tf.split 拆分（需要sizes）
        tf_split_result, tf_split_time = benchmark(
                lambda: tf.split(large_tensor, sizes_tensor, axis=0),
                "标准 tf.split"
        )
        
        # tf.slice 循环拆分（逐段切片）
        tf_slice_result, tf_slice_time = benchmark(
                lambda: [tf.slice(large_tensor, [start, 0], [length, -1]) for start, length in offsets_data],
                "标准 tf.slice 循环"
        )
        
        # 汇总
        print("\n=== 性能对比汇总 ===")
        print(f"自定义算子速度提升 (对比 tf.split): {(tf_split_time / custom_time - 1) * 100:.1f}%")
        print(f"自定义算子速度提升 (对比 tf.slice 循环): {(tf_slice_time / custom_time - 1) * 100:.1f}%")
        
        # 抽查结果正确性
        print("\n抽查结果正确性:")
        for index in [0, len(custom_result) // 2, len(custom_result) - 1]:
                start, length = offsets_data[index]
                expected = tf.slice(large_tensor, [start, 0], [length, -1])
                mse_custom = tf.reduce_mean(tf.square(expected - custom_result[index]))
                mse_tf_split = tf.reduce_mean(tf.square(expected - tf_split_result[index]))
                mse_tf_slice = tf.reduce_mean(tf.square(expected - tf_slice_result[index]))
                print(f"tensor{index}: mse(自定义)={mse_custom:.2e}, mse(tf.split)={mse_tf_split:.2e}, mse(tf.slice)={mse_tf_slice:.2e}")

def example_row_parallel_optimal():
        """触发按行并行策略的最佳性能示例"""
        print("\n=== 按行并行策略触发示例（最佳性能区间）===")
        
        # 创建满足最佳row_size条件的tensor：64-1024元素/行，≥64行
        print("测试用例1: row_size=128 (最佳性能区间)")
        row_size = 128  # 在64-1024最佳区间内
        total_rows = 200  # 远大于64行的最小要求
        
        # 创建输入tensor [200, 128]
        input_tensor = tf.random.normal([total_rows, row_size], dtype=tf.float32)
        
        # 创建多个输出，确保有足够的并行度
        offsets_data = [
                [0, 80],      # 前80行
                [80, 60],     # 中间60行  
                [140, 60]     # 后60行
        ]
        offsets = tf.constant(offsets_data, dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"row_size: {row_size} (在最佳区间64-1024内)")
        print(f"total_rows: {total_rows} (远大于最小要求64)")
        print(f"预期策略: 按行并行 (Row-Parallel)")
        print(f"偏移量信息: {offsets_data}")
        
        # 执行拆分
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分结果:")
        for i, tensor in enumerate(split_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")
        
        # 验证正确性
        print(f"正确性验证:")
        for i, (tensor, (start, length)) in enumerate(zip(split_tensors, offsets_data)):
                expected = tf.slice(input_tensor, [start, 0], [length, -1])
                mse = tf.reduce_mean(tf.square(expected - tensor))
                print(f"tensor{i+1} 均方误差: {mse} (应该接近0)")

def example_row_parallel_extended():
        """触发按行并行策略的扩展范围示例"""
        print("\n=== 按行并行策略触发示例（扩展范围）===")
        
        # 测试用例2: row_size=512，更大的tensor
        print("测试用例2: row_size=512 (仍在最佳区间)")
        row_size = 512
        total_rows = 300
        
        input_tensor = tf.random.normal([total_rows, row_size], dtype=tf.float32)
        
        # 创建更多输出来测试GPU并行度
        offsets_data = [
                [0, 50],      # 第1组：50行
                [50, 75],     # 第2组：75行
                [125, 100],   # 第3组：100行
                [225, 50],    # 第4组：50行
                [275, 25]     # 第5组：25行
        ]
        offsets = tf.constant(offsets_data, dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"row_size: {row_size} (在最佳区间64-1024内)")  
        print(f"total_rows: {total_rows}")
        print(f"输出数量: {len(offsets_data)} (测试分支分歧处理)")
        print(f"预期策略: 按行并行 (Row-Parallel)")
        
        # 执行拆分
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分结果:")
        for i, tensor in enumerate(split_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")

def example_row_parallel_boundary():
        """测试按行并行策略的边界条件"""
        print("\n=== 按行并行策略边界测试 ===")
        
        # 测试用例3: row_size=64 (最佳区间下边界)
        print("测试用例3: row_size=64 (最佳区间下边界)")
        row_size = 64
        total_rows = 100
        
        input_tensor = tf.random.normal([total_rows, row_size], dtype=tf.float32)
        offsets = tf.constant([[0, 40], [40, 35], [75, 25]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"row_size: {row_size} (最佳区间下边界)")
        print(f"预期策略: 按行并行")
        
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        for i, tensor in enumerate(split_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")
        
        # 测试用例4: row_size=1024 (最佳区间上边界)
        print("测试用例4: row_size=1024 (最佳区间上边界)")
        row_size = 1024
        total_rows = 80
        
        input_tensor = tf.random.normal([total_rows, row_size], dtype=tf.float32)
        offsets = tf.constant([[0, 30], [30, 25], [55, 25]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"row_size: {row_size} (最佳区间上边界)")
        print(f"预期策略: 按行并行")
        
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        for i, tensor in enumerate(split_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")

def example_element_parallel_trigger():
        """触发按元素并行策略的示例（对比测试）"""
        print("\n=== 按元素并行策略触发示例（对比测试）===")
        
        # 测试用例5: row_size太小，应该触发元素并行
        print("测试用例5: row_size=16 (小于最佳区间，应触发元素并行)")
        row_size = 16
        total_rows = 100
        
        input_tensor = tf.random.normal([total_rows, row_size], dtype=tf.float32)
        offsets = tf.constant([[0, 40], [40, 35], [75, 25]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"row_size: {row_size} (小于最佳区间64)")
        print(f"预期策略: 按元素并行 (Element-Parallel)")
        
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        for i, tensor in enumerate(split_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")
        
        # 测试用例6: row_size太大，应该触发元素并行
        print("测试用例6: row_size=10000 (超过最大区间，应触发元素并行)")
        row_size = 10000
        total_rows = 80
        
        input_tensor = tf.random.normal([total_rows, row_size], dtype=tf.float32)
        offsets = tf.constant([[0, 30], [30, 25], [55, 25]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"row_size: {row_size} (超过最大区间8192)")
        print(f"预期策略: 按元素并行 (Element-Parallel)")
        
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        for i, tensor in enumerate(split_tensors):
                print(f"输出tensor{i+1} shape: {tensor.shape}")

def example_memory_alignment_test():
        """内存对齐优化测试"""
        print("\n=== 内存对齐优化测试 ===")
        
        # 测试用例7: 内存对齐的row_size (32的倍数，128字节对齐)
        print("测试用例7: row_size=128 (32的倍数，内存对齐优化)")
        row_size = 128  # 32 * sizeof(float) = 128字节对齐
        total_rows = 150
        
        input_tensor = tf.random.normal([total_rows, row_size], dtype=tf.float32)
        offsets = tf.constant([[0, 50], [50, 60], [110, 40]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"row_size: {row_size} (128字节对齐，应获得memory coalescing bonus)")
        print(f"预期策略: 按行并行 + 内存对齐优化")
        
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        
        print(f"拆分完成，输出数量: {len(split_tensors)}")
        
        # 验证内存对齐的性能优势（通过重复执行测量时间）
        import time
        num_iterations = 10
        start_time = time.time()
        for _ in range(num_iterations):
                _ = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
        aligned_time = (time.time() - start_time) / num_iterations
        
        # 对比：非对齐的row_size
        row_size_unaligned = 127  # 不是32的倍数
        input_tensor_unaligned = tf.random.normal([total_rows, row_size_unaligned], dtype=tf.float32)
        
        start_time = time.time()
        for _ in range(num_iterations):
                _ = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor_unaligned, offsets)
        unaligned_time = (time.time() - start_time) / num_iterations
        
        print(f"内存对齐版本平均用时: {aligned_time:.6f}秒")
        print(f"非对齐版本平均用时: {unaligned_time:.6f}秒")
        print(f"性能提升: {((unaligned_time - aligned_time) / unaligned_time * 100):.1f}%")

def example_custom_alignment():
        """演示自定义alignment参数的使用"""
        print("\n=== 自定义Alignment参数示例 ===")
        
        # 创建测试数据
        input_tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=tf.float32)
        offsets = tf.constant([[0, 2], [2, 1]], dtype=tf.int32)
        
        print(f"输入tensor shape: {input_tensor.shape}")
        print(f"输入tensor:\n{input_tensor}")
        print(f"偏移量信息: {offsets}")
        
        # 测试不同的alignment值
        alignment_values = [1, 4, 16, 64, 128]
        
        print(f"测试不同alignment值的效果:")
        for alignment in alignment_values:
                print(f"--- Alignment = {alignment} ---")
                
                # 测试启用零拷贝
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                        input_tensor, offsets, use_alignment=True, alignment=alignment)
                
                print(f"零拷贝模式 (alignment={alignment}):")
                for i, tensor in enumerate(output_tensors):
                        print(f"  输出tensor{i+1}: {tensor.numpy().tolist()}")
        
        # 演示alignment对大tensor零拷贝的影响
        print(f"--- 大Tensor的Alignment效果 ---")
        
        # 创建较大的tensor
        np.random.seed(123)
        large_data = np.random.rand(1000, 64).astype(np.float32)
        large_tensor = tf.constant(large_data)
        large_offsets = tf.constant([[0, 500], [500, 500]], dtype=tf.int32)
        
        print(f"大tensor shape: {large_tensor.shape}")
        
        # 比较不同alignment的性能
        import time
        alignments_to_test = [16, 32, 64, 128]
        
        for alignment in alignments_to_test:
                # 测量执行时间
                start_time = time.time()
                outputs = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                        large_tensor, large_offsets, use_alignment=True, alignment=alignment)
                # 强制执行（确保tensor被计算）
                _ = [tensor.numpy() for tensor in outputs]
                execution_time = time.time() - start_time
                
                print(f"Alignment {alignment:3d}: {execution_time:.6f}秒, 输出shapes: {[t.shape for t in outputs]}")
        
        print(f"注意: alignment值影响零拷贝优化的成功率")
        print(f"- 较小的alignment值 (如1, 4) 更容易满足对齐条件，零拷贝成功率更高")
        print(f"- 较大的alignment值 (如64, 128) 对内存对齐要求更严格，但可能带来更好的性能")

def example_alignment_validation():
        """演示alignment参数验证"""
        print("\n=== Alignment参数验证示例 ===")
        
        input_tensor = tf.constant([1, 2, 3, 4], dtype=tf.float32)
        offsets = tf.constant([[0, 2], [2, 2]], dtype=tf.int32)
        
        print("测试有效的alignment值:")
        valid_alignments = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
        for alignment in valid_alignments:
                try:
                        outputs = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                                input_tensor, offsets, alignment=alignment)
                        print(f"  alignment={alignment}: ✓ 有效")
                except ValueError as e:
                        print(f"  alignment={alignment}: ✗ 错误 - {e}")
        
        print(f"测试无效的alignment值:")
        invalid_alignments = [0, -1, 3, 5, 7, 9, 15, 31, 63, 65]
        
        for alignment in invalid_alignments:
                try:
                        outputs = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                                input_tensor, offsets, alignment=alignment)
                        print(f"  alignment={alignment}: ✗ 应该失败但没有失败")
                except ValueError as e:
                        print(f"  alignment={alignment}: ✓ 正确拒绝 - {str(e)[:50]}...")
        
        print(f"说明:")
        print(f"- alignment必须是正整数")
        print(f"- alignment必须是2的幂 (1, 2, 4, 8, 16, 32, 64, 128, ...)")
        print(f"- 常用值: 64 (默认), 32 (GPU优化), 128 (高性能)")

def example_alignment_zero_copy_comparison():
        """演示alignment对零拷贝行为的影响"""
        print("\n=== Alignment对零拷贝行为的影响 ===")
        
        # 创建大tensor以便观察零拷贝效果
        np.random.seed(789)
        test_data = np.random.rand(2000, 32).astype(np.float32)
        input_tensor = tf.constant(test_data)
        offsets = tf.constant([[0, 1000], [1000, 1000]], dtype=tf.int32)
        
        print(f"测试tensor shape: {input_tensor.shape}")
        print(f"偏移量信息: {offsets}")
        
        alignments = [16, 32, 64, 128]
        
        print(f"比较不同alignment下的零拷贝vs数据复制:")
        
        for alignment in alignments:
                print(f"--- Alignment = {alignment} ---")
                
                # 测量零拷贝模式
                import time
                start_time = time.time()
                zero_copy_outputs = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                        input_tensor, offsets, use_alignment=True, alignment=alignment)
                zero_copy_results = [tensor.numpy() for tensor in zero_copy_outputs]
                zero_copy_time = time.time() - start_time
                
                # 测量数据复制模式
                start_time = time.time()
                copy_outputs = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                        input_tensor, offsets, use_alignment=False, alignment=alignment)
                copy_results = [tensor.numpy() for tensor in copy_outputs]
                copy_time = time.time() - start_time
                
                # 验证结果一致性
                results_match = all(
                        np.allclose(zc, cp, rtol=1e-6) 
                        for zc, cp in zip(zero_copy_results, copy_results)
                )
                
                print(f"  零拷贝模式: {zero_copy_time:.6f}秒")
                print(f"  数据复制模式: {copy_time:.6f}秒")
                print(f"  性能提升: {((copy_time - zero_copy_time) / copy_time * 100):.1f}%")
                print(f"  结果一致性: {'✓' if results_match else '✗'}")
        
        print(f"总结:")
        print(f"- 零拷贝模式通常比数据复制模式更快")
        print(f"- alignment值影响零拷贝的成功率和性能")
        print(f"- 较小的alignment值更容易满足对齐条件")
        print(f"- 较大的alignment值在满足条件时可能有更好的性能")

if __name__ == "__main__":
        print("TensorSplitByOffsets算子多维度使用示例")
        print("=" * 60)
        
        # 运行基础示例
        print("=== 基础功能示例 ===")
        example_1d_tensor()
        example_2d_tensor()
        example_3d_tensor()
        example_4d_tensor()
        example_empty_slice()
        example_single_slice()
        example_roundtrip_test()
        example_performance_test()
        
        # 运行alignment相关示例
        print("\n" + "=" * 80)
        print("=== 自定义Alignment参数示例 ===")
        print("演示如何使用alignment参数优化零拷贝行为")
        print("=" * 80)
        
        example_custom_alignment()
        example_alignment_validation()
        example_alignment_zero_copy_comparison()
        
        # 运行GPU策略测试示例
        print("\n" + "=" * 80)
        print("=== GPU并行策略测试示例 ===")
        print("根据row_size自动选择最优GPU kernel策略")
        print("=" * 80)
        
        example_row_parallel_optimal()
        example_row_parallel_extended()
        example_row_parallel_boundary()
        example_element_parallel_trigger()
        example_memory_alignment_test()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("\n基础功能注意事项:")
        print("1. 算子支持任意维度的tensor，拆分只在第一个维度上进行")
        print("2. offsets格式: [[start1, length1], [start2, length2], ...]")
        print("3. 支持空切片（length=0）")
        print("4. 偏移量不能越界，start和length必须有效")
        print("5. 可以与tensor_merge_with_offsets完美配合实现往返操作")
        print("6. 算子支持float32、int32、int64等数据类型")
        print("7. GPU加速支持，适合处理大tensor")
        print("8. 现支持自定义alignment参数，优化零拷贝性能")
        
        print("\nGPU并行策略优化说明:")
        print("1. 按行并行 (Row-Parallel):")
        print("   - 最佳row_size: 64-1024元素")
        print("   - 要求: 总行数≥64，有足够GPU并行度")
        print("   - 优势: 更好的内存访问模式，减少线程分歧")
        print("   - 适用: 中等大小的行，规整的数据结构")
        
        print("2. 按元素并行 (Element-Parallel):")
        print("   - 触发条件: row_size<32 或 >8192，行数不足")
        print("   - 优势: 更细粒度的并行，适应不规则数据")
        print("   - 适用: 小行、超大行、或行数较少的情况")
        
        print("3. 内存对齐优化:")
        print("   - 最佳对齐: row_size为32的倍数 (128字节对齐)")
        print("   - 性能提升: 利用GPU memory coalescing")
        print("   - 建议: 在可能的情况下设计对齐的数据结构")
        
        print("\n4. Alignment参数优化:")
        print("   - alignment参数控制内存对齐要求，默认值为64字节")
        print("   - 必须是2的幂 (1, 2, 4, 8, 16, 32, 64, 128...)")  
        print("   - 较大的alignment值可能提高性能，但降低零拷贝成功率")
        print("   - 较小的alignment值更容易满足对齐条件")
        print("   - 建议根据具体应用场景调参优化")
        
        print("\n5. tf.concat交叉验证:")
        print("   - 本示例已在多个函数中使用tf.concat进行交叉验证")
        print("   - verify_with_tf_concat函数确保拆分结果可以正确重新合并")
        print("   - 这提供了额外的正确性保证和与标准TensorFlow算子的兼容性验证")
