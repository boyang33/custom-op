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
"""Tests for split_by_offset ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test
from tensor_split_by_offsets.python.ops import tensor_split_by_offsets_ops


class TensorSplitByOffsetsTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testSplitByOffset1D(self):
        """测试1维tensor的拆分"""
        with self.cached_session():
            # 创建输入tensor
            input_tensor = constant_op.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtypes.int32)
            offsets = constant_op.constant([[0, 3], [3, 2], [5, 4]], dtype=dtypes.int64)
            
            # 调用算子
            with tf.device('/GPU:0'):
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
            
            # 验证结果
            expected_outputs = [
                [1, 2, 3],      # offset=0, length=3
                [4, 5],         # offset=3, length=2
                [6, 7, 8, 9]    # offset=5, length=4
            ]
            
            output_values = [tensor.eval() for tensor in output_tensors]
            for i, (expected, actual) in enumerate(zip(expected_outputs, output_values)):
                self.assertAllEqual(expected, actual, msg=f"Output tensor {i} mismatch")

    @test_util.run_deprecated_v1
    def testSplitByOffset2D(self):
        """测试2维tensor的拆分"""
        with self.cached_session():
            # 创建2维输入tensor
            input_tensor = constant_op.constant([
                [1, 2, 3], [4, 5, 6],           # 前2行
                [7, 8, 9],                      # 第3行
                [10, 11, 12], [13, 14, 15], [16, 17, 18]  # 后3行
            ], dtype=dtypes.int32)  # shape: [6, 3]
            
            offsets = constant_op.constant([[0, 2], [2, 1], [3, 3]], dtype=dtypes.int64)
            
            # 调用算子 - 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
            
            # 验证结果
            expected_outputs = [
                [[1, 2, 3], [4, 5, 6]],                           # offset=0, length=2
                [[7, 8, 9]],                                      # offset=2, length=1
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]]       # offset=3, length=3
            ]
            
            output_values = [tensor.eval() for tensor in output_tensors]
            for i, (expected, actual) in enumerate(zip(expected_outputs, output_values)):
                self.assertAllEqual(expected, actual, msg=f"Output tensor {i} mismatch")

    @test_util.run_deprecated_v1
    def testSplitByOffset3D(self):
        """测试3维tensor的拆分"""
        with self.cached_session():
            # 创建3维输入tensor
            input_tensor = constant_op.constant([
                [[1, 2], [3, 4]], [[5, 6], [7, 8]],           # 前2个
                [[9, 10], [11, 12]],                           # 第3个
                [[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]  # 后3个
            ], dtype=dtypes.float32)  # shape: [6, 2, 2]
            
            offsets = constant_op.constant([[0, 2], [2, 1], [3, 3]], dtype=dtypes.int64)
            
            # 调用算子 - 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
            
            # 验证结果
            expected_outputs = [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],                           # offset=0, length=2
                [[[9, 10], [11, 12]]],                                           # offset=2, length=1
                [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]  # offset=3, length=3
            ]
            
            output_values = [tensor.eval() for tensor in output_tensors]
            for i, (expected, actual) in enumerate(zip(expected_outputs, output_values)):
                self.assertAllClose(expected, actual, rtol=1e-6, msg=f"Output tensor {i} mismatch")

    @test_util.run_deprecated_v1
    def testSplitByOffset4D(self):
        """测试4维tensor的拆分"""
        with self.cached_session():
            # 创建4维输入tensor
            np.random.seed(42)
            input_data = np.random.rand(6, 3, 4, 2).astype(np.float32)  # shape: [6, 3, 4, 2]
            input_tensor = constant_op.constant(input_data)
            
            offsets = constant_op.constant([[0, 2], [2, 1], [3, 3]], dtype=dtypes.int64)
            
            # 调用算子 - 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
            
            # 验证结果形状
            expected_shapes = [(2, 3, 4, 2), (1, 3, 4, 2), (3, 3, 4, 2)]
            output_values = [tensor.eval() for tensor in output_tensors]
            
            for i, (expected_shape, actual_output) in enumerate(zip(expected_shapes, output_values)):
                self.assertEqual(expected_shape, actual_output.shape, msg=f"Output tensor {i} shape mismatch")
            
            # 验证数据正确性
            self.assertAllClose(input_data[0:2], output_values[0], rtol=1e-6)
            self.assertAllClose(input_data[2:3], output_values[1], rtol=1e-6)
            self.assertAllClose(input_data[3:6], output_values[2], rtol=1e-6)

    @test_util.run_deprecated_v1
    def testSplitByOffsetFloat32(self):
        """测试float32类型的2维tensor"""
        with self.cached_session():
            # 测试float32类型
            input_tensor = constant_op.constant([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=dtypes.float32)
            offsets = constant_op.constant([[0, 2], [2, 1]], dtype=dtypes.int64)
            
            # 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
            
            expected_outputs = [
                [[1.1, 2.2], [3.3, 4.4]],  # offset=0, length=2
                [[5.5, 6.6]]               # offset=2, length=1
            ]
            
            output_values = [tensor.eval() for tensor in output_tensors]
            for i, (expected, actual) in enumerate(zip(expected_outputs, output_values)):
                self.assertAllClose(expected, actual, rtol=1e-6, msg=f"Output tensor {i} mismatch")

    @test_util.run_deprecated_v1
    def testSplitByOffsetEmptySlice(self):
        """测试包含空切片的拆分"""
        with self.cached_session():
            # 测试包含空切片
            input_tensor = constant_op.constant([[1, 2], [3, 4], [5, 6]], dtype=dtypes.int32)
            offsets = constant_op.constant([[0, 2], [2, 0], [2, 1]], dtype=dtypes.int64)  # 中间一个是空切片
            
            # 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
            
            expected_shapes = [(2, 2), (0, 2), (1, 2)]  # 第二个输出是空tensor
            output_values = [tensor.eval() for tensor in output_tensors]
            
            for i, (expected_shape, actual_output) in enumerate(zip(expected_shapes, output_values)):
                self.assertEqual(expected_shape, actual_output.shape, msg=f"Output tensor {i} shape mismatch")
            
            # 验证非空输出的内容
            self.assertAllEqual([[1, 2], [3, 4]], output_values[0])
            self.assertAllEqual([[5, 6]], output_values[2])

    @test_util.run_deprecated_v1
    def testSplitByOffsetSingleOutput(self):
        """测试单个输出的情况"""
        with self.cached_session():
            # 测试单个输出
            input_tensor = constant_op.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtypes.int32)
            offsets = constant_op.constant([[0, 3]], dtype=dtypes.int64)  # 只有一个输出
            
            # 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
            
            self.assertEqual(len(output_tensors), 1)
            output_value = output_tensors[0].eval()
            expected_output = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            
            self.assertAllEqual(expected_output, output_value)



    @test_util.run_deprecated_v1
    def testSplitByOffsetGradient(self):
        """测试梯度计算"""
        with self.cached_session():
            # 测试梯度计算
            input_tensor = constant_op.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtypes.float32)
            offsets = constant_op.constant([[0, 2], [2, 1]], dtype=dtypes.int64)
            
            # 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
            
            # 对第一个输出计算梯度
            err = gradient_checker.compute_gradient_error(
                    input_tensor, [3, 2], output_tensors[0], [2, 2])
            self.assertLess(err, 1e-4)

    @test_util.run_deprecated_v1
    def testRoundTripWithMerge(self):
        """测试与tensor_merge_with_offsets的往返一致性"""
        with self.cached_session():
            # 创建原始tensor列表
            original_tensors = [
                constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32),      # shape: [2, 2]
                constant_op.constant([[5, 6]], dtype=dtypes.float32),              # shape: [1, 2]
                constant_op.constant([[7, 8], [9, 10], [11, 12]], dtype=dtypes.float32)  # shape: [3, 2]
            ]
            
            # 模拟merge操作（使用tf.concat）
            merged_tensor = tf.concat(original_tensors, axis=0)
            
            # 计算offsets
            offsets_data = []
            current_offset = 0
            for tensor in original_tensors:
                length = tensor.shape[0]
                offsets_data.append([current_offset, length])
                current_offset += length
            offsets = constant_op.constant(offsets_data, dtype=dtypes.int64)
            
            # 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(merged_tensor, offsets)
            
            # 验证往返一致性
            split_values = [tensor.eval() for tensor in split_tensors]
            original_values = [tensor.eval() for tensor in original_tensors]
            
            for i, (original, split) in enumerate(zip(original_values, split_values)):
                self.assertAllClose(original, split, rtol=1e-6, msg=f"Round-trip failed for tensor {i}")

    @test_util.run_deprecated_v1
    def testRoundTrip3D(self):
        """测试3维tensor的往返一致性"""
        with self.cached_session():
            # 创建3维原始tensor列表
            original_tensors = [
                constant_op.constant([[[1, 2], [3, 4]]], dtype=dtypes.float32),                      # shape: [1, 2, 2]
                constant_op.constant([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=dtypes.float32) # shape: [2, 2, 2]
            ]
            
            # 模拟merge操作
            merged_tensor = tf.concat(original_tensors, axis=0)
            
            # 计算offsets
            offsets = constant_op.constant([[0, 1], [1, 2]], dtype=dtypes.int64)
            
            # 使用GPU测试修复的kernel
            with tf.device('/GPU:0'):
                split_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(merged_tensor, offsets)
            
            # 验证往返一致性
            split_values = [tensor.eval() for tensor in split_tensors]
            original_values = [tensor.eval() for tensor in original_tensors]
            
            for i, (original, split) in enumerate(zip(original_values, split_values)):
                self.assertAllClose(original, split, rtol=1e-6, msg=f"3D Round-trip failed for tensor {i}")

    @test_util.run_deprecated_v1
    def testZeroCopyOptimization(self):
        """测试零拷贝优化情况"""
        with self.cached_session():
            # 创建一个大的tensor，满足零拷贝条件（大小足够且内存对齐）
            # 使用较大的tensor以触发零拷贝优化
            np.random.seed(123)
            large_data = np.random.rand(1000, 128).astype(np.float32)  # 大tensor，128K个元素
            input_tensor = constant_op.constant(large_data)
            
            # CPU设备上测试零拷贝（GPU上总是复制数据）
            with tf.device('/CPU:0'):
                # 创建连续的大切片，更容易触发零拷贝
                offsets = constant_op.constant([
                    [0, 400],    # 第一个大切片：400行 * 128列 = 51200个元素
                    [400, 300],  # 第二个大切片：300行 * 128列 = 38400个元素  
                    [700, 300]   # 第三个大切片：300行 * 128列 = 38400个元素
                ], dtype=dtypes.int64)
                
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
                
                # 验证结果正确性
                output_values = [tensor.eval() for tensor in output_tensors]
                
                # 验证形状
                expected_shapes = [(400, 128), (300, 128), (300, 128)]
                for i, (expected_shape, actual_output) in enumerate(zip(expected_shapes, output_values)):
                    self.assertEqual(expected_shape, actual_output.shape, msg=f"Zero-copy tensor {i} shape mismatch")
                
                # 验证数据正确性
                self.assertAllClose(large_data[0:400], output_values[0], rtol=1e-6, msg="Zero-copy slice 1 data mismatch")
                self.assertAllClose(large_data[400:700], output_values[1], rtol=1e-6, msg="Zero-copy slice 2 data mismatch")
                self.assertAllClose(large_data[700:1000], output_values[2], rtol=1e-6, msg="Zero-copy slice 3 data mismatch")

    @test_util.run_deprecated_v1
    def testAllZeroCopyCase(self):
        """测试全部输出都使用零拷贝的情况"""
        with self.cached_session():
            # 创建一个特别设计的tensor，所有切片都满足零拷贝条件
            # 使用对齐的大小和连续的内存布局
            np.random.seed(456)
            aligned_data = np.random.rand(2048, 64).astype(np.float32)  # 2048*64 = 131072个元素
            input_tensor = constant_op.constant(aligned_data)
            
            # CPU设备上测试（GPU上不使用零拷贝）
            with tf.device('/CPU:0'):
                # 创建多个大的连续切片
                offsets = constant_op.constant([
                    [0, 512],     # 512行 * 64列 = 32768个元素
                    [512, 512],   # 512行 * 64列 = 32768个元素
                    [1024, 512],  # 512行 * 64列 = 32768个元素
                    [1536, 512]   # 512行 * 64列 = 32768个元素
                ], dtype=dtypes.int64)
                
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
                
                # 验证所有输出都正确
                output_values = [tensor.eval() for tensor in output_tensors]
                
                # 验证形状
                expected_shape = (512, 64)
                for i, actual_output in enumerate(output_values):
                    self.assertEqual(expected_shape, actual_output.shape, msg=f"All-zero-copy tensor {i} shape mismatch")
                
                # 验证数据正确性
                self.assertAllClose(aligned_data[0:512], output_values[0], rtol=1e-6, msg="All-zero-copy slice 1 mismatch")
                self.assertAllClose(aligned_data[512:1024], output_values[1], rtol=1e-6, msg="All-zero-copy slice 2 mismatch")
                self.assertAllClose(aligned_data[1024:1536], output_values[2], rtol=1e-6, msg="All-zero-copy slice 3 mismatch")
                self.assertAllClose(aligned_data[1536:2048], output_values[3], rtol=1e-6, msg="All-zero-copy slice 4 mismatch")

    @test_util.run_deprecated_v1
    def testMixedZeroCopyAndCopy(self):
        """测试零拷贝和数据复制混合的情况"""
        with self.cached_session():
            # 创建混合情况：一些切片满足零拷贝条件，一些不满足
            np.random.seed(789)
            mixed_data = np.random.rand(1200, 100).astype(np.float32)
            input_tensor = constant_op.constant(mixed_data)
            
            with tf.device('/CPU:0'):
                offsets = constant_op.constant([
                    [0, 500],    # 大切片，可能零拷贝：500*100 = 50000个元素
                    [500, 10],   # 小切片，需要复制：10*100 = 1000个元素
                    [510, 600],  # 大切片，可能零拷贝：600*100 = 60000个元素
                    [1110, 5],   # 小切片，需要复制：5*100 = 500个元素
                    [1115, 85]   # 中等切片：85*100 = 8500个元素
                ], dtype=dtypes.int64)
                
                output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(input_tensor, offsets)
                output_values = [tensor.eval() for tensor in output_tensors]
                
                # 验证形状
                expected_shapes = [(500, 100), (10, 100), (600, 100), (5, 100), (85, 100)]
                for i, (expected_shape, actual_output) in enumerate(zip(expected_shapes, output_values)):
                    self.assertEqual(expected_shape, actual_output.shape, msg=f"Mixed tensor {i} shape mismatch")
                
                # 验证数据正确性
                self.assertAllClose(mixed_data[0:500], output_values[0], rtol=1e-6, msg="Mixed case slice 1 mismatch")
                self.assertAllClose(mixed_data[500:510], output_values[1], rtol=1e-6, msg="Mixed case slice 2 mismatch")
                self.assertAllClose(mixed_data[510:1110], output_values[2], rtol=1e-6, msg="Mixed case slice 3 mismatch")
                self.assertAllClose(mixed_data[1110:1115], output_values[3], rtol=1e-6, msg="Mixed case slice 4 mismatch")
                self.assertAllClose(mixed_data[1115:1200], output_values[4], rtol=1e-6, msg="Mixed case slice 5 mismatch")

    @test_util.run_deprecated_v1
    def testDifferentAlignmentValues(self):
        """测试不同alignment参数值的效果"""
        with self.cached_session():
            # 创建测试数据
            input_tensor = constant_op.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtypes.float32)
            offsets = constant_op.constant([[0, 4], [4, 3], [7, 3]], dtype=dtypes.int64)
            
            # 测试不同的alignment值
            alignment_values = [1, 2, 4, 8, 16, 32, 64, 128]
            
            expected_outputs = [
                [1.0, 2.0, 3.0, 4.0],      # offset=0, length=4
                [5.0, 6.0, 7.0],           # offset=4, length=3
                [8.0, 9.0, 10.0]           # offset=7, length=3
            ]
            
            for alignment in alignment_values:
                try:
                    output_tensors = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                            input_tensor, offsets, alignment=alignment)
                    
                    output_values = [tensor.eval() for tensor in output_tensors]
                    
                    # 验证结果与alignment无关，数据一致性应该保持  
                    for i, (expected, actual) in enumerate(zip(expected_outputs, output_values)):
                        self.assertAllEqual(expected, actual, 
                                                            msg=f"Alignment {alignment}, output tensor {i} mismatch")
                except Exception as e:
                    self.fail(f"Failed with alignment={alignment}: {e}")

    @test_util.run_deprecated_v1
    def testInvalidAlignmentValues(self):
        """测试无效的alignment参数值"""
        input_tensor = constant_op.constant([1, 2, 3, 4], dtype=dtypes.float32)
        offsets = constant_op.constant([[0, 2], [2, 2]], dtype=dtypes.int64)
        
        # 测试无效的alignment值
        invalid_alignments = [0, -1, 3, 5, 7, 9, 15, 31, 63]  # 非2的幂或非正数
        
        for invalid_alignment in invalid_alignments:
            with self.assertRaises(ValueError, msg=f"Should reject alignment={invalid_alignment}"):
                tensor_split_by_offsets_ops.tensor_split_by_offsets(
                        input_tensor, offsets, alignment=invalid_alignment)

    @test_util.run_deprecated_v1
    def testAlignmentWithZeroCopyBehavior(self):
        """测试alignment对零拷贝行为的影响"""
        with self.cached_session():
            # 创建较大的tensor以便观察零拷贝效果
            np.random.seed(42)
            large_data = np.random.rand(2000, 50).astype(np.float32)
            input_tensor = constant_op.constant(large_data)
            offsets = constant_op.constant([[0, 1000], [1000, 1000]], dtype=dtypes.int64)
            
            # 测试不同alignment对零拷贝的影响
            alignments_to_test = [16, 32, 64, 128]
            
            for alignment in alignments_to_test:
                try:
                    # 测试启用零拷贝
                    output_zero_copy = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                            input_tensor, offsets, use_alignment=True, alignment=alignment)
                    
                    # 测试禁用零拷贝
                    output_copy = tensor_split_by_offsets_ops.tensor_split_by_offsets(
                            input_tensor, offsets, use_alignment=False, alignment=alignment)
                    
                    # 验证结果一致性（无论是否零拷贝，结果应该相同）
                    zero_copy_values = [tensor.eval() for tensor in output_zero_copy]
                    copy_values = [tensor.eval() for tensor in output_copy]
                    
                    for i, (zc_val, cp_val) in enumerate(zip(zero_copy_values, copy_values)):
                        self.assertAllClose(zc_val, cp_val, rtol=1e-6,
                                                            msg=f"Alignment {alignment}, tensor {i}: zero-copy vs copy mismatch")
                        
                        # 验证与原始数据的一致性
                        expected_start = 0 if i == 0 else 1000
                        expected_end = 1000 if i == 0 else 2000
                        self.assertAllClose(large_data[expected_start:expected_end], zc_val, rtol=1e-6,
                                                            msg=f"Alignment {alignment}, tensor {i}: data mismatch")
                except Exception as e:
                    self.fail(f"Failed with alignment={alignment}: {e}")


if __name__ == "__main__":
    test.main()