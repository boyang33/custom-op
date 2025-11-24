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
"""Tests for tensor_concat_with_offsets ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test
from tensor_concat_with_offsets.python.ops import tensor_concat_with_offsets_ops


class TensorConcatWithOffsetsTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsets(self):
        with self.cached_session():
            # 创建测试数据
            input1 = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
            input2 = constant_op.constant([4.0, 5.0], dtype=dtypes.float32)
            input3 = constant_op.constant([6.0, 7.0, 8.0, 9.0], dtype=dtypes.float32)
            
            # 调用算子
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1, input2, input3], use_alignment=False)
            
            # 验证结果
            expected_concatenated = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            expected_offsets = [[0, 3], [3, 2], [5, 4]]
            
            self.assertAllEqual(concatenated.eval(), expected_concatenated)
            self.assertAllEqual(offsets.eval(), expected_offsets)

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsInt32(self):
        with self.cached_session():
            # 测试int32类型
            input1 = constant_op.constant([1, 2], dtype=dtypes.int32)
            input2 = constant_op.constant([3, 4, 5], dtype=dtypes.int32)
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1, input2], use_alignment=False)
            
            expected_concatenated = [1, 2, 3, 4, 5]
            expected_offsets = [[0, 2], [2, 3]]
            
            self.assertAllEqual(concatenated.eval(), expected_concatenated)
            self.assertAllEqual(offsets.eval(), expected_offsets)

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsSingleInput(self):
        with self.cached_session():
            # 测试单个输入
            input1 = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1], use_alignment=False)
            
            expected_concatenated = [1.0, 2.0, 3.0]
            expected_offsets = [[0, 3]]
            
            self.assertAllEqual(concatenated.eval(), expected_concatenated)
            self.assertAllEqual(offsets.eval(), expected_offsets)

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsEmptyTensor(self):
        with self.cached_session():
            # 测试包含空tensor的情况
            input1 = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
            input2 = constant_op.constant([], dtype=dtypes.float32)
            input3 = constant_op.constant([3.0], dtype=dtypes.float32)
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1, input2, input3], use_alignment=False)
            
            expected_concatenated = [1.0, 2.0, 3.0]
            expected_offsets = [[0, 2], [2, 0], [2, 1]]
            
            self.assertAllEqual(concatenated.eval(), expected_concatenated)
            self.assertAllEqual(offsets.eval(), expected_offsets)

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsAlignment(self):
        with self.cached_session():
            # 测试对齐功能
            input1 = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
            input2 = constant_op.constant([3.0, 4.0, 5.0], dtype=dtypes.float32)
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1, input2], alignment=64, use_alignment=True)
            
            # 验证offsets格式正确
            offsets_val = offsets.eval()
            self.assertEqual(offsets_val.shape, (2, 2))
            # 第一个tensor从0开始，长度2
            self.assertEqual(offsets_val[0, 1], 2)
            # 第二个tensor长度3
            self.assertEqual(offsets_val[1, 1], 3)

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsMultiDimensional(self):
        """测试多维tensor的合并"""
        with self.cached_session():
            # 创建2D tensor
            input1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32)  # shape: [2, 2]
            input2 = constant_op.constant([[5.0, 6.0]], dtype=dtypes.float32)  # shape: [1, 2]
            input3 = constant_op.constant([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=dtypes.float32)  # shape: [3, 2]
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1, input2, input3], use_alignment=False)
            
            # 验证offsets
            expected_offsets = [[0, 2], [2, 1], [3, 3]]
            self.assertAllEqual(offsets.eval(), expected_offsets)
            
            # 验证数据
            expected_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], 
                           [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
            concatenated_val = concatenated.eval()
            self.assertAllClose(concatenated_val, expected_data)
            
            # 验证shape
            self.assertEqual(concatenated_val.shape, (6, 2))  # 2 + 1 + 3 = 6 rows


class TensorConcatWithOffsetsGPUTest(test.TestCase):
    """GPU专用测试"""

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsGPU(self):
        """测试GPU上的基本合并操作"""
        if not test_util.is_gpu_available(cuda_only=True):
            self.skipTest("GPU not available")
        
        with self.cached_session(use_gpu=True):
            # 创建测试数据
            input1 = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
            input2 = constant_op.constant([4.0, 5.0], dtype=dtypes.float32)
            input3 = constant_op.constant([6.0, 7.0, 8.0, 9.0], dtype=dtypes.float32)
            
            # 调用算子
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1, input2, input3], use_alignment=False)
            
            # 验证结果
            expected_concatenated = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            expected_offsets = [[0, 3], [3, 2], [5, 4]]
            
            self.assertAllEqual(concatenated.eval(), expected_concatenated)
            self.assertAllEqual(offsets.eval(), expected_offsets)

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsGPULargeData(self):
        """测试GPU上的大数据合并"""
        if not test_util.is_gpu_available(cuda_only=True):
            self.skipTest("GPU not available")
        
        with self.cached_session(use_gpu=True):
            # 创建较大的测试数据
            size1, size2, size3 = 1000, 500, 2000
            input1 = constant_op.constant(np.arange(size1, dtype=np.float32))
            input2 = constant_op.constant(np.arange(size1, size1 + size2, dtype=np.float32))
            input3 = constant_op.constant(np.arange(size1 + size2, size1 + size2 + size3, dtype=np.float32))
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1, input2, input3], use_alignment=False)
            
            # 验证结果
            expected_concatenated = np.arange(size1 + size2 + size3, dtype=np.float32)
            expected_offsets = [[0, size1], [size1, size2], [size1 + size2, size3]]
            
            self.assertAllClose(concatenated.eval(), expected_concatenated)
            self.assertAllEqual(offsets.eval(), expected_offsets)

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsGPUMultiDimensional(self):
        """测试GPU上的多维tensor合并"""
        if not test_util.is_gpu_available(cuda_only=True):
            self.skipTest("GPU not available")
        
        with self.cached_session(use_gpu=True):
            # 创建3D tensor
            input1 = constant_op.constant(np.random.randn(10, 20, 30).astype(np.float32))
            input2 = constant_op.constant(np.random.randn(5, 20, 30).astype(np.float32))
            input3 = constant_op.constant(np.random.randn(15, 20, 30).astype(np.float32))
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1, input2, input3], use_alignment=False)
            
            # 验证offsets：[offset, length]
            # input1: offset=0, length=10
            # input2: offset=10, length=5
            # input3: offset=15, length=15
            expected_offsets = [[0, 10], [10, 5], [15, 15]]
            self.assertAllEqual(offsets.eval(), expected_offsets)
            
            # 验证实际输出shape - 在eval后检查
            concatenated_val = concatenated.eval()
            self.assertEqual(concatenated_val.shape, (30, 20, 30))

    @test_util.run_deprecated_v1
    def testTensorConcatWithOffsetsGPUDataTypes(self):
        """测试GPU上的不同数据类型"""
        if not test_util.is_gpu_available(cuda_only=True):
            self.skipTest("GPU not available")
        
        with self.cached_session(use_gpu=True):
            # 测试int32
            input1_int = constant_op.constant([1, 2, 3], dtype=dtypes.int32)
            input2_int = constant_op.constant([4, 5], dtype=dtypes.int32)
            
            concat_int, offsets_int = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1_int, input2_int], use_alignment=False)
            
            self.assertAllEqual(concat_int.eval(), [1, 2, 3, 4, 5])
            
            # 测试double
            input1_dbl = constant_op.constant([1.0, 2.0], dtype=dtypes.float64)
            input2_dbl = constant_op.constant([3.0, 4.0, 5.0], dtype=dtypes.float64)
            
            concat_dbl, offsets_dbl = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                [input1_dbl, input2_dbl], use_alignment=False)
            
            self.assertAllClose(concat_dbl.eval(), [1.0, 2.0, 3.0, 4.0, 5.0])


if __name__ == "__main__":
    test.main()