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


    @test_util.run_deprecated_v1
    def testCPUParallelization(self):
        """测试CPU并行化功能"""
        import os
        with self.cached_session():
            # 创建足够大的数据以触发并行化
            # 默认阈值是1MB (1024*1024个元素)
            np.random.seed(100)
            # 创建多个大tensor，总数超过阈值
            num_inputs = 10
            inputs = []
            for i in range(num_inputs):
                # 每个tensor: 150行 * 800列 = 120,000个元素
                data = np.random.rand(150, 800).astype(np.float32)
                inputs.append(constant_op.constant(data, dtype=dtypes.float32))
            
            # 在CPU设备上测试
            with test.mock.patch.dict(os.environ, {'TF_CONCAT_PARALLEL_THRESHOLD': '1000000'}):
                concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                    inputs, use_alignment=False)
                
                # 验证结果
                concatenated_val = concatenated.eval()
                offsets_val = offsets.eval()
                
                # 验证形状
                self.assertEqual(concatenated_val.shape, (1500, 800))  # 10 * 150 = 1500
                self.assertEqual(offsets_val.shape, (10, 2))
                
                # 验证offsets正确性
                expected_offset = 0
                for i in range(num_inputs):
                    self.assertEqual(offsets_val[i, 0], expected_offset)
                    self.assertEqual(offsets_val[i, 1], 150)
                    expected_offset += 150

    @test_util.run_deprecated_v1
    def testCPUParallelizationSmallData(self):
        """测试CPU小数据不触发并行化"""
        with self.cached_session():
            # 创建小数据，不应触发并行化
            inputs = [
                constant_op.constant(np.random.rand(30, 50).astype(np.float32)),
                constant_op.constant(np.random.rand(40, 50).astype(np.float32)),
                constant_op.constant(np.random.rand(30, 50).astype(np.float32))
            ]
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                inputs, use_alignment=False)
            
            concatenated_val = concatenated.eval()
            offsets_val = offsets.eval()
            
            # 验证结果正确性（无论是否并行化，结果应该一致）
            self.assertEqual(concatenated_val.shape, (100, 50))  # 30 + 40 + 30 = 100
            expected_offsets = [[0, 30], [30, 40], [70, 30]]
            self.assertAllEqual(offsets_val, expected_offsets)

    @test_util.run_deprecated_v1
    def testParallelizationWithMixedSizes(self):
        """测试混合大小输入的并行化"""
        with self.cached_session():
            # 创建包含大小不一的输入
            np.random.seed(500)
            inputs = [
                constant_op.constant(np.random.rand(800, 400).astype(np.float32)),   # 大
                constant_op.constant(np.random.rand(50, 400).astype(np.float32)),    # 小
                constant_op.constant(np.random.rand(700, 400).astype(np.float32)),   # 大
                constant_op.constant(np.random.rand(100, 400).astype(np.float32)),   # 小
                constant_op.constant(np.random.rand(850, 400).astype(np.float32))    # 大
            ]
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                inputs, use_alignment=False)
            
            concatenated_val = concatenated.eval()
            offsets_val = offsets.eval()
            
            # 验证形状
            self.assertEqual(concatenated_val.shape, (2500, 400))  # 800+50+700+100+850=2500
            
            # 验证offsets
            expected_offsets = [[0, 800], [800, 50], [850, 700], [1550, 100], [1650, 850]]
            self.assertAllEqual(offsets_val, expected_offsets)


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

    @test_util.run_deprecated_v1
    def testGPUMultiStream(self):
        """测试GPU多流并发功能"""
        if not test_util.is_gpu_available(cuda_only=True):
            self.skipTest("GPU not available")
        
        with self.cached_session(use_gpu=True):
            # 创建大量输入（>8个以触发GPU多流）
            np.random.seed(200)
            # 创建10个输入tensor（超过8个以触发多流优化）
            num_inputs = 10
            inputs = []
            for i in range(num_inputs):
                data = np.random.rand(150, 512).astype(np.float32)
                inputs.append(constant_op.constant(data, dtype=dtypes.float32))
            
            # 在GPU设备上测试
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                inputs, use_alignment=False)
            
            # 验证结果
            concatenated_val = concatenated.eval()
            offsets_val = offsets.eval()
            
            # 验证形状
            self.assertEqual(concatenated_val.shape, (1500, 512))  # 10 * 150 = 1500
            self.assertEqual(offsets_val.shape, (10, 2))
            
            # 验证offsets
            expected_offset = 0
            for i in range(num_inputs):
                self.assertEqual(offsets_val[i, 0], expected_offset)
                self.assertEqual(offsets_val[i, 1], 150)
                expected_offset += 150

    @test_util.run_deprecated_v1
    def testGPUSingleStream(self):
        """测试GPU单流情况（输入数量<=8）"""
        if not test_util.is_gpu_available(cuda_only=True):
            self.skipTest("GPU not available")
        
        with self.cached_session(use_gpu=True):
            # 创建数据，但输入数量少（<=8），不触发多流
            np.random.seed(300)
            num_inputs = 6
            inputs = []
            for i in range(num_inputs):
                data = np.random.rand(100, 256).astype(np.float32)
                inputs.append(constant_op.constant(data, dtype=dtypes.float32))
            
            concatenated, offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                inputs, use_alignment=False)
            
            concatenated_val = concatenated.eval()
            offsets_val = offsets.eval()
            
            # 验证结果正确性
            self.assertEqual(concatenated_val.shape, (600, 256))  # 6 * 100 = 600
            
            # 验证offsets
            for i in range(num_inputs):
                self.assertEqual(offsets_val[i, 0], i * 100)
                self.assertEqual(offsets_val[i, 1], 100)

    @test_util.run_deprecated_v1
    def testCPUGPUConsistency(self):
        """测试CPU和GPU结果的一致性"""
        if not test_util.is_gpu_available(cuda_only=True):
            self.skipTest("GPU not available")
        
        with self.cached_session():
            # 创建相同的测试数据
            np.random.seed(400)
            num_inputs = 5
            inputs = []
            for i in range(num_inputs):
                data = np.random.rand(200, 400).astype(np.float32)
                inputs.append(constant_op.constant(data, dtype=dtypes.float32))
            
            # CPU执行
            with test.mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
                cpu_concat, cpu_offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                    inputs, use_alignment=False)
                cpu_concat_val = cpu_concat.eval()
                cpu_offsets_val = cpu_offsets.eval()
            
            # GPU执行
            gpu_concat, gpu_offsets = tensor_concat_with_offsets_ops.tensor_concat_with_offsets(
                inputs, use_alignment=False)
            gpu_concat_val = gpu_concat.eval()
            gpu_offsets_val = gpu_offsets.eval()
            
            # 验证CPU和GPU结果一致
            self.assertAllClose(cpu_concat_val, gpu_concat_val, rtol=1e-5, atol=1e-6,
                              msg="CPU/GPU concatenated tensor inconsistency")
            self.assertAllEqual(cpu_offsets_val, gpu_offsets_val,
                              msg="CPU/GPU offsets inconsistency")


if __name__ == "__main__":
    test.main()