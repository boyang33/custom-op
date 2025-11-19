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


if __name__ == "__main__":
    test.main()