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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("TensorSplitByOffsets")
    .Input("input: T")
    .Input("offsets: int64")
    .Output("output: N * T")
    .Attr("N: int >= 1")
    .Attr("T: {float, double, int32, int64}")
    .Attr("use_alignment: bool = true")
    .Attr("alignment: int = 64")

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input_shape));

      ::tensorflow::shape_inference::ShapeHandle offsets_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &offsets_shape));

      // offsets应该是[N, 2]的形状
      ::tensorflow::shape_inference::DimensionHandle offsets_cols;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(offsets_shape, 1), 2, &offsets_cols));

      // 获取N的值
      int N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));

      // 验证offsets的第一维是否等于N
      ::tensorflow::shape_inference::DimensionHandle offsets_rows = c->Dim(offsets_shape, 0);
      if (c->ValueKnown(offsets_rows) && c->Value(offsets_rows) != N) {
        return tensorflow::errors::InvalidArgument("offsets first dimension must equal N, got ",
                                                   c->Value(offsets_rows), " vs ", N);
      }

      // 输出形状：每个输出tensor保持输入tensor除第0维外的所有维度
      const int input_rank = c->Rank(input_shape);
      std::vector<::tensorflow::shape_inference::ShapeHandle> output_shapes(N);

      for (int i = 0; i < N; ++i) {
        std::vector<::tensorflow::shape_inference::DimensionHandle> dims;
        // 第0维长度未知（由offsets决定）
        dims.push_back(c->UnknownDim());
        // 其他维度与输入相同
        for (int dim = 1; dim < input_rank; ++dim) {
          dims.push_back(c->Dim(input_shape, dim));
        }
        output_shapes[i] = c->MakeShape(dims);
      }

      TF_RETURN_IF_ERROR(c->set_output("output", output_shapes));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
    Split a tensor into multiple tensors based on offset information along dimension 0.

    input: Input tensor to be split (must have at least 1 dimension).
    offsets: A 2D tensor of shape [N, 2] where each row contains [start, length] for dimension 0.
    output: N output tensors, each containing the corresponding slice of the input along dimension 0.
    All dimensions except dimension 0 remain unchanged.
    N: Number of output tensors, must match the first dimension of offsets.
    use_alignment: Whether to use alignment-based slice optimization when possible. Default is true.
    alignment: Memory alignment requirement in bytes for slice optimization. Default is 64.
    Must be a power of 2. Used to determine if memory pointers are properly aligned.
    )doc");
