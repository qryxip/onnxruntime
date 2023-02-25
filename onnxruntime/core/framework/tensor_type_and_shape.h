// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/framework/tensor_shape.h"
#include "core/session/onnxruntime_c_api.h"

struct OrtTensorTypeAndShapeInfo {
 public:

  using Ptr = std::unique_ptr<OrtTensorTypeAndShapeInfo>;

  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  onnxruntime::TensorShape shape;
  // dim_param values. empty string if dim_value or no dim_param was specified.
  // one entry per dimension in shape. only guaranteed to be populated for graph inputs and outputs
  std::vector<std::string> dim_params;

  OrtTensorTypeAndShapeInfo() = default;
  ~OrtTensorTypeAndShapeInfo();

  Ptr Clone() const {
    return std::make_unique<OrtTensorTypeAndShapeInfo>(*this);
  }

  OrtTensorTypeAndShapeInfo(const OrtTensorTypeAndShapeInfo& other) = default;
  OrtTensorTypeAndShapeInfo& operator=(const OrtTensorTypeAndShapeInfo& other) = default;
};

constexpr ONNXTensorElementDataType TensorDataTypeToOnnxRuntimeTensorElementDataType(int32_t dtype);
