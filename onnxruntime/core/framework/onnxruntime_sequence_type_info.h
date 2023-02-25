// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_c_api.h"
#include <memory>

#include "core/framework/onnxruntime_typeinfo.h"

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtSequenceTypeInfo {
 public:

  using Ptr = std::unique_ptr<OrtSequenceTypeInfo>;

  explicit OrtSequenceTypeInfo(OrtTypeInfo::Ptr sequence_key_type) noexcept;
  ~OrtSequenceTypeInfo();

  OrtTypeInfo::Ptr sequence_key_type_;

  Ptr Clone() const;

  static Ptr FromTypeProto(const ONNX_NAMESPACE::TypeProto*);

 private:
  OrtSequenceTypeInfo(const OrtSequenceTypeInfo& other) = delete;
  OrtSequenceTypeInfo& operator=(const OrtSequenceTypeInfo& other) = delete;
};
