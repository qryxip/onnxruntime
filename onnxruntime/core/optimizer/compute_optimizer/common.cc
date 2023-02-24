// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE

#include "core/optimizer/compute_optimizer/common.h"

namespace onnxruntime::optimizer::compute_optimizer {

bool EnforceNodeAllInputOutputHaveShapes(const Node& node) {
  for (const auto* input_def : node.InputDefs()) {
    if (!input_def->Shape()) {
      return false;
    }
  }

  for (const auto* output_def : node.OutputDefs()) {
    if (!output_def->Shape()) {
      return false;
    }
  }
  return true;
}

}  // namespace onnxruntime::optimizer::compute_optimizer
#endif
