

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#pragma once

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include "core/graph/graph_utils.h"
// Uncomment for debugging
// #define NEED_LOG_DEBUG_INFO 1

#ifndef LOG_DEBUG_INFO
#ifdef NEED_LOG_DEBUG_INFO
#define LOG_DEBUG_INFO(logger, message) LOGS(logger, WARNING) << message
#else
#define LOG_DEBUG_INFO(logger, message) \
  ORT_UNUSED_PARAMETER(logger);         \
  do {                                  \
  } while (0)
#endif
#endif

namespace onnxruntime::optimizer::compute_optimizer {

using OPSET_VERSION_LIST = std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>;
const OPSET_VERSION_LIST opset_1{1};
const OPSET_VERSION_LIST opset_13_1{13, 1};
const OPSET_VERSION_LIST opset_13_9_1{13, 9, 1};
const OPSET_VERSION_LIST opset_13_11_1{13, 11, 1};
const OPSET_VERSION_LIST opset_13_9_6_1{13, 9, 6, 1};
const OPSET_VERSION_LIST opset_14_13_5_1{14, 13, 5, 1};
const OPSET_VERSION_LIST opset_14_13_7_6_1{14, 13, 7, 6, 1};
const OPSET_VERSION_LIST opset_13_12_10_7_6_1{13, 12, 10, 7, 6, 1};

/**
 * @brief Base class for all upstream operator passthrough actors.
 */
struct UpStreamOperatorActorBase {
};

/**
 * @brief Base class for all upstream operator info .
 */
struct UpstreamOperatorInfoBase {
  UpstreamOperatorInfoBase(Node* node) : node_ptr(node) {}

  Node* node_ptr;  // The node that triggers the optimization search.
};

/**
 * @brief Pass through configuration for specific operator.
 *
 * For each operator:
 * > `input_indices` can be used to explicitly specify the input indices that Slicing op can be passed through.
 *   This could be helpful if some inputs are not applicable for pass through. If not specified, all inputs
 *   are considered (but there will be checks to ignore those inputs that are not affected by the slicing axis).
 * > `actor` will be used to perform the actual pass through, including both pre-check stage and post process
 *   stage.
 */
template <typename T>
struct OpPassThroughConfig {
  OpPassThroughConfig(const std::vector<int>& input_indices,
                      std::shared_ptr<T> actor,
                      const OPSET_VERSION_LIST& opset_list)
      : input_indices(input_indices), actor(actor), opsets(opset_list) {
    // Compile-time check
    static_assert(std::is_base_of<UpStreamOperatorActorBase, T>::value,
                  "type parameter of this class must derive from UpStreamOperatorActorBase");
  }

  std::vector<int> input_indices;
  std::shared_ptr<T> actor;
  const OPSET_VERSION_LIST& opsets;
};

bool EnforceNodeAllInputOutputHaveShapes(const Node& node);

}  // namespace onnxruntime::optimizer::compute_optimizer
#endif
