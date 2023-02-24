// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/compute_optimizer/common.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief Graph transformer base that helps reduce compute FLOP while maintaining mathematically equivalent result.
 *
 * The series of graph transformations (inheriting from this base class) tries to identify opportunities to reduce
 * unnecessary computations on the graph level.
 * Currently, the major optimization is to bring some slice operators ahead as much as possible, to leave more ops
 * operate on sliced input data. Gather and GatherND are the entry operators that trigger the optimization search.
 */
template <typename T1, typename T2>
class UpStreamGraphTransformerBase : public GraphTransformer {
 public:
  UpStreamGraphTransformerBase(const std::string& name,
                               const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer(name, compatible_execution_providers) {
    // Compile-time check
    static_assert(std::is_base_of<UpstreamOperatorInfoBase, T1>::value,
                  "type parameter of this class must derive from UpstreamOperatorInfoBase");
    static_assert(std::is_base_of<UpStreamOperatorActorBase, T2>::value,
                  "type parameter of this class must derive from UpStreamOperatorActorBase");
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 protected:
  virtual std::optional<T1> IsSupportedForUpstream(Graph& graph, Node& node, const logging::Logger& logger) const = 0;

  virtual bool UpStreamInternal(Graph& graph, std::deque<T1>& queue,
                                Node& current_node, T1& info,
                                const OpPassThroughConfig<T2>& pass_through_config,
                                const logging::Logger& logger,
                                const std::string& entry_node_name) const = 0;

  std::string GetFullQualifiedOpName(const std::string& op_type, const std::string& domain) const {
    return domain + "::" + op_type;
  }

  std::unordered_map<std::string, OpPassThroughConfig<T2>> allowed_passthrough_ops_;

 private:
  bool Upstream(Graph& graph, std::deque<T1>& queue, Node& current_node, T1& info, const logging::Logger& logger,
                std::string& entry_node_name)
      const {
    const std::string op_type = GetFullQualifiedOpName(current_node.OpType(), current_node.Domain());
    if (allowed_passthrough_ops_.count(op_type)) {
      auto& pass_through_config = allowed_passthrough_ops_.at(op_type);
      LOG_DEBUG_INFO(logger, "Enter reorder handle for node " + current_node.Name() + "(" + op_type + ")");

      if (!graph_utils::IsSupportedOptypeVersionAndDomain(current_node, current_node.OpType(),
                                                          pass_through_config.opsets, current_node.Domain())) {
        LOG_DEBUG_INFO(logger, "Unsupported opset for " + current_node.Name() + "(" + op_type + ") since version: " +
                                   std::to_string(current_node.SinceVersion()));
        return false;
      }

      if (!EnforceNodeAllInputOutputHaveShapes(current_node)) {
        LOG_DEBUG_INFO(logger, "Some inputs/outputs' shape not found for node " + current_node.Name() + "(" +
                                   op_type + ")");
        return false;
      }

      return UpStreamInternal(graph, queue, current_node, info, pass_through_config, logger, entry_node_name);
    } else {
      LOG_DEBUG_INFO(logger, "op_type not supported for " + current_node.Name() + "(" + op_type + ")");
      return false;
    }
  }
};

}  // namespace onnxruntime::optimizer::compute_optimizer
#endif
