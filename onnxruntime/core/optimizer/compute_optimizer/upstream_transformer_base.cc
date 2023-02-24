// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#include <onnx/defs/attr_proto_util.h>

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"

namespace onnxruntime::optimizer::compute_optimizer {

template <typename T1, typename T2>
Status UpStreamGraphTransformerBase<T1, T2>::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                       const logging::Logger& logger)
    const {
  LOG_DEBUG_INFO(logger, "Enter UpStreamGraphTransformerBase");
  bool reordered = false;
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  const auto& graph_outputs = graph.GetOutputs();

  size_t reordered_node_count = 0;  // For summary
  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed.
      continue;

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    std::optional<T1> op_info = IsSupportedForUpstream(graph, node, logger);
    if (!op_info.has_value()) {
      continue;
    }

    auto& output_arg = node.MutableOutputDefs()[0];
    if (std::find(graph_outputs.begin(), graph_outputs.end(), output_arg) != graph_outputs.end()) {
      continue;
    }

    std::deque<T1> queue;
    queue.push_back(std::move(op_info.value()));

    std::string node_name = node.Name();
    std::string node_type = node.OpType();
    std::string log_prefix = "Entry node " + node_name + " (" + node_type + ") ";
    LOG_DEBUG_INFO(logger, log_prefix + " starts re-ordering check");

    // DON'T operate on `node` once this loop starts, as it may be removed from the graph.
    while (!queue.empty()) {
      T1 info = queue.front();
      Node* node_to_upstream = info.node_ptr;
      queue.pop_front();
      Node* input_tensor_producer_node =
          graph.GetMutableProducerNode(node_to_upstream->MutableInputDefs()[0]->Name());
      if (input_tensor_producer_node == nullptr) {
        break;
      }

      if (graph.GetConsumerNodes(input_tensor_producer_node->MutableOutputDefs()[0]->Name()).size() > 1) {
        LOG_DEBUG_INFO(logger, log_prefix + " stops at node " + input_tensor_producer_node->Name() +
                                   " since multiple consumer found");
        continue;
      }

      auto ret = Upstream(graph, queue, *input_tensor_producer_node, info, logger, node_name);
      if (ret) {
        LOG_DEBUG_INFO(logger, log_prefix + " moves up across node " + input_tensor_producer_node->Name());
        modified = true;
        reordered = true;
      } else {
        LOG_DEBUG_INFO(logger, log_prefix + " stops when handling " + input_tensor_producer_node->Name());
      }
    }

    if (reordered) {
      ++reordered_node_count;
    }
  }

  LOGS(logger, INFO) << "Exit UpStreamGraphTransformerBase with summary - reorderd_node_count:" << reordered_node_count
                     << " nodes.";
  return Status::OK();
}

template class UpStreamGraphTransformerBase<SliceInfo, UpStreamGatherOperatorActorBase>;

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
