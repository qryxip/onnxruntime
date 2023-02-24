// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

// #ifdef ENABLE_TRAINING_CORE
// #include <onnx/defs/attr_proto_util.h>

// #include "core/common/safeint.h"
// #include "core/graph/graph_utils.h"
// #include "core/optimizer/initializer.h"
// #include "core/optimizer/utils.h"
// #include "core/optimizer/compute_optimizer/upstream_gather_actors.h"
// #include "core/optimizer/compute_optimizer/upstream_reshape.h"
// #include "core/optimizer/compute_optimizer/common.h"

// using SliceInfo = onnxruntime::optimizer::compute_optimizer::SliceInfo;
// using namespace onnxruntime::optimizer::compute_optimizer;
// namespace onnxruntime {

// namespace {

// bool EnforceNodeAllInputOutputHaveShapes(const Node& node) {
//   for (const auto* input_def : node.InputDefs()) {
//     if (!input_def->Shape()) {
//       return false;
//     }
//   }

//   for (const auto* output_def : node.OutputDefs()) {
//     if (!output_def->Shape()) {
//       return false;
//     }
//   }
//   return true;
// }

// // using OPSET_VERSION_LIST = std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>;
// // const OPSET_VERSION_LIST opset_1{1};
// // const OPSET_VERSION_LIST opset_13_1{13, 1};
// // const OPSET_VERSION_LIST opset_13_9_1{13, 9, 1};
// // const OPSET_VERSION_LIST opset_13_11_1{13, 11, 1};
// // const OPSET_VERSION_LIST opset_13_9_6_1{13, 9, 6, 1};
// // const OPSET_VERSION_LIST opset_14_13_5_1{14, 13, 5, 1};
// // const OPSET_VERSION_LIST opset_14_13_7_6_1{14, 13, 7, 6, 1};
// // const OPSET_VERSION_LIST opset_13_12_10_7_6_1{13, 12, 10, 7, 6, 1};

// /**
//  * @brief Functor to trigger the optimization search for a given Reshape node.
//  */
// struct UpStreamReshapeHandle : public UpstreamHandleBase {
//   static std::unordered_map<std::string, OpPassThroughConfig>& GetOpPassThroughConfigMap() {
//     static std::unordered_map<std::string, OpPassThroughConfig> allowed_passthrough_ops;
//     static std::once_flag allowed_ops_init;
//     std::call_once(allowed_ops_init, []() {
//       allowed_passthrough_ops.insert({
//           // Things to consider when more operators are added here:
//           // 1. Whether the operator is safe to pass through in term of compute equivalence.
//           //    If optype is not enough to guarantee the equivalence, we need to add a customized pre-check function
//           //    (as LayerNormalization did).
//           // 2. Whether the outputs have the same dim changes if Gather node moves before that operator.
//           // 3. Should all inputs be allowed when track back further (bottom-up);
//           //    if not, add the input index restriction as MatMul did.
//           {GetFullQualifiedOpName("Add", kOnnxDomain),
//            OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_14_13_7_6_1)},
//           {GetFullQualifiedOpName("BiasGelu", kMSDomain),
//            OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
//           {GetFullQualifiedOpName("BitmaskBiasDropout", kMSDomain),
//            OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
//           {GetFullQualifiedOpName("Cast", kOnnxDomain),
//            OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_13_9_6_1)},
//           {GetFullQualifiedOpName("Div", kOnnxDomain),
//            OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_14_13_7_6_1)},
//           {GetFullQualifiedOpName("Dropout", kOnnxDomain),
//            OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_13_12_10_7_6_1)},
//           {GetFullQualifiedOpName("Gelu", kMSDomain),
//            OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
//           {// Be noted, this is our own implementation of ONNX domain op.
//            GetFullQualifiedOpName("LayerNormalization", kOnnxDomain),
//            OpPassThroughConfig({0}, std::make_shared<ReductionOpPassThroughActor>(), opset_1)},
//           {GetFullQualifiedOpName("MatMul", kOnnxDomain),
//            OpPassThroughConfig({}, std::make_shared<MatMulPassThroughActor>(), opset_13_9_1)},
//           {GetFullQualifiedOpName("Reshape", kOnnxDomain),
//            OpPassThroughConfig({0}, std::make_shared<ReshapePassThroughActor>(), opset_14_13_5_1)},
//           {GetFullQualifiedOpName("Softmax", kOnnxDomain),
//            OpPassThroughConfig({0}, std::make_shared<ReductionOpPassThroughActor>(), opset_13_11_1)},
//           {GetFullQualifiedOpName("Transpose", kOnnxDomain),
//            OpPassThroughConfig({}, std::make_shared<TransposePassThroughActor>(), opset_13_1)},
//       });
//     });

//     return allowed_passthrough_ops;
//   }

//   UpStreamReshapeHandle(Graph& graph, std::deque<std::unique_ptr<UpstreamOperatorInfoBase>>& queue,
//                         const std::string& node_name, const logging::Logger& logger)
//       : UpstreamHandleBase(graph, queue, node_name, logger) {
//   }

//   bool Upstream(Node& current_node, std::unique_ptr<UpstreamOperatorInfoBase>& info) override;

//  private:
//   /**
//    * @brief Pass through Slicing op from current_node's output to its specific input.
//    *
//    * Propagate the slicing operation into current_node's current_input_index-th input, e.g. a slicing op is inserted
//    * between current_node's current_input_index-th input and current_node. For example, if current_node is Add,
//    * and slice_node is a Gather(axis=1, indices=[1]):
//    *
//    *    input_0 [M, N, K]    input_1 [M, N, K]
//    *                \        /
//    *                Add [M, N, K]
//    *                     |
//    *            Gather0(axis=1, indices=[1])
//    *                     |
//    *              output [M, 1, K]
//    *
//    * After the pass through, the graph will be:
//    *
//    *   input_0 [M, N, K]                      input_1 [M, N, K]
//    *                \                                /
//    *     Gather1(axis=1, indices=[1])       Gather2(axis=1, indices=[1])
//    *                     \                       /
//    *                       \                   /
//    *                          \             /
//    *                           Add [M, N, K]
//    *                               |
//    *                       Gather0(axis=1, indices=[1])
//    *                               |
//    *                         output [M, 1, K]
//    *
//    * Be noted: Gather1 and Gather2 are inserted on Add's two inputs.
//    * Gather0's removal and Add's output shape update is done in RemoveOriginSlicingOp.
//    *
//    *
//    * @param graph Graph to iterate.
//    * @param slice_node Slicing op node the takes current_node's output as input.
//    * @param current_node Current node.
//    * @param current_node_input_index The current_node_input_index-th input to propagate the Slice op pass through.
//    * @param info slice_node's SliceInfo.
//    * @param logger Logger.
//    * @param new_axis The new axis (for the new Slice op) upon current_node's original current_node_input_index-th input.
//    * @return  SliceInfo for new created slicing op.
//    */
//   SliceInfo PropagateSlicingForInput(Graph& graph, Node& slice_node, Node& current_node, int current_node_input_index,
//                                      SliceInfo& info, int new_axis, const logging::Logger& logger);

//   /**
//    * @brief Remove the origin slicing op (for example Gather/GatherND) and update shapes.
//    *
//    * In the above example, the graph will be cleaned up to:
//    *   input_0 [M, N, K]                      input_1 [M, N, K]
//    *                \                                /
//    *     Gather1(axis=1, indices=[1])       Gather2(axis=1, indices=[1])
//    *                     \                       /
//    *                       \                   /
//    *                          \             /
//    *                           Add [M, 1, K]
//    *                               |
//    *                               |
//    *                         output [M, 1, K]
//    *
//    * Be noted: Gather0 is removed, Add's output shape is updated.
//    *
//    * @param graph Graph to iterate.
//    * @param slice_node Slicing op node the takes current_node's output as input.
//    * @param current_node Current node.
//    * @param logger Logger.
//    * @param info slice_node's SliceInfo.
//    * @return
//    */
//   Status RemoveOriginSlicingOp(Graph& graph, Node& slice_node, Node& current_node,
//                                const logging::Logger& logger, SliceInfo& info);
// };

// bool UpStreamReshapeHandle::Upstream(Node& current_node, std::unique_ptr<UpstreamOperatorInfoBase>& info) {
//   SliceInfo* info_ptr = dynamic_cast<SliceInfo*>(info.get());
//   ORT_ENFORCE(info_ptr != nullptr, "Invalid SliceInfo.");
//   Node& slice_node = *info_ptr->node_ptr;
//   const std::string op_type = GetFullQualifiedOpName(current_node.OpType(), current_node.Domain());
//   if (GetOpPassThroughConfigMap().count(op_type)) {
//     auto& pass_through_config = GetOpPassThroughConfigMap().at(op_type);
//     LOG_DEBUG_INFO(logger_, "Enter reorder handle for node " + current_node.Name() + "(" + op_type + ")");

//     if (!graph_utils::IsSupportedOptypeVersionAndDomain(current_node, current_node.OpType(),
//                                                         pass_through_config.opsets, current_node.Domain())) {
//       LOG_DEBUG_INFO(logger_, "Unsupported opset for " + current_node.Name() + "(" + op_type + ") since version: " +
//                                   std::to_string(current_node.SinceVersion()));
//       return false;
//     }

//     if (!EnforceNodeAllInputOutputHaveShapes(current_node)) {
//       LOG_DEBUG_INFO(logger_, "Some inputs/outputs' shape not found for node " + current_node.Name() + "(" +
//                                   op_type + ")");
//       return false;
//     }

//     std::unordered_map<int, int> candidate_input_indices;
//     bool input_has_dim_1_for_axis = false;
//     if (!pass_through_config.actor->PreCheck(graph_, current_node, *info_ptr, pass_through_config.input_indices, logger_,
//                                              candidate_input_indices, input_has_dim_1_for_axis)) {
//       LOG_DEBUG_INFO(logger_, "Pre-check failed for " + current_node.Name() + "(" + op_type + ")");
//       return false;
//     }

//     if (candidate_input_indices.empty()) {
//       LOG_DEBUG_INFO(logger_, "Skip handling current node " + current_node.Name() + "(" + op_type +
//                                   ") because the requirement is not met.");
//       return false;
//     }

//     // Be noted, once we reach this point after PreCheck, graph modification started, any failure after this should
//     // be reported as ERROR.
//     std::vector<std::unique_ptr<SliceInfo>> populated_slicing_infos;  // Slicing infos that are populated into current_node's inputs.
//     populated_slicing_infos.reserve(candidate_input_indices.size());
//     std::unordered_map<int, SliceInfo> new_gather_infos;
//     for (auto pair : candidate_input_indices) {
//       auto input_index = pair.first;  // input index of current_node
//       int new_axis = pair.second;     // new axis of current_node's input to be sliced
//       SliceInfo gather_info = PropagateSlicingForInput(graph_, slice_node, current_node, input_index, *info_ptr, new_axis,
//                                                        logger_);

//       ORT_ENFORCE(gather_info.node_ptr, "New added gather node should not be null.");
//       populated_slicing_infos.push_back(std::make_unique<SliceInfo>(gather_info));
//       new_gather_infos.insert({{input_index, gather_info}});
//     }

//     int index_of_output =
//         optimizer_utils::IndexOfNodeOutput(current_node, *slice_node.InputDefs()[info_ptr->GetDataInputIndex()]);
//     ORT_ENFORCE(RemoveOriginSlicingOp(graph_, slice_node, current_node, logger_, *info_ptr).IsOK());
//     if (!pass_through_config.actor->PostProcess(graph_, current_node, index_of_output, info_ptr->non_negative_axis,
//                                                 info_ptr->is_scalar_slice, input_has_dim_1_for_axis,
//                                                 info_ptr->output_dim_on_axis,
//                                                 entry_node_name_, new_gather_infos,
//                                                 logger_)) {
//       ORT_THROW("Post-process failed for " + current_node.Name() + "(" + op_type + ")");
//     }

//     queue_.insert(queue_.end(), std::make_move_iterator(populated_slicing_infos.begin()), std::make_move_iterator(populated_slicing_infos.end()));
//     return true;
//   } else {
//     LOG_DEBUG_INFO(logger_, "op_type not supported for " + current_node.Name() + "(" + op_type + ")");
//     return false;
//   }
// }

// SliceInfo UpStreamReshapeHandle::PropagateSlicingForInput(Graph& graph,
//                                                           Node& slice_node,
//                                                           Node& current_node,
//                                                           int current_node_input_index,
//                                                           SliceInfo& info,
//                                                           int new_axis,
//                                                           const logging::Logger& logger) {
//   LOG_DEBUG_INFO(logger, "PropagateSlicingForInput for Node " + slice_node.Name() + "(" + slice_node.OpType() +
//                              ") with input index " + std::to_string(current_node_input_index) + ", keep_dim = " +
//                              std::to_string(!info.is_scalar_slice));

//   InlinedVector<NodeArg*> input_args;
//   input_args.reserve(slice_node.InputDefs().size());
//   // The first slice op's data input should be current_node's current_node_input_index-th input.
//   // For some cases when rank changes, slice op's slice input should also be adapted.
//   input_args.push_back(current_node.MutableInputDefs()[current_node_input_index]);
//   for (size_t i = 1; i < slice_node.InputDefs().size(); ++i) {
//     input_args.push_back(slice_node.MutableInputDefs()[i]);
//   }

//   // Update the axis attribute if new_axis is not same with the original slicing axis (which happens when data
//   // layout got changed by Transpose or Reshape ops)
//   onnxruntime::NodeAttributes attributes = slice_node.GetAttributes();
//   if (info.non_negative_axis != new_axis) {
//     attributes[info.axis_attr_name] =
//         ONNX_NAMESPACE::MakeAttribute(info.axis_attr_name, static_cast<int64_t>(new_axis));
//   }

//   InlinedVector<NodeArg*> output_args;
//   output_args.push_back(
//       &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(info.entry_slice_arg_name),
//                                 current_node.MutableInputDefs()[current_node_input_index]->TypeAsProto()));

//   /* new node input index to connect to current_node's input node*/
//   int new_slice_input_index_to_connect = info.GetDataInputIndex();
//   /* new node output index to connect to current_node*/
//   int new_slice_output_index_to_connect = info.GetOutputIndex();
//   Node* new_slice_node = InsertIntermediateNodeOnDestInput(graph, current_node,
//                                                            current_node_input_index,
//                                                            new_slice_input_index_to_connect,
//                                                            new_slice_output_index_to_connect,
//                                                            graph.GenerateNodeName(info.entry_slice_arg_name),
//                                                            slice_node.OpType(),
//                                                            "Duplicated Gather node",
//                                                            input_args,
//                                                            output_args,
//                                                            attributes,
//                                                            slice_node.Domain(),
//                                                            logger);

//   new_slice_node->SetExecutionProviderType(slice_node.GetExecutionProviderType());

//   // Set correct shape for new created node.
//   auto new_slice_out_arg = new_slice_node->MutableOutputDefs()[new_slice_output_index_to_connect];
//   int reversed_axis = new_axis - new_slice_out_arg->Shape()->dim_size();
//   UpdateSliceOutputShape(*new_slice_out_arg, reversed_axis, info.output_dim_on_axis);
//   auto new_slice_info = SliceInfo(new_slice_node, info.is_scalar_slice, info.axis_attr_name, new_axis);
//   new_slice_info.entry_slice_arg_name = info.entry_slice_arg_name;
//   return new_slice_info;
// }

// Status UpStreamReshapeHandle::RemoveOriginSlicingOp(Graph& graph, Node& slice_node, Node& current_node,
//                                                     const logging::Logger& logger, SliceInfo& info) {
//   LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp target_node " + current_node.Name() + "(" + current_node.OpType() +
//                              ") slice_node " + slice_node.Name() + "(" + slice_node.OpType() + "), keep_dim = " +
//                              std::to_string(!(info.is_scalar_slice)));

//   auto slice_input_arg = slice_node.MutableInputDefs()[info.GetDataInputIndex()];
//   int slice_input_rank = slice_input_arg->Shape()->dim_size();
//   int output_index = optimizer_utils::IndexOfNodeOutput(current_node, *slice_input_arg);
//   auto slice_op_output_arg = slice_node.MutableOutputDefs()[info.GetOutputIndex()];

//   // Loop all outputs of target node, update the shape accordingly.
//   // For elementwise ops like (LayerNorm/Dropout/Add), we should handle all outputs.
//   // If some output rank is lower than sliced axis, we should just ignore it (the correctness is guaranteed by devs
//   // who adds more operator coverage in the pass through).
//   for (size_t i = 0; i < current_node.MutableOutputDefs().size(); ++i) {
//     UpdateSliceOutputShape(*current_node.MutableOutputDefs()[i], info.non_negative_axis - slice_input_rank,
//                            info.output_dim_on_axis);
//   }
//   LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp Replace all usage of output " + slice_op_output_arg->Name() + ":0" +
//                              " with " + current_node.MutableOutputDefs()[output_index]->Name() + ":" +
//                              std::to_string(output_index));

//   graph_utils::ReplaceDownstreamNodeInput(graph, slice_node, info.GetOutputIndex() /*output_idx*/, current_node,
//                                           output_index /*replacement_output_idx*/);
//   auto gather_origin_consumer_nodes = graph.GetConsumerNodes(slice_op_output_arg->Name());
//   std::vector<Node*> slice_op_consumers;
//   slice_op_consumers.reserve(gather_origin_consumer_nodes.size());
//   for (auto& consumer_node : gather_origin_consumer_nodes) {
//     slice_op_consumers.push_back(graph.GetNode(consumer_node->Index()));
//     LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp Gather's consumer node " + consumer_node->Name() + "(" +
//                                consumer_node->OpType() + ")");
//   }
//   graph.UpdateConsumerNodes(current_node.OutputDefs()[output_index]->Name(), slice_op_consumers);

//   graph.UpdateConsumerNodes(slice_op_output_arg->Name(), {});
//   graph.RemoveNode(slice_node.Index());

//   return Status::OK();
// }

// }  // namespace

// std::unique_ptr<UpstreamOperatorInfoBase> UpStreamReshapeHandle::IsSupportedForUpstream(
//     Graph& graph,
//     Node& node,
//     const logging::Logger& logger) const {
//   std::optional<ReshapeInfo> gather_info;
//   // Same ideas might apply for GatherElements, Slice, Split, etc..
//   gather_info = IsSupportedReshape(graph, node, logger);
//   if (!gather_info.has_value()) {
//     return std::make_unique<SliceInfo>(gather_info.value());
//   }

//   return nullptr;
// }

// std::optional<ReshapeInfo> UpStreamReshapeHandle::IsSupportedReshape(Graph& /*graph*/, Node& node,
//                                                                      const logging::Logger& logger) const {
//   if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", {1, 5, 13, 14}, kOnnxDomain)) {
//     return std::nullopt;
//   }

//   auto data_shape = node.MutableInputDefs()[0]->Shape();
//   auto reshape_out_shape = node.MutableOutputDefs()[0]->Shape();
//   if (data_shape == nullptr || reshape_out_shape == nullptr) {
//     LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to undefined shape.");
//     return std::nullopt;
//   }

//   const auto data_rank = data_shape->dim_size();
//   if (data_rank != 3) {
//     LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to data rank != 3.");
//     return std::nullopt;
//   }

//   if (!graph.IsConstantInitializer(node.InputDefs()[1]->Name(), /* check_outer_scope */ false)) {
//     LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due target shape is non-constant initializer.");
//     continue;
//   }

//   InlinedVector<int64_t> new_shape_const_values;
//   optimizer_utils::AppendTensorFromInitializer(graph, *node.InputDefs()[1], new_shape_const_values, true);
//   if (new_shape_const_values.size() != 2 || new_shape_const_values[0] != -1) {
//     LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to target shape is not merging first two dims.");
//     return std::nullopt;
//   }

//   return ReshapeInfo(&node, true);
// }

// std::unique_ptr<UpstreamHandleBase> UpStreamGatherGraphTransformer::CreateUpstreamHandle(
//     Graph& graph, const Node& node,
//     std::deque<std::unique_ptr<UpstreamOperatorInfoBase>>& queue,
//     const logging::Logger& logger) const {
//   return std::make_unique<SliceOperationReorderHandle>(graph, queue, node.Name(), logger);
// }

// }  // namespace onnxruntime

// #endif
