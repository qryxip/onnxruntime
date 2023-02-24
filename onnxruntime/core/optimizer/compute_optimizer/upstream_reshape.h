// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

// #ifdef ENABLE_TRAINING_CORE
// #pragma once

// #include "core/optimizer/compute_optimizer/upstream_operator_base.h"
// // #include "core/optimizer/compute_optimizer/passthrough_actors.h"
// #include "core/optimizer/graph_transformer.h"
// #include "core/optimizer/utils.h"

// namespace onnxruntime {

// /**
//  * @brief Struct to hold the information of the slicing operations.
//  *
//  * Initially, an instance of this class for entry node is created, as the slice op propagates to entry node's inputs,
//  * more instances of this class are created. The propagation stops when the all inputs are not supported to be sliced.
//  */
// struct ReshapeInfo : public UpstreamOperatorInfoBase {
//   static constexpr int kSliceDataInputIndex = 0;
//   static constexpr int kSliceOutputIndex = 0;

//   ReshapeInfo(Node* slice_node,
//               // bool is_slice_scalar,
//               // const std::string& slice_axis_attr_name,
//               // int slice_axis,
//               bool is_entry_node_ptr = false)
//       : UpstreamOperatorInfoBase(slice_node) {
//     axis_attr_name = slice_axis_attr_name;

//     // const NodeArg* input = node_ptr->InputDefs()[kSliceDataInputIndex];
//     const NodeArg* output = node_ptr->OutputDefs()[kSliceOutputIndex];
//     // input_rank = input->Shape()->dim_size();
//     // non_negative_axis = slice_axis < 0 ? input_rank + slice_axis : slice_axis;

//     // if (!is_scalar_slice) {
//     output_dim_on_axis = output->Shape()->dim(non_negative_axis);
//     // }

//     if (is_entry_node_ptr) {
//       entry_slice_arg_name = node_ptr->OutputDefs()[kSliceOutputIndex]->Name();
//     }
//   }

//   int GetDataInputIndex() const {
//     return kSliceDataInputIndex;
//   }

//   int GetOutputIndex() const {
//     return kSliceOutputIndex;
//   }

//   // bool is_scalar_slice;  // whether the slice is a scalar, if it is, after Gather, rank will be reduced by 1.
//   // std::string axis_attr_name;
//   // int non_negative_axis;  // The axis to slice on
//   std::string entry_slice_arg_name;

//   // int input_rank;  // rank of the Gather data input tensor

//   // The dimension of the output tensor on the slicing axis
//   // Be noted: if it is a scalar slicing, this dim will not be set, which means, afterward when use it to update
//   // shapes, that dim at axis will be removed.
//   ONNX_NAMESPACE::TensorShapeProto_Dimension output_dim_on_axis;
// };

// /**
//  * @brief Graph transformer that helps reduce compute FLOP while maintaining mathematically equivalent result.
//  *
//  * 3-D into 2D Reshape (by flatten the first two dims) are the entry operators that trigger the optimization search.
//  *
//  */
// class UpstreamReshape : public UpStreamGraphTransformerBase {
//  public:
//   UpstreamReshape(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
//       : UpStreamGraphTransformerBase("UpstreamReshape", compatible_execution_providers) {}

//   std::unique_ptr<UpstreamOperatorInfoBase> IsSupportedForUpstream(Graph& graph, Node& node,
//                                                                    const logging::Logger& logger) const override;

//   std::unique_ptr<UpstreamHandleBase> CreateUpstreamHandle(Graph& graph, const Node& node,
//                                                            std::deque<std::unique_ptr<UpstreamOperatorInfoBase>>& queue,
//                                                            const logging::Logger& logger) const override;

//  private:
//   std::optional<ReshapeInfo> IsSupportedReshape(Graph& graph, Node& node, const logging::Logger& logger) const;
// };

// }  // namespace onnxruntime
// #endif
