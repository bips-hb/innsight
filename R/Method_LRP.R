

# Arguments:
#   layer           : Instance of the torch module 'dense_layer' from 'Layer_dense.R'
#   rule_name       : Name of the LRP-rule ("simple", "epsilon", "alpha_beta")
#   rule_param      : Parameter of the rule ("simple": no parameter, "epsilon": epsilon value,
#                     set default to 0.001, "alpha_beta": alpha value, set default to 0.5)
#   relevance       : relevance score from the upper layer to the output, torch Tensor
#                   : of size [batch_size, dim_out, model_dim_out]
#
#   output          : torch Tensor of size [batch_size, dim_in, model_dim_out]

lrp_dense <- function(layer, relevance, rule_name = 'simple', rule_param = NULL) {
  relevance <- NULL

  #
  # toDo
  #

  return(relevance)
}


# Arguments:
#   layer           : Instance of the torch module 'conv1d_layer' from 'Layer_conv1d.R'
#   rule_name       : Name of the LRP-rule ("simple", "epsilon", "alpha_beta")
#   rule_param      : Parameter of the rule ("simple": no parameter, "epsilon": epsilon value,
#                     set default to 0.001, "alpha_beta": alpha value, set default to 0.5)
#   relevance       : relevance score from the upper layer to the output, torch Tensor
#                   : of size [batch_size, out_channels, out_length, model_out]
#
#   output          : torch Tensor of size [batch_size, in_channels, in_length, model_out]

lrp_conv1d <- function(layer, relevance, rule_name = 'simple', rule_param = NULL) {
  relevance <- NULL

  #
  # toDo
  #

  return(relevance)
}


# Arguments:
#   layer           : Instance of the torch module 'conv2d_layer' from 'Layer_conv2d.R'
#   rule_name       : Name of the LRP-rule ("simple", "epsilon", "alpha_beta")
#   rule_param      : Parameter of the rule ("simple": no parameter, "epsilon": epsilon value,
#                     set default to 0.001, "alpha_beta": alpha value, set default to 0.5)
#   relevance       : relevance score from the upper layer to the output, torch Tensor
#                   : of size [batch_size, out_channels, out_height, out_width model_out]
#
#   output          : torch Tensor of size [batch_size, in_channels, in_height, in_width, model_out]

lrp_conv1d <- function(layer, relevance, rule_name = 'simple', rule_param = NULL) {
  relevance <- NULL

  #
  # toDo
  #

  return(relevance)
}
