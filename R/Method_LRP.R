

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

lrp_conv2d <- function(layer, relevance, rule_name = 'simple', rule_param = NULL) {

  # get preactivation [batch_size, out_channels, out_height, out_width]
  z <-  layer$preactivation

  if (rule_name == 'simple') {
    # add a small stabilizer
    z <- z + (z==0)*1e-16
    rel_lower <- torch::nnf_conv_transpose1d(input = relevance / z$unsqueeze(5),
                                            weight = layer$W$unsqueeze(5),
                                            bias = NULL,
                                            stride = layer$stride,
                                            padding = layer$padding,
                                            dilation = layer$dilation)
    rel_lower <- torch_mul(rel_lower, layer$input$unsqueeze(5))
  }
  else if (rule_name == 'epsilon') {

    # set default parameter
    if (is.null(rule_param)) {
      epsilon <- 0.001
    }
    else {
      epsilon <- rule_param
    }

    z <- z + epsilon * torch::torch_sgn(z)
    rel_lower <- torch::nnf_conv_transpose1d(input = relevance / z$unsqueeze(5),
                                             weight = layer$W$unsqueeze(5),
                                             bias = NULL,
                                             stride = layer$stride,
                                             padding = layer$padding,
                                             dilation = layer$dilation)
    rel_lower <- torch_mul(rel_lower, layer$input$unsqueeze(5))
  }
  else if (rule_name == 'alpha_beta') {
    # set default parameter
    if (is.null(rule_param)) {
      alpha <- 0.5
    }
    else {
      alpha <- rule_param
    }

    input <- layer$input
    weight <- layer$W
    bias <- layer$b

    input_pos <- input * (input >  0)
    input_neg <- input * (input <= 0)

    weight_pos <- weight * (weight >  0)
    weight_neg <- weight * (weight <= 0)

    bias_pos <- bias * (bias >  0) * 0.5
    bias_neg <- bias * (bias <= 0) * 0.5

    conv_2d <- function(input, weight, bias) {
      out <- torch::nnf_conv2d(input = input, weight = weight, bias = bias,
                               stride = layer$stride, padding = layer$padding,
                               dilation = layer$dilation)
      out
    }

    conv_2d_transpose <- function(input, weight) {
      out <- torch::nnf_conv_transpose2d(input, weight$unsqueeze(5), bias = NULL, stride = layer$stride,
                                  padding = layer$padding, dilation = layer$dilation)
      out
    }

    ## positive relevance
    # input (+) x weight (+)
    z1 <- conv_2d(input_pos, weight_pos, bias_pos)
    # input (-) x weight (-)
    z2 <- conv_2d(input_neg, weight_neg, bias_pos)

    z <- z1 + z2
    z <- relevance / ( z + (z == 0) * 1e-16 )$unsqueeze(5)

    t1 <- conv_2d_transpose(z, weight_pos)
    t2 <- conv_2d_transpose(z, weight_neg)

    rel_pos <- torch_mul(t1, input_pos$unsqueeze(5)) + torch_mul(t2, input_neg$unsqueeze(5))

    ## negative relevance
    # input (-) x weight (+)
    z1 <- conv_2d(input_neg, weight_pos, bias_neg)
    # input (+) x weight (-)
    z2 <- conv_2d(input_pos, weight_neg, bias_neg)

    z <- z1 + z2
    z <- relevance / ( z + (z == 0) * 1e-16 )$unsqueeze(5)

    t1 <- conv_2d_transpose(z, weight_pos)
    t2 <- conv_2d_transpose(z, weight_neg)

    rel_neg <- torch_mul(t1, input_neg$unsqueeze(5)) + torch_mul(t2, input_pos$unsqueeze(5))


    # calculate over all relevance for the layer
    rel_lower <- rel_pos * alpha + rel_neg * (1 - alpha)

  }

  rel_lower
}
