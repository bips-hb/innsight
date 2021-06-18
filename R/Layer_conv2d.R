
#' @include Layer.R
#' @export
conv2d_layer <- torch::nn_module(
  classname = "Conv2D_Layer",
  inherit = Layer,

  #
  # weight: [out_channels, in_channels, kernel_height, kernel_width]
  # bias  : [out_channels]
  #
  initialize = function(weight,
                        bias,
                        dim_in,
                        dim_out,
                        stride = 1,
                        padding = 0,
                        dilation = 1,
                        activation_name = 'linear', ...) {

    # [in_channels, in_height, in_width]
    self$input_dim <- dim_in
    # [out_channels, out_height, out_width]
    self$output_dim <- dim_out
    self$in_channels <- dim(weight)[2]
    self$out_channels <- dim(weight)[1]
    self$kernel_size <- dim(weight)[-c(1,2)]

    self$W <- torch_tensor(weight, dtype = torch_float())
    self$b <- torch_tensor(bias, dtype = torch_float())

    self$stride <- stride
    self$padding <- padding
    self$dilation <- dilation

    self$get_activation(activation_name, ...)
  },

  #
  # x : Tensor [minibatch, in_channels, in_height, in_width]
  #
  forward = function(x) {
    self$input <- x
    self$preactivation <- torch::nnf_conv2d(x, self$W, self$b, self$stride, self$padding, self$dilation)
    if (self$activation_name == 'linear') {
      self$output <- self$preactivation
    }
    else {
      self$output <- self$activation_f(self$preactivation)
    }
    self$output
  },

  #
  # x_ref: Tensor of size [in_channels, in_height, in_width]
  #
  update_ref = function(x_ref) {

    self$input_ref <- x_ref
    self$preactivation_ref <- torch::nnf_conv2d(x_ref, self$W, self$b, self$stride, self$padding, self$dilation)
    if (self$activation_name == 'linear') {
      self$output_ref <- self$preactivation_ref
    }
    else {
      self$output_ref <- self$activation_f(self$preactivation_ref)
    }

    self$output_ref
  },

  # Arguments:
  #   rule_name       : Name of the LRP-rule ("simple", "epsilon", "alpha_beta")
  #   rule_param      : Parameter of the rule ("simple": no parameter, "epsilon": epsilon value,
  #                     set default to 0.001, "alpha_beta": alpha value, set default to 0.5)
  #   relevance       : relevance score from the upper layer to the output, torch Tensor
  #                   : of size [batch_size, out_channels, out_height, out_width, model_out]
  #
  #   output          : torch Tensor of size [batch_size, in_channels, in_height, in_width, model_out]

  get_input_relevances = function(relevance, rule_name = 'simple', rule_param = NULL) {

    if (rule_name == 'simple') {
      z <-  self$preactivation
      # add a small stabilizer
      z <- z + (z==0)*1e-16
      rel_lower <- self$get_gradient(relevance / z$unsqueeze(5), self$W)
      rel_lower <- torch::torch_mul(rel_lower, self$input$unsqueeze(5))
    }
    else if (rule_name == 'epsilon') {
      # set default parameter
      if (is.null(rule_param)) {
        epsilon <- 0.001
      }
      else {
        epsilon <- rule_param
      }

      z <-  self$preactivation
      z <- z + epsilon * torch::torch_sgn(z)
      rel_lower <- self$get_gradient(relevance / z$unsqueeze(5), self$W)
      rel_lower <- torch::torch_mul(rel_lower, self$input$unsqueeze(5))
    }
    else if (rule_name == 'alpha_beta') {
      # set default parameter
      if (is.null(rule_param)) {
        alpha <- 0.5
      }
      else {
        alpha <- rule_param
      }

      output_partition <- self$get_pos_and_neg_outputs(self$input, use_bias = TRUE)

      z <- relevance / ( output_partition$pos + (output_partition$pos == 0) * 1e-16 )$unsqueeze(5)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      input <- self$input$unsqueeze(5)
      rel_pos <- torch::torch_mul(t1, (input * (input > 0))) +
                 torch::torch_mul(t2, (input * (input <= 0)))


      z <- relevance / ( output_partition$neg + (output_partition$neg == 0) * 1e-16 )$unsqueeze(5)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- torch::torch_mul(t1, (input * (input <= 0))) +
                 torch::torch_mul(t2, (input * (input > 0)))


      # calculate over all relevance for the layer
      rel_lower <- rel_pos * alpha + rel_neg * (1 - alpha)

    }

    rel_lower
  },

  get_input_multiplier = function(multiplier, rule_name = "rescale") {

    #
    # --------------------- Non-linear part---------------------------
    #
    mult_pos <- multiplier
    mult_neg <- multiplier
    if (self$activation_name != "linear") {
      if (rule_name == "rescale") {

        # output       [batch_size, out_channels, out_height, out_width]
        # delta_output [batch_size, out_channels, out_height, out_width, 1]
        delta_output <- (self$output - self$output_ref)$unsqueeze(5)
        delta_preact <- (self$preactivation - self$preactivation_ref)$unsqueeze(5)

        nonlin_mult <- delta_output / (delta_preact + 1e-16 * (delta_preact == 0))

        # multiplier    [batch_size, out_channels, out_height, out_width, model_out]
        # nonlin_mult   [batch_size, out_channels, out_height, out_width, 1]
        mult_pos <- multiplier * nonlin_mult
        mult_neg <- multiplier * nonlin_mult

      }
      else if (rule_name == "reveal_cancel") {

        pos_and_neg_output <- self$get_pos_and_neg_outputs(self$input - self$input_ref)

        delta_x_pos <- pos_and_neg_output$pos
        delta_x_neg <- pos_and_neg_output$neg

        act <- self$activation_f
        x <- self$preactivation
        x_ref <- self$preactivation_ref

        delta_output_pos <-
          0.5 * (act(x_ref + delta_x_pos) - act(x_ref)) +
          0.5 * (act(x) - act(x_ref + delta_x_neg))

        delta_output_neg <-
          0.5 * (act(x_ref + delta_x_neg) - act(x_ref)) +
          0.5 * (act(x) - act(x_ref + delta_x_pos))

        mult_pos <- multiplier * (delta_output_pos / (delta_x_pos + 1e-16))$unsqueeze(5)
        mult_neg <- multiplier * (delta_output_neg / (delta_x_neg - 1e-16))$unsqueeze(5)
      }
    }

    #
    # -------------- Linear part -----------------------
    #

    # input        [batch_size, in_channels, in_height, in_width]
    # delta_input  [batch_size, in_channels, in_height, in_width, 1]
    delta_input <- (self$input - self$input_ref)$unsqueeze(5)

    # weight      [out_channels, in_channels, kernel_height, kernel_width]
    weight <- self$W

    # multiplier    [batch_size, in_channels, in_height, in_width, model_out]
    multiplier <-
      self$get_gradient(mult_pos, weight * (weight > 0)) * (delta_input > 0) +
      self$get_gradient(mult_pos, weight * (weight < 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, weight * (weight > 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, weight * (weight < 0)) * (delta_input > 0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), weight) * (delta_input == 0)

    multiplier
  },

  get_gradient = function(input, weight) {
    if (length(self$stride) == 1) {
      stride <- c(self$stride, self$stride, 1)
    }
    else if (length(self$stride) == 1) {
      stride <- c(self$stride, 1)
    }
    else {
      stop("Wrong stide format!")
    }
    out <- torch::nnf_conv_transpose2d(input, weight$unsqueeze(5),
                                       bias = NULL,
                                       stride = stride,
                                       padding = self$padding,
                                       dilation = self$dilation)
    out
  },

  get_pos_and_neg_outputs = function(input, use_bias = FALSE) {
    output <- NULL

    if (use_bias == TRUE) {
      b_pos <- self$b * (self$b > 0) * 0.5
      b_neg <- self$b * (self$b <= 0) * 0.5
    }
    else {
      b_pos <- NULL
      b_neg <- NULL
    }

    conv2d <- function(x, W, b) {
      out <- torch::nnf_conv2d(x, W,
                              bias = b,
                              stride = self$stride,
                              padding = self$padding,
                              dilation = self$dilation)
      out
    }

    # input (+) x weight (+) and input (-) x weight (-)
    output$pos <- conv2d(input * (input > 0), self$W * (self$W > 0), b_pos) +
                  conv2d(input * (input < 0), self$W * (self$W < 0), b_pos)

    # input (+) x weight (-) and input (-) x weight (+)
    output$neg <- conv2d(input * (input > 0), self$W * (self$W < 0), b_neg) +
                  conv2d(input * (input < 0), self$W * (self$W > 0), b_neg)

    output
  }
)
