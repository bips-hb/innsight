
#' @include Layer.R
#' @export
dense_layer <- torch::nn_module(
  classname = "Dense_Layer",
  inherit = Layer,

  #
  # weight: [out_features, in_features]
  # bias  : [out_features]
  #
  initialize = function(weight, bias, activation_name, dtype = "float") {
    self$input_dim <- dim(weight)[2]
    self$output_dim <- dim(weight)[1]
    self$get_activation(activation_name)

    # Check if weight is already a tensor
    if (!inherits(weight, "torch_tensor")) {
      self$W <- torch::torch_tensor(weight)
    }
    else {
      self$W <- weight
    }
    # Check if bias is already a tensor
    if (!inherits(bias, "torch_tensor")) {
      self$b <- torch::torch_tensor(bias)
    }
    else {
      self$b <- bias
    }

    self$set_dtype(dtype)
  },

  set_dtype = function(dtype) {
    if (dtype == "float") {
      self$W <- self$W$to(torch::torch_float())
      self$b <- self$b$to(torch::torch_float())
    }
    else if (dtype == "double") {
      self$W <- self$W$to(torch::torch_double())
      self$b <- self$b$to(torch::torch_double())
    }
    else {
      stop(sprintf("Unknown argument for 'dtype' : %s . Use 'float' or 'double' instead"))
    }
    self$dtype <- dtype
  },

  #
  # x: [batch_size, in_features]
  #
  forward = function(x) {

    self$input <- x
    self$preactivation <- torch::nnf_linear(x, self$W, self$b)
    self$output <- self$activation_f(self$preactivation)

    self$output
  },

  #
  # x_ref: Tensor of size [1,in_features]
  #
  update_ref = function(x_ref) {

    self$input_ref <- x_ref
    self$preactivation_ref <- torch::nnf_linear(x_ref, self$W, self$b)
    self$output_ref <- self$activation_f(self$preactivation_ref)

    self$output_ref
  },

  #
  #   rel_output   [batch_size, dim_out, model_out]
  #
  #   output       [batch_size, dim_in, model_out]
  #
  get_input_relevances = function(rel_output, rule_name = 'simple', rule_param = NULL) {

    if (is.null(rule_param)) {
      if (rule_name == "epsilon"){
        rule_param = 0.001
      }
      else if (rule_name == "alpha_beta") {
        rule_param = 0.5
      }
    }

    input <- self$input$unsqueeze(3)
    z <- self$preactivation$unsqueeze(3)

    if(rule_name == "simple"){
      z <- z + (z == 0) * 1e-16

      rel_input <-
        self$get_gradient(rel_output / z, self$W) * input

    }
    else if(rule_name == "epsilon"){
      z <- z + rule_param * torch::torch_sgn(z) + (z == 0) * 1e-16
      rel_input <-
        self$get_gradient(rel_output / z, self$W) * input
    }
    else if(rule_name == "alpha_beta"){

      out_part <- self$get_pos_and_neg_outputs(self$input)

      # Apply the simple rule for each part:
      # - positive part
      z <- rel_output / ( out_part$pos + (out_part$pos == 0) * 1e-16 )$unsqueeze(3)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_pos <- t1 *  (input * (input > 0)) + t2 * (input * (input <= 0))

      # - negative part
      z <- rel_output / ( out_part$neg + (out_part$neg == 0) * 1e-16 )$unsqueeze(3)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- t1 * (input * (input <= 0)) + t2 * (input * (input > 0))

      # calculate over all relevance for the lower layer
      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
    }
    #else if(rule_name == "ww"){
    #  relevance <- torch_matmul(( torch_transpose( (torch_square(weights) / torch_sum(torch_transpose(torch_square(weights),1,2),2,keepdim=FALSE)) ,1,2)) , relevance)
    #}


    rel_input
  },

  #
  #   mult_output   [batch_size, dim_out, model_out]
  #
  #   output        [batch_size, dim_in, model_out]
  #
  get_input_multiplier = function(mult_output, rule_name = "rescale") {

    #
    # --------------------- Non-linear part---------------------------
    #
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      if (rule_name == "rescale") {
        delta_output <- (self$output - self$output_ref)$unsqueeze(3)
        delta_preact <- (self$preactivation - self$preactivation_ref)$unsqueeze(3)

        nonlin_mult <- delta_output / (delta_preact + 1e-16 * (delta_preact == 0))

        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult

      }
      else if (rule_name == "reveal_cancel") {
        act <- self$activation_f
        x <- self$preactivation
        x_ref <- self$preactivation_ref
        delta_x <- self$get_pos_and_neg_outputs(self$input - self$input_ref)

        delta_output_pos <-
          0.5 * (act(x_ref + delta_x$pos) - act(x_ref)) +
          0.5 * (act(x) - act(x_ref + delta_x$neg))

        delta_output_neg <-
          0.5 * (act(x_ref + delta_x$neg) - act(x_ref)) +
          0.5 * (act(x) - act(x_ref + delta_x$pos))

        mult_pos <- mult_output * (delta_output_pos / (delta_x$pos + 1e-16))$unsqueeze(3)
        mult_neg <- mult_output * (delta_output_neg / (delta_x$neg - 1e-16))$unsqueeze(3)
      }
    }

    #
    # -------------- Linear part -----------------------
    #

    # input        [batch_size, dim_in]
    # delta_input  [batch_size, dim_in, 1]
    delta_input <- (self$input - self$input_ref)$unsqueeze(3)

    # mult_input    [batch_size, model_out, dim_in]
    mult_input <-
      self$get_gradient(mult_pos, self$W * (self$W > 0)) * (delta_input > 0) +
      self$get_gradient(mult_pos, self$W * (self$W < 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, self$W * (self$W > 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, self$W * (self$W < 0)) * (delta_input > 0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), self$W) * (delta_input == 0)

    mult_input
  },

  #
  #   grad_out   [batch_size, dim_out, model_out]
  #   weight     [dim_out, dim_in]
  #
  #   output  [batch_size, dim_in, model_out]
  #
  get_gradient = function(grad_out, weight) {
    grad_in <- torch::torch_matmul(weight$t(), grad_out)

    grad_in
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

    W <- self$W

    output$pos <-
      torch::nnf_linear(input * (input > 0), W * (W > 0), bias = b_pos) +
      torch::nnf_linear(input * (input < 0), W * (W < 0), bias = b_pos)

    output$neg <-
      torch::nnf_linear(input * (input > 0), W * (W < 0), bias = b_neg) +
      torch::nnf_linear(input * (input < 0), W * (W > 0), bias = b_neg)

    output
  }
)
