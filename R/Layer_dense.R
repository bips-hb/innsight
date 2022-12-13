
###############################################################################
#                               Dense Layer
###############################################################################

dense_layer <- nn_module(
  classname = "Dense_Layer",
  inherit = InterpretingLayer,

  #
  # weight: [out_features, in_features]
  # bias  : [out_features]
  #
  initialize = function(weight, bias, activation_name,
                        dim_in = NULL,
                        dim_out = NULL,
                        dtype = "float") {
    if (is.null(dim_in)) {
      self$input_dim <- dim(weight)[2]
    } else {
      self$input_dim <- dim_in
    }
    if (is.null(dim_out)) {
      self$output_dim <- dim(weight)[1]
    } else {
      self$output_dim <- dim_out
    }
    self$get_activation(activation_name)

    # Check if weight is already a tensor
    if (!inherits(weight, "torch_tensor")) {
      self$W <- torch_tensor(weight)
    } else {
      self$W <- weight
    }
    # Check if bias is already a tensor
    if (!inherits(bias, "torch_tensor")) {
      self$b <- torch_tensor(bias)
    } else {
      self$b <- bias
    }

    self$set_dtype(dtype)
  },

  # x       : [batch_size, in_features]
  #
  # output  : [batch_size, out_features]
  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE, ...) {
    if (save_input) {
      self$input <- x
    }
    preactivation <- nnf_linear(x, self$W, self$b)
    if (save_preactivation) {
      self$preactivation <- preactivation
    }
    output <- self$activation_f(preactivation)
    if (save_output) {
      self$output <- output
    }

    output
  },

  # x_ref   : [1, in_features]
  #
  # output  : [1, out_features]
  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE, ...) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    preactivation_ref <- nnf_linear(x_ref, self$W, self$b)
    if (save_preactivation) {
      self$preactivation_ref <- preactivation_ref
    }
    output_ref <- self$activation_f(preactivation_ref)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  #   grad_out   [batch_size, dim_out, model_out]
  #   weight     [dim_out, dim_in]
  #
  #   output  [batch_size, dim_in, model_out]
  get_gradient = function(grad_out, weight) {
    grad_in <- torch_matmul(weight$t(), grad_out)

    grad_in
  },

  get_pos_and_neg_outputs = function(input, use_bias = FALSE) {
    output <- NULL

    if (use_bias == TRUE) {
      b_pos <- self$b * (self$b > 0) * 0.5
      b_neg <- self$b * (self$b <= 0) * 0.5
    } else {
      b_pos <- NULL
      b_neg <- NULL
    }

    W <- self$W

    output$pos <-
      nnf_linear(input * (input > 0), W * (W > 0), bias = b_pos) +
      nnf_linear(input * (input <= 0), W * (W <= 0), bias = b_pos)

    output$neg <-
      nnf_linear(input * (input > 0), W * (W <= 0), bias = b_neg) +
      nnf_linear(input * (input <= 0), W * (W > 0), bias = b_neg)

    output
  }
)
