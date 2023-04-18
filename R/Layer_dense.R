
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
                        act_func = NULL,
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
    self$get_activation(activation_name, act_func)

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
  get_gradient = function(grad_out, weight, ...) {
    grad_in <- torch_matmul(weight$t(), grad_out)

    grad_in
  },

  get_pos_and_neg_outputs = function(input, use_bias = FALSE) {
    output <- NULL

    if (use_bias) {
      b_pos <- torch_clamp(self$b, min = 0)
      b_neg <- torch_clamp(self$b, max = 0)
    } else {
      b_pos <- NULL
      b_neg <- NULL
    }

    input_pos <- torch_clamp(input, min = 0)
    input_neg <- torch_clamp(input, max = 0)
    W_pos <- torch_clamp(self$W, min = 0)
    W_neg <- torch_clamp(self$W, max = 0)

    output$pos <-
      nnf_linear(input_pos, W_pos, bias = b_pos) +
      nnf_linear(input_neg, W_neg, bias = NULL)

    output$neg <-
      nnf_linear(input_pos, W_neg, bias = b_neg) +
      nnf_linear(input_neg, W_pos, bias = NULL)

    output
  }
)
