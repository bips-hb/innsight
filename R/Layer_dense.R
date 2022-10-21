
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

  # rel_output   [batch_size, dim_out, model_out]
  #
  #   output       [batch_size, dim_in, model_out]
  get_input_relevances = function(rel_output,
                                  rule_name = "simple",
                                  rule_param = NULL) {
    if (is.null(rule_param)) {
      if (rule_name == "epsilon") {
        rule_param <- 0.001
      } else if (rule_name == "alpha_beta") {
        rule_param <- 0.5
      }
    }

    input <- self$input$unsqueeze(3)
    z <- self$preactivation$unsqueeze(3)

    # Get stabilizer
    eps <- self$get_stabilizer()

    if (rule_name == "simple") {
      z <- z + (z == 0) * eps

      rel_input <-
        self$get_gradient(rel_output / z, self$W) * input
    } else if (rule_name == "epsilon") {
      z <- z + rule_param * (torch_sgn(z) + (z == 0))
      rel_input <-
        self$get_gradient(rel_output / z, self$W) * input
    } else if (rule_name == "alpha_beta") {
      out_part <- self$get_pos_and_neg_outputs(self$input, use_bias = TRUE)

      # Apply the simple rule for each part:
      # - positive part
      z <-
        rel_output / (out_part$pos + (out_part$pos == 0) * eps + torch_sgn(out_part$pos) * eps)$unsqueeze(3)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_pos <- t1 * (input * (input > 0)) + t2 * (input * (input <= 0))

      # - negative part
      z <-
        rel_output / (out_part$neg + torch_sgn(out_part$neg) * eps - (out_part$neg == 0) * eps)$unsqueeze(3)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- t1 * (input * (input <= 0)) + t2 * (input * (input > 0))

      # calculate over all relevance for the lower layer
      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
    }


    rel_input
  },

  #   mult_output   [batch_size, dim_out, model_out]
  #
  #   output        [batch_size, dim_in, model_out]
  get_input_multiplier = function(mult_output, rule_name = "rescale") {

    # --------------------- Non-linear part---------------------------
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      # Get stabilizer
      eps <- self$get_stabilizer()

      if (rule_name == "rescale") {
        delta_output <- (self$output - self$output_ref)$unsqueeze(3)
        delta_preact <-
          (self$preactivation - self$preactivation_ref)$unsqueeze(3)

        # Near zero needs special treatment
        mask <- (abs(delta_preact) < eps) * (1.0)
        x <-  mask * (self$preactivation + self$preactivation_ref)$unsqueeze(3) / 2
        x$requires_grad <- TRUE

        y <- sum(self$activation_f(x))
        grad <- autograd_grad(y, x)[[1]]

        nonlin_mult <- (1 - mask) * (delta_output / delta_preact) +
          mask * grad

        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult
      } else if (rule_name == "reveal_cancel") {
        act <- self$activation_f
        x <- self$preactivation
        x_ref <- self$preactivation_ref
        delta_x <- self$get_pos_and_neg_outputs(self$input - self$input_ref)

        delta_output_pos <-
          ( 0.5 * (act(x_ref + delta_x$pos) - act(x_ref)) +
            0.5 * (act(x) - act(x_ref + delta_x$neg)))

        delta_output_neg <-
          ( 0.5 * (act(x_ref + delta_x$neg) - act(x_ref)) +
            0.5 * (act(x) - act(x_ref + delta_x$pos)))

        mult_pos <-
          mult_output * (delta_output_pos / (delta_x$pos + (delta_x$pos == 0) * eps))$unsqueeze(3)
        mult_neg <-
          mult_output * (delta_output_neg / (delta_x$neg - (delta_x$neg == 0) * eps))$unsqueeze(3)
      }
    }

    # -------------- Linear part -----------------------

    # input        [batch_size, dim_in]
    # delta_input  [batch_size, dim_in, 1]
    delta_input <- (self$input - self$input_ref)$unsqueeze(3)

    # mult_input    [batch_size, model_out, dim_in]
    mult_input <-
      self$get_gradient(mult_pos, self$W * (self$W > 0)) * (delta_input > 0) +
      self$get_gradient(mult_pos, self$W * (self$W < 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, self$W * (self$W > 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, self$W * (self$W < 0)) * (delta_input > 0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), self$W) *
        (delta_input == 0)

    mult_input
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
