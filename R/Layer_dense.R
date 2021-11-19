#' @include InterpretingLayer.R
#'
NULL

#'
#' Dense Layer of a Neural Network
#'
#'
#' Implementation of a dense Neural Network layer as a torch module
#' where input, preactivation and output values of the last forward pass are
#' stored (same for a reference input, if this is needed). Applies a torch
#' function for forwarding an input through a linear function followed by an
#' activation function \eqn{\sigma} to the input data, i.e.
#' \deqn{y= \sigma(\text{nnf_dense(x,W,b)})}
#'
#' @param weight The weight matrix of dimensions \emph{(out_features,
#' in_features)}
#' @param bias The bias vector of dimension \emph{(out_features)}
#' @param activation_name The name of the activation function used by the layer
#' @param dim_in The input dimension of this layer. Use the default value
#' `NULL` to calculate the input dimension from the weight matrix.
#' @param dim_out The output dimension of this layer. Use the default value
#' `NULL` to calculate the output dimension from the weight matrix.
#' @param dtype The data type of all the parameters (Use `'float'` or
#' `'double'`)
#'
#' @section Attributes:
#' \describe{
#'   \item{`self$W`}{The weight matrix of this layer with shape
#'     \emph{(out_features, in_features)}}
#'   \item{`self$b`}{The bias vector of this layer with shape
#'     \emph{(out_features)}}
#'   \item{`self$...`}{Many attributes are inherited from the superclass
#'     [InterpretingLayer], e.g. `input`, `input_dim`, `preactivation`,
#'     `activation_name`, etc.}
#' }
#'
#' @noRd
#'
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

  #
  # x: [batch_size, in_features]
  #

  #' @section `self$forward()`:
  #' The forward function takes an input and forwards it through the layer
  #'
  #' ## Usage
  #' `self(x)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x`}{The input torch tensor of dimensions
  #'     \emph{(batch_size, in_features)}}
  #' }
  #'
  #' ## Return
  #' Returns the output of the layer with respect to the given inputs, with
  #' dimensions \emph{(batch_size, out_features)}
  #'
  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE) {
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

  #
  # x_ref: Tensor of size [1,in_features]
  #
  #' @section `self$update_ref()`:
  #' This function takes the reference input and runs it through
  #' the layer, updating the the values of `input_ref`, `preactivation_ref`
  #' and `output_ref`
  #'
  #' ## Usage
  #' `self$update_ref(x_ref)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x_ref`}{The new reference input, of dimensions
  #'     \emph{(1, in_features)}}
  #' }
  #'
  #' ## Return
  #' Returns the output of the reference input after
  #' passing through the layer, of dimension \emph{(1, out_features)}
  #'
  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE) {
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

  #
  # rel_output   [batch_size, dim_out, model_out]
  #
  #   output       [batch_size, dim_in, model_out]
  #
  #' @section `self$get_input_relevances()`:
  #' This method uses the output layer relevances and calculates the input
  #' layer relevances using the specified rule.
  #'
  #' ## Usage
  #' `self$get_input_relevances(`\cr
  #' `  rel_output,`\cr
  #' `  rule_name = 'simple',` \cr
  #' `  rule_param = NULL)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`rel_output`}{The output relevances, of dimensions
  #'     \emph{(batch_size, out_features, model_out)}}
  #'   \item{`rule_name`}{The name of the rule, with which the relevance
  #'     scores are calculated. Implemented are `"simple"`, `"epsilon"`,
  #'     `"alpha_beta"` (default: `"simple"`).}
  #'   \item{`rule_param`}{The parameter of the selected rule. Note: Only the
  #'   rules `"epsilon"` and `"alpha_beta"` take use of the parameter. Use
  #'   the default value `NULL` for the default parameters (`"epsilon"` :
  #'   \eqn{0.01}, `"alpha_beta"` : \eqn{0.5}).}
  #' }
  #'
  #' ## Return
  #' Returns the relevance score of the layer's input to the model output as a
  #' torch tensor of size \emph{(batch_size, in_features, model_out)}
  #'
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

    if (rule_name == "simple") {
      z <- z + (z == 0) * 1e-16

      rel_input <-
        self$get_gradient(rel_output / z, self$W) * input
    } else if (rule_name == "epsilon") {
      z <- z + rule_param * torch_sgn(z) + (z == 0) * 1e-16
      rel_input <-
        self$get_gradient(rel_output / z, self$W) * input
    } else if (rule_name == "alpha_beta") {
      out_part <- self$get_pos_and_neg_outputs(self$input)

      # Apply the simple rule for each part:
      # - positive part
      z <-
        rel_output / (out_part$pos + (out_part$pos == 0) * 1e-16)$unsqueeze(3)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_pos <- t1 * (input * (input > 0)) + t2 * (input * (input <= 0))

      # - negative part
      z <-
        rel_output / (out_part$neg + (out_part$neg == 0) * 1e-16)$unsqueeze(3)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- t1 * (input * (input <= 0)) + t2 * (input * (input > 0))

      # calculate over all relevance for the lower layer
      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
    }


    rel_input
  },

  #
  #   mult_output   [batch_size, dim_out, model_out]
  #
  #   output        [batch_size, dim_in, model_out]
  #
  #' @section `self$get_input_multiplier()`:
  #' This function is the local implementation of the DeepLift method for this
  #' layer and returns the multiplier from the input contribution to the
  #' output.
  #'
  #' ## Usage
  #' `self$get_input_multiplier(mult_output, rule_name = "rescale")`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`mult_output`}{The multiplier of the layer output contribution
  #'   to the model output. A torch tensor of shape
  #'   \emph{(batch_size, out_features, model_out)}}
  #'   \item{`rule_name`}{The name of the rule, with which the multiplier is
  #'        calculated. Implemented are `"rescale"` and `"reveal_cancel"`
  #'        (default: `"rescale"`).}
  #' }
  #'
  #' ## Return
  #' Returns the contribution multiplier of the layer's input to the model
  #' output as torch tensor of dimension \emph{(batch_size, in_features,
  #' model_out)}.
  #'
  get_input_multiplier = function(mult_output, rule_name = "rescale") {

    #
    # --------------------- Non-linear part---------------------------
    #
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      if (rule_name == "rescale") {
        delta_output <- (self$output - self$output_ref)$unsqueeze(3)
        delta_preact <-
          (self$preactivation - self$preactivation_ref)$unsqueeze(3)

        nonlin_mult <-
          delta_output / (delta_preact + 1e-16 * (delta_preact == 0))

        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult
      } else if (rule_name == "reveal_cancel") {
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

        mult_pos <-
          mult_output * (delta_output_pos / (delta_x$pos + 1e-16))$unsqueeze(3)
        mult_neg <-
          mult_output * (delta_output_neg / (delta_x$neg - 1e-16))$unsqueeze(3)
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
      self$get_gradient(mult_pos, self$W * (self$W <= 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, self$W * (self$W > 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, self$W * (self$W <= 0)) * (delta_input > 0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), self$W) *
        (delta_input == 0)

    mult_input
  },

  #
  #   grad_out   [batch_size, dim_out, model_out]
  #   weight     [dim_out, dim_in]
  #
  #   output  [batch_size, dim_in, model_out]
  #
  #' @section `self$get_gradient()`:
  #' This method uses \code{\link[torch]{torch_matmul}} to multiply the input
  #' with the gradient of a layer's output with respect to the layer's input.
  #' This results in the gradients of the model output with respect to
  #' layer's input.
  #'
  #' ## Usage
  #' `self$get_gradient(input, weight)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`grad_out`}{The gradients of the upper layer, a tensor of
  #'     dimension
  #'   \emph{(batch_size, out_features, model_out)}}
  #'   \item{`weight`}{A weight tensor of dimensions \emph{(out_features,
  #'     in_features)}}
  #' }
  #'
  #' ## Return
  #' Returns the gradient of the model's output with respect to the layer input
  #' as a torch tensor of dimension \emph{(batch_size, in_features,
  #' model_out)}.
  #'
  get_gradient = function(grad_out, weight) {
    grad_in <- torch_matmul(weight$t(), grad_out)

    grad_in
  },


  #' @section `self$get_pos_and_neg_outputs()`:
  #' This method separates the linear layer output (i.e. the preactivation)
  #' into the positive and negative parts.
  #'
  #' ## Usage
  #' `self$get_pos_and_neg_outputs(input, use_bias = FALSE)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`input`}{The input whose linear output we want to decompose into
  #'   the positive and negative parts}
  #'   \item{`use_bias`}{Boolean whether the bias vector should be considered
  #'   (default: FALSE)}
  #' }
  #'
  #' ## Return
  #' Returns a decomposition of the linear output of this layer with
  #' input `input` into the positive and negative parts. A list of two torch
  #' tensors with size \emph{(batch_size, out_features)} and keys `$pos`
  #' and `$neg`
  #'
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
  },

  #' @section `self$set_dtype()`:
  #' This function changes the data type of the weight and bias tensor to be
  #' either `"float"` or `"double"`.
  #'
  #' ## Usage
  #' `self$set_dtype(dtype)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`dtype`}{The data type of the layer's parameters. Use `"float"` or
  #'   `"double"`}
  #' }
  #'
  set_dtype = function(dtype) {
    if (dtype == "float") {
      self$W <- self$W$to(torch_float())
      self$b <- self$b$to(torch_float())
    } else if (dtype == "double") {
      self$W <- self$W$to(torch_double())
      self$b <- self$b$to(torch_double())
    } else {
      stop("Unknown argument for 'dtype' : '", dtype, "'. ",
           "Use 'float' or 'double' instead!")
    }
    self$dtype <- dtype
  }
)
