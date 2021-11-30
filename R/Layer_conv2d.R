#' @include InterpretingLayer.R
#'
NULL

#' Two-dimensional Convolution Layer of a Neural Network
#'
#' Implementation of a two-dimensional Convolutional Neural Network layer as
#' a \code{\link[torch]{nn_conv2d}} module where input, preactivation and
#' output values of the last forward pass are stored (same for a reference
#' input, if this is needed). Applies the torch function
#' \code{\link[torch]{nnf_conv2d}} for forwarding an input through a 2d
#' convolution followed by an activation function \eqn{\sigma} to the input
#' data, i.e.
#' \deqn{y= \sigma(\code{nnf_conv2d(x,W,b)})}
#'
#' @param weight The weight matrix of dimension \emph{(out_channels,
#' in_channels, kernel_height, kernel_width)}
#' @param bias The bias vector of dimension \emph{(out_channels)}
#' @param dim_in The input dimension of the layer: \emph{(in_channels,
#' in_height, in_width)}
#' @param dim_out The output dimensions of the layer: \emph{(out_channels,
#' out_height, oout_width)}
#' @param stride The stride used in the convolution, by default `1`
#' @param padding The padding of the layer, by default `c(0,0,0,0)` (left,
#' right, top, bottom), can be an integer or a four-dimensional tuple
#' @param dilation The dilation of the layer, by default `1`
#' @param activation_name The name of the activation function used, by
#' default `"linear"`
#' @param dtype The data type of all the parameters (Use `'float'` or
#' `'double'`)
#'
#' @section Attributes:
#' \describe{
#'   \item{`self$W`}{The weight matrix of this layer with shape
#'     \emph{(out_channels, in_channels, kernel_height, kernel_width)}}
#'   \item{`self$b`}{The bias vector of this layer with shape
#'     \emph{(out_channels)}}
#'   \item{`self$...`}{Many attributes are inherited from the superclass
#'     [InterpretingLayer], e.g. `input`, `input_dim`, `preactivation`,
#'     `activation_name`, etc.}
#' }
#'
#' @noRd
#'
conv2d_layer <- nn_module(
  classname = "Conv2D_Layer",
  inherit = InterpretingLayer,

  #
  # weight: [out_channels, in_channels, kernel_height, kernel_width]
  # bias  : [out_channels]
  #
  initialize = function(weight,
                        bias,
                        dim_in,
                        dim_out,
                        stride = 1,
                        padding = c(0, 0, 0, 0),
                        dilation = 1,
                        activation_name = "linear",
                        dtype = "float") {

    # [in_channels, in_height, in_width]
    self$input_dim <- dim_in
    # [out_channels, out_height, out_width]
    self$output_dim <- dim_out
    self$in_channels <- dim(weight)[2]
    self$out_channels <- dim(weight)[1]
    self$kernel_size <- dim(weight)[-c(1, 2)]
    # int or tuple of length 2
    self$stride <- stride
    # tuple of length 4
    # padding goes from the last to the first dimension, i.e.
    # padding [left, right, top, bottom]
    self$padding <- padding
    # int or tuple of length 2
    self$dilation <- dilation

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
  # x : Tensor [minibatch, in_channels, in_height, in_width]
  #
  #' @section `self$forward()`:
  #' The forward function takes an input and forwards it through the layer,
  #' updating the the values of `input`, `preactivation` and `output`
  #'
  #' ## Usage
  #' `self(x)`
  #'
  #' ## Arguments
  #' \describe{
  #' \item{`x`}{The input torch tensor of dimensions \emph{(batch_size,
  #' in_channels, in_height, in_width)}}
  #' }
  #'
  #' ## Return
  #' Returns the output of the layer with respect to the given inputs, with
  #' dimensions \emph{(batch_size, out_channels, out_height, out_width)}
  #'
  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE) {
    if (save_input) {
      self$input <- x
    }

    # Apply padding
    x <- nnf_pad(x, pad = self$padding)
    # Apply convolution (2D)
    preactivation <- nnf_conv2d(x, self$W,
      bias = self$b,
      stride = self$stride,
      padding = 0,
      dilation = self$dilation
    )
    if (save_preactivation) {
      self$preactivation <- preactivation
    }
    # Apply non-linearity
    output <- self$activation_f(preactivation)
    if (save_output) {
      self$output <- output
    }

    output
  },

  #
  # x_ref: Tensor of size [in_channels, in_height, in_width]
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
  #'   \item{`x_ref`}{The new reference input, of dimensions \emph{(1,
  #'     in_channels, in_height, in_width)}}
  #' }
  #'
  #' ## Return
  #' Returns the output of the reference input after
  #' passing through the layer, of dimension \emph{(1, out_channels,
  #' out_height, out_width)}
  #'
  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    # Apply padding
    x_ref <- nnf_pad(x_ref, pad = self$padding)
    # Apply convolution (2D)
    preactivation_ref <- nnf_conv2d(x_ref, self$W,
      bias = self$b,
      stride = self$stride,
      padding = 0,
      dilation = self$dilation
    )
    if (save_preactivation) {
      self$preactivation_ref <- preactivation_ref
    }
    # Apply non-linearity
    output_ref <- self$activation_f(preactivation_ref)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  # Arguments:
  #   rule_name       : Name of the LRP-rule ("simple", "epsilon","alpha_beta")
  #   rule_param      : Parameter of the rule ("simple": no parameter,
  #                     "epsilon": epsilon value, set default to 0.001,
  #                     "alpha_beta": alpha value, set default to 0.5)
  #   rel_output      : relevance score from the upper layer to the output,
  #                     torch Tensor of size [batch_size, out_channels,
  #                     out_height, out_width, model_out]
  #
  #   output          : torch Tensor of size [batch_size, in_channels,
  #                     in_height, in_width, model_out]

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
  #'   \emph{(batch_size, out_channels, out_height, out_width, model_out)}}
  #'   \item{`rule_name`}{The name of the rule, with which the relevance scores
  #'     are calculated. Implemented are `"simple"`, `"epsilon"`,
  #'     `"alpha_beta"`, `"ww"` (default: `"simple"`).}
  #'   \item{`rule_param`}{The parameter of the selected rule. Note: Only
  #'     the rules `"epsilon"` and `"alpha_beta"` take use of the parameter.
  #'     Use the default value `NULL` for the default parameters
  #'     (`"epsilon"` : \eqn{0.01}, `"alpha_beta"` : \eqn{0.5}).}
  #' }
  #'
  #' ## Return
  #' Returns the relevance score of the layer's input to the model output as a
  #' torch tensor of size \emph{(batch_size, in_channels, in_height, in_width,
  #' model_out)}
  #'
  get_input_relevances = function(rel_output,
                                  rule_name = "simple",
                                  rule_param = NULL) {
    # set default parameter
    if (is.null(rule_param)) {
      if (rule_name == "epsilon") {
        rule_param <- 0.001
      } else if (rule_name == "alpha_beta") {
        rule_param <- 0.5
      }
    }

    if (rule_name == "simple") {
      z <- self$preactivation$unsqueeze(5)
      # add a small stabilizer
      z <- z + (z == 0) * 1e-16
      rel_input <-
        self$get_gradient(rel_output / z, self$W) * self$input$unsqueeze(5)
    } else if (rule_name == "epsilon") {
      z <- self$preactivation$unsqueeze(5)
      z <- z + rule_param * torch_sgn(z) + (z == 0) * 1e-16
      rel_input <-
        self$get_gradient(rel_output / z, self$W) * self$input$unsqueeze(5)
    } else if (rule_name == "alpha_beta") {
      # Get positive and negative decomposition of the linear output
      output <- self$get_pos_and_neg_outputs(self$input, use_bias = TRUE)

      # Apply the simple rule for each part:
      # - positive part
      z <- rel_output / (output$pos + (output$pos == 0) * 1e-16)$unsqueeze(5)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      input <- self$input$unsqueeze(5)
      rel_pos <- t1 * (input * (input > 0)) + t2 * (input * (input <= 0))

      # - negative part
      z <- rel_output / (output$neg + (output$neg == 0) * 1e-16)$unsqueeze(5)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- t1 * (input * (input <= 0)) + t2 * (input * (input > 0))

      # calculate over all relevance for the lower layer
      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
    } else {
      stop(sprintf("Unknown LRP-rule '%s'!", rule_name))
    }

    rel_input
  },

  #
  #   mult_output [batch_size, out_channels, out_height, out_width, model_out]
  #
  #   output [batch_size, in_channels, in_height, in_width, model_out]
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
  #'   \emph{(batch_size, out_channels, out_height, out_width, model_out)}}
  #'   \item{`rule_name`}{The name of the rule, with which the multiplier is
  #'        calculated. Implemented are `"rescale"` and `"reveal_cancel"`
  #'        (default: `"rescale"`).}
  #' }
  #'
  #' ## Return
  #' Returns the contribution multiplier of the layer's input to the model
  #' output as torch tensor of dimension \emph{(batch_size, in_channels,
  #' in_height, in_width, model_out)}.
  #'
  get_input_multiplier = function(mult_output, rule_name = "rescale") {

    #
    # --------------------- Non-linear part---------------------------
    #
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      if (rule_name == "rescale") {

        # output       [batch_size, out_channels, out_height, out_width]
        # delta_output [batch_size, out_channels, out_height, out_width, 1]
        delta_output <- (self$output - self$output_ref)$unsqueeze(5)
        delta_preact <-
          (self$preactivation - self$preactivation_ref)$unsqueeze(5)

        nonlin_mult <-
          delta_output / (delta_preact + 1e-16 * (delta_preact == 0))

        # mult_output
        #  [batch_size, out_channels, out_height, out_width, model_out]
        # nonlin_mult [batch_size, out_channels, out_height, out_width, 1]
        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult
      } else if (rule_name == "reveal_cancel") {
        output <- self$get_pos_and_neg_outputs(self$input - self$input_ref)

        delta_x_pos <- output$pos
        delta_x_neg <- output$neg

        act <- self$activation_f
        x <- self$preactivation
        x_ref <- self$preactivation_ref

        delta_output_pos <-
          ( 0.5 * (act(x_ref + delta_x_pos) - act(x_ref)) +
            0.5 * (act(x) - act(x_ref + delta_x_neg))) * (delta_x_pos != 0)

        delta_output_neg <-
          ( 0.5 * (act(x_ref + delta_x_neg) - act(x_ref)) +
            0.5 * (act(x) - act(x_ref + delta_x_pos))) * (delta_x_neg != 0)

        mult_pos <-
          mult_output * (delta_output_pos / (delta_x_pos + 1e-16))$unsqueeze(5)
        mult_neg <-
          mult_output * (delta_output_neg / (delta_x_neg - 1e-16))$unsqueeze(5)
      } else {
        stop(sprintf("Unknown DeepLift rule '%s'!", rule_name))
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

    # mult_input    [batch_size, in_channels, in_height, in_width, model_out]
    mult_input <-
      self$get_gradient(mult_pos, weight * (weight > 0)) * (delta_input > 0) +
      self$get_gradient(mult_pos, weight * (weight < 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, weight * (weight > 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, weight * (weight < 0)) * (delta_input > 0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), weight) *
        (delta_input == 0)

    mult_input
  },

  #
  #   input   [batch_size, out_channels, out_height, out_width, model_out]
  #   weight  [out_channels, in_channels, kernel_height, kernel_width]
  #
  #   output  [batch_size, in_channels, in_height, in_width, model_out]
  #
  #' @section `self$get_gradient()`:
  #' This method uses \code{\link[torch]{nnf_conv_transpose2d}} to multiply
  #' the input with the gradient of a layer's output with respect to the
  #' layer's input. This results in the gradients of the model output with
  #' respect to layer's input.
  #'
  #' ## Usage
  #' `self$get_gradient(input, weight)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`input`}{The gradients of the upper layer, a tensor of dimension
  #'   \emph{(batch_size, out_channels, out_height, out_width, model_out)}}
  #'   \item{`weight`}{A weight tensor of dimensions \emph{(out_channels,
  #'     in_channels, kernel_height, kernel_width)}}
  #' }
  #'
  #' ## Return
  #' Returns the gradient of the model's output with respect to the layer input
  #' as a torch tensor of dimension \emph{(batch_size, in_channels, in_height,
  #' in_width, model_out)}.
  #'
  get_gradient = function(input, weight) {
    # Since we have added the model_out dimension, strides and dilation need to
    # be extended by 1.

    # stride is a number or a tuple of length 2
    if (length(self$stride) == 1) {
      stride <- c(self$stride, self$stride, 1)
    } else if (length(self$stride) == 2) {
      stride <- c(self$stride, 1)
    }

    # dilation is a number or a tuple of length 2
    if (length(self$dilation) == 1) {
      dilation <- c(self$dilation, self$dilation, 1)
    } else if (length(self$dilation) == 2) {
      dilation <- c(self$dilation, 1)
    }

    out <- nnf_conv_transpose3d(input, weight$unsqueeze(5),
      bias = NULL,
      stride = stride,
      padding = 0,
      dilation = dilation
    )

    # If stride is > 1, it could happen that the reconstructed input after
    # padding (out) lost some dimensions, because multiple input shapes are
    # mapped to the same output shape. Therefore, we use padding with zeros to
    # fill in the missing irrelevant input values.
    lost_h <-
      self$input_dim[2] + self$padding[3] + self$padding[4] - dim(out)[3]
    lost_w <-
      self$input_dim[3] + self$padding[1] + self$padding[2] - dim(out)[4]

    # (begin last axis, end last axis, begin 2nd to last axis, end 2nd to last
    # axis, begin 3rd to last axis, etc.)
    out <- nnf_pad(out, pad = c(0, 0, 0, lost_w, 0, lost_h))
    # Now we have added the missing values such that
    # dim(out) = dim(padded_input)

    # Apply the inverse padding to obtain dim(out) = dim(input)
    dim_out <- dim(out)
    out <- out[
      , , (self$padding[3] + 1):(dim_out[3] - self$padding[4]),
      (self$padding[1] + 1):(dim_out[4] - self$padding[2]),
    ]


    out
  },

  #
  # input   [batch_size, in_channels, in_height, in_width]
  #
  # output$pos [batch_size, out_channels, out_height, out_width]
  # output$neg [batch_size, out_channels, out_height, out_width]
  #
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
  #' tensors with size \emph{(batch_size, out_channels, out_height, out_width)}
  #' and keys `$pos` and `$neg`
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

    conv2d <- function(x, W, b) {
      x <- nnf_pad(x, pad = self$padding)
      out <- nnf_conv2d(x, W,
        bias = b,
        stride = self$stride,
        padding = 0,
        dilation = self$dilation
      )
      out
    }

    # input (+) x weight (+) and input (-) x weight (-)
    output$pos <- conv2d(input * (input > 0), self$W * (self$W > 0), b_pos) +
      conv2d(input * (input <= 0), self$W * (self$W <= 0), b_pos)

    # input (+) x weight (-) and input (-) x weight (+)
    output$neg <- conv2d(input * (input > 0), self$W * (self$W <= 0), b_neg) +
      conv2d(input * (input <= 0), self$W * (self$W > 0), b_neg)

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
