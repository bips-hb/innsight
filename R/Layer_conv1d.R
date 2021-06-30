#' @include Layer.R
NULL

#' One-dimensional convolution layer of a Neural Network
#'
#' Implementation of a one-dimensional Convolutional Neural Network layer as a \code{\link[torch]{nn_conv1d}} module
#' where input, preactivation and output values of the last forward pass are stored
#' (same for a reference input, if this is needed). Applies the torch function \code{\link[torch]{nnf_conv1d}} for
#' forwarding an input through a 1d convolution followed by an activation function
#' \eqn{\sigma} to the input data, i.e.
#' \deqn{y= \sigma(\code{nnf_conv1d(x,W,b)})}
#'
#' @param weight The weight matrix of dimension \emph{(out_channels, in_channels, kernel_size)}
#' @param bias The bias vector of dimension \emph{(out_channels)}
#' @param dim_in The input dimension of the layer: \emph{(in_channels, in_length)}
#' @param dim_out The output dimensions of the layer: \emph{(out_channels, out_length)}
#' @param stride The stride used in the convolution, by default `1`
#' @param padding The padding of the layer, by default `c(0,0)` (left, right), can be an integer or a two-dimensional tuple
#' @param dilation The dilation of the layer, by default `1`
#' @param activation_name The name of the activation function used, by default `"linear"`
#' @param dtype The data type of all the parameters (Use `'float'` or `'double'`)
#'
#' @section Attributes:
#' \describe{
#'   \item{`self$W`}{The weight matrix of this layer with shape \emph{(out_channels, in_channels, kernel_size)}}
#'   \item{`self$b`}{The bias vector of this layer with shape \emph{(out_channels)}}
#'   \item{`self$...`}{Many attributes are inherited from the superclass [Layer], e.g.
#'   `input`, `input_dim`, `preactivation`, `activation_name`, etc.}
#' }
#'
#'@export
#'
#' @section Methods:
#'
conv1d_layer <- torch::nn_module(
  classname = "Conv1D_Layer",
  inherit = Layer,

  #
  # weight: [out_channels, in_channels, kernel_size]
  # bias  : [out_channels]
  initialize = function(weight,
                        bias,
                        dim_in,
                        dim_out,
                        stride = 1,
                        padding = c(0,0),
                        dilation = 1,
                        activation_name = 'linear',
                        dtype = "float") {

    # [in_channels, in_length]
    self$input_dim <- dim_in
    # [out_channels, out_length]
    self$output_dim <- dim_out
    self$in_channels <- dim(weight)[2]
    self$out_channels <- dim(weight)[1]
    self$kernel_size <- dim(weight)[-c(1,2)]
    self$stride <- stride
    # padding     [left, right]
    self$padding <- padding
    self$dilation <- dilation

    self$get_activation(activation_name)

    if (!inherits(weight, "torch_tensor")) {
      self$W <- torch_tensor(weight)
    }
    else {
      self$W <- weight
    }
    if (!inherits(bias, "torch_tensor")) {
      self$b <- torch_tensor(bias)
    }
    else {
      self$b <- bias
    }
    self$set_dtype(dtype)
  },


  #' @section `self$forward()`:
  #' The forward function takes an input and forwards it through the layer.
  #'
  #' ## Usage
  #' `self(x)`
  #'
  #' ## Arguments
  #' \describe{
  #' \item{`x`}{The input torch tensor of dimensions \emph{(batch_size, in_channels, in_length)}}
  #' }
  #'
  #' ## Return
  #' Returns the output of the layer with respect to the given inputs, with dimensions
  #' \emph{(batch_size, out_channels, out_length)}
  #'
  forward = function(x) {
    self$input <- x
    # Pad the input
    x <- torch::nnf_pad(x, pad = self$padding)
    # Apply conv1d
    self$preactivation <- torch::nnf_conv1d(x, self$W,
                                            bias = self$b,
                                            stride = self$stride,
                                            padding = 0,
                                            dilation = self$dilation)
    self$output <- self$activation_f(self$preactivation)

    self$output
  },

  #' @section `self$update_ref()`:
  #' This function takes the reference input and runs it through
  #' the layer, updating the the values of `input_ref`, `preactivation_ref` and `output_ref`
  #'
  #' ## Usage
  #' `self$update_ref(x_ref)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x_ref`}{The new reference input, of dimensions \emph{(1, in_channels, in_length)}}
  #' }
  #'
  #' ## Return
  #' Returns the output of the reference input after
  #' passing through the layer, of dimension \emph{(1, out_channels, out_length)}
  #'
  update_ref = function(x_ref) {
    self$input_ref <- x_ref
    # Apply padding
    x_ref <- torch::nnf_pad(x_ref, pad = self$padding)
    # Apply conv1d
    self$preactivation_ref <- torch::nnf_conv1d(x_ref, self$W,
                                                bias = self$b,
                                                stride = self$stride,
                                                padding = 0,
                                                dilation = self$dilation)

    self$output_ref <- self$activation_f(self$preactivation_ref)

    self$output_ref
  },



  #' @section `self$get_input_relevances()`:
  #' This method uses the output layer relevances and calculates the input layer
  #' relevances using the specified rule.
  #'
  #' ## Usage
  #' `self$get_input_relevances(rel_output,`
  #' `  rule_name = 'simple',`
  #' `  rule_param = NULL)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`rel_output`}{The output relevances, of dimensions \emph{(batch_size, out_channels, out_length, model_out)}}
  #'   \item{`rule_name`}{The name of the rule, with which the relevance scores are
  #'        calculated. Implemented are `"simple"`, `"epsilon"`, `"alpha_beta"`,
  #'        `"ww"` (default: `"simple"`).}
  #'   \item{`rule_param`}{The parameter of the selected rule. Note: Only the rules
  #'        `"epsilon"` and `"alpha_beta"` take use of the parameter. Use the default
  #'        value `NULL` for the default parameters (`"epsilon"` : \eqn{0.01}, `"alpha_beta"` : \eqn{0.5}).}
  #' }
  #'
  get_input_relevances = function(rel_output, rule_name = 'simple', rule_param = NULL) {

    if (rule_name == 'simple') {
      z <-  self$preactivation$unsqueeze(4)
      # add a small stabilizer
      z <- z + (z==0) * 1e-12

      rel_input <-
        self$get_gradient(rel_output / z, self$W) * self$input$unsqueeze(4)
    }
    else if (rule_name == 'epsilon') {
      # set default parameter
      if (is.null(rule_param)) {
        epsilon <- 0.001
      }
      else {
        epsilon <- rule_param
      }

      z <-  self$preactivation$unsqueeze(4)
      z <- z + epsilon * torch::torch_sgn(z) + (z==0) * 1e-12

      rel_input <-
        self$get_gradient(rel_output / z, self$W) * self$input$unsqueeze(4)
    }
    else if (rule_name == 'alpha_beta') {
      # set default parameter
      if (is.null(rule_param)) {
        alpha <- 0.5
      }
      else {
        alpha <- rule_param
      }

      # Get positive and negative part of the output
      out_part <- self$get_pos_and_neg_outputs(self$input, use_bias = TRUE)
      input <- self$input$unsqueeze(4)

      # Apply simple rule on the positive part
      z <- rel_output / ( out_part$pos + (out_part$pos == 0) * 1e-16 )$unsqueeze(4)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_pos <- t1 * (input * (input > 0)) + t2 * (input * (input <= 0))

      # Apply simple rule on the negative part
      z <- rel_output / ( out_part$neg + (out_part$neg == 0) * 1e-16 )$unsqueeze(4)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- t1 * (input * (input <= 0)) + t2 * (input * (input > 0))

      # Calculate over all relevance for the lower layer
      rel_input <- rel_pos * alpha + rel_neg * (1 - alpha)
    }

    rel_input
  },

  #' @section `self$get_input_multiplier()`:
  #'
  get_input_multiplier = function(mult_output, rule_name = "rescale") {

    #
    # --------------------- Non-linear part---------------------------
    #
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      if (rule_name == "rescale") {

        # output       [batch_size, out_channels, out_length]
        # delta_output [batch_size, out_channels, out_length, 1]
        delta_output <- (self$output - self$output_ref)$unsqueeze(4)
        delta_preact <- (self$preactivation - self$preactivation_ref)$unsqueeze(4)

        nonlin_mult <- delta_output / (delta_preact + 1e-16 * (delta_preact == 0))

        # mult_output   [batch_size, out_channels, out_length, model_out]
        # nonlin_mult   [batch_size, out_channels, out_length, 1]
        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult

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

        mult_pos <- mult_output * (delta_output_pos / (delta_x_pos + (delta_x_pos == 0) * 1e-16))$unsqueeze(4)
        mult_neg <- mult_output * (delta_output_neg / (delta_x_neg - (delta_x_neg == 0) * 1e-16))$unsqueeze(4)
      }
    }

    #
    # -------------- Linear part -----------------------
    #

    # input        [batch_size, in_channels, in_length]
    # delta_input  [batch_size, in_channels, in_length, 1]
    delta_input <- (self$input - self$input_ref)$unsqueeze(4)

    # weight      [out_channels, in_channels, kernel_length]
    weight <- self$W

    # mult_input    [batch_size, in_channels, in_length, model_out]
    mult_input <-
      self$get_gradient(mult_pos, weight * (weight > 0)) * (delta_input > 0) +
      self$get_gradient(mult_pos, weight * (weight < 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, weight * (weight > 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, weight * (weight < 0)) * (delta_input > 0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), weight) * (delta_input == 0)

    mult_input
  },

  #' @section `self$get_gradient()`:
  #' This method uses \code{\link[torch]{nnf_conv_transpose1d}} to multiply the input with the
  #' gradient of a layer's output with respect to the layer's input. This results in the
  #' gradients of the model output with respect to layer's input.
  #'
  #' ## Usage
  #' `self$get_gradient(input, weight)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`input`}{The gradients of the upper layer, a tensor of dimension \emph{(batch_size, out_channels, out_length, model_out)}}
  #'   \item{`weight`}{A weight tensor of dimensions \emph{(out_channels, in_channels, kernel_size)}}
  #' }
  #'
  #' ## Return
  #' This returns the gradient of the model's output with respect to the layer input.
  #'
  get_gradient = function(input, weight) {

    # Since we have added the model_out dimension, strides and dilation need to
    # be extended by 1.

    stride <- c(self$stride,1)

    # dilation is a number or a tuple of length 2
    dilation <- c(self$dilation,1)

    out <- torch::nnf_conv_transpose2d(input, weight$unsqueeze(4),
                                       bias = NULL,
                                       stride = stride,
                                       padding = 0,
                                       dilation = dilation)

    # If stride is > 1, it could happen that the reconstructed input after
    # padding (out) lost some dimensions, because multiple input shapes are
    # mapped to the same output shape. Therefore, we use padding with zeros to
    # fill in the missing irrelevant input values.
    lost_length <- self$input_dim[2] + self$padding[1] + self$padding[2] - dim(out)[3]

    out <- torch::nnf_pad(out, pad = c(0,0,0, lost_length))
    # Now we have added the missing values such that dim(out) = dim(padded_input)

    # Apply the inverse padding to obtain dim(out) = dim(input)
    out <- out[,,(self$padding[1]+1):(dim(out)[3]-self$padding[2]),]

    out
  },

  #' @section `self$get_pos_and_neg_outputs()`:
  #'
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

    conv1d <- function(x, W, b) {
      x <- torch::nnf_pad(x, pad = self$padding)
      out <- torch::nnf_conv1d(x, W,
                               bias = b,
                               stride = self$stride,
                               padding = 0,
                               dilation = self$dilation)
      out
    }

    # input (+) x weight (+) and input (-) x weight (-)
    output$pos <-
      conv1d(input * (input > 0), self$W * (self$W > 0), b_pos) +
      conv1d(input * (input < 0), self$W * (self$W < 0), b_pos)

    # input (+) x weight (-) and input (-) x weight (+)
    output$neg <-
      conv1d(input * (input > 0), self$W * (self$W < 0), b_neg) +
      conv1d(input * (input < 0), self$W * (self$W > 0), b_neg)

    output
  },


  #' @section `self$set_dtype()`:
  #'
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
  }
)

