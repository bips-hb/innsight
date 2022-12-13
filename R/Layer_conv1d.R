
###############################################################################
#                           1D-convolutional Layer
###############################################################################

conv1d_layer <- nn_module(
  classname = "Conv1D_Layer",
  inherit = InterpretingLayer,

  # weight  : [out_channels, in_channels, kernel_length]
  # bias    : [out_channels]
  # dim_in  : [in_channels, in_length]
  # dim_out : [out_channels, out_length]
  # padding : [left, right] can be an integer or a two-dimensional tuple
  initialize = function(weight,
                        bias,
                        dim_in,
                        dim_out,
                        stride = 1,
                        padding = c(0, 0),
                        dilation = 1,
                        activation_name = "linear",
                        dtype = "float") {

    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$in_channels <- dim(weight)[2]
    self$out_channels <- dim(weight)[1]
    self$kernel_length <- dim(weight)[-c(1, 2)]
    self$stride <- stride
    self$padding <- padding
    self$dilation <- dilation

    self$get_activation(activation_name)

    if (!inherits(weight, "torch_tensor")) {
      self$W <- torch_tensor(weight)
    } else {
      self$W <- weight
    }
    if (!inherits(bias, "torch_tensor")) {
      self$b <- torch_tensor(bias)
    } else {
      self$b <- bias
    }
    self$set_dtype(dtype)
  },

  # x       : [batch_size, in_channels, in_length]
  #
  # output  : [batch_size, out_channels, out_length]
  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE, ...) {
    if (save_input) {
      self$input <- x
    }
    # Pad the input
    x <- nnf_pad(x, pad = self$padding)
    # Apply conv1d
    preactivation <- nnf_conv1d(x, self$W,
      bias = self$b,
      stride = self$stride,
      padding = 0,
      dilation = self$dilation
    )
    if (save_preactivation) {
      self$preactivation <- preactivation
    }

    output <- self$activation_f(preactivation)
    if (save_output) {
      self$output <- output
    }

    output
  },

  # x_ref   : [1, in_channels, in_length]
  #
  # output  : [1, out_channels, out_length]
  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE, ...) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    # Apply padding
    x_ref <- nnf_pad(x_ref, pad = self$padding)
    # Apply conv1d
    preactivation_ref <- nnf_conv1d(x_ref, self$W,
      bias = self$b,
      stride = self$stride,
      padding = 0,
      dilation = self$dilation
    )
    if (save_preactivation) {
      self$preactivation_ref <- preactivation_ref
    }

    output_ref <- self$activation_f(preactivation_ref)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  # mult_output   : [batch_size, out_channels, out_length, model_out]
  # rule_name     : "rescale" or "reveal_cancel"
  #
  # output        : [batch_size, in_channels, in_length, model_out]
  get_input_multiplier = function(mult_output, rule_name = "rescale") {

    # --------------------- Non-linear part---------------------------
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      # Get stabilizer
      eps <- self$get_stabilizer()

      if (rule_name == "rescale") {

        # output       [batch_size, out_channels, out_length]
        # delta_output [batch_size, out_channels, out_length, 1]
        delta_output <- (self$output - self$output_ref)$unsqueeze(4)
        delta_preact <-
          (self$preactivation - self$preactivation_ref)$unsqueeze(4)

        # Near zero needs special treatment
        mask <- (abs(delta_preact) < eps) * (1.0)
        x <-  mask * (self$preactivation + self$preactivation_ref)$unsqueeze(4) / 2
        x$requires_grad <- TRUE

        y <- sum(self$activation_f(x))
        grad <- autograd_grad(y, x)[[1]]

        nonlin_mult <- (1 - mask) * (delta_output / delta_preact) +
          mask * grad

        # mult_output   [batch_size, out_channels, out_length, model_out]
        # nonlin_mult   [batch_size, out_channels, out_length, 1]
        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult
      } else if (rule_name == "reveal_cancel") {
        pos_and_neg_output <-
          self$get_pos_and_neg_outputs(self$input - self$input_ref)

        delta_x_pos <- pos_and_neg_output$pos
        delta_x_neg <- pos_and_neg_output$neg

        act <- self$activation_f
        x <- self$preactivation
        x_ref <- self$preactivation_ref

        delta_output_pos <-
          ( 0.5 * (act(x_ref + delta_x_pos) - act(x_ref)) +
            0.5 * (act(x) - act(x_ref + delta_x_neg)))

        delta_output_neg <-
          ( 0.5 * (act(x_ref + delta_x_neg) - act(x_ref)) +
            0.5 * (act(x) - act(x_ref + delta_x_pos)))

        mult_pos <- mult_output *
          (delta_output_pos /
            (delta_x_pos + (delta_x_pos == 0) * eps))$unsqueeze(4)
        mult_neg <- mult_output *
          (delta_output_neg /
            (delta_x_neg - (delta_x_neg == 0) * eps))$unsqueeze(4)
      }
    }

    # -------------- Linear part -----------------------

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
      self$get_gradient(0.5 * (mult_pos + mult_neg), weight) *
        (delta_input == 0)

    mult_input
  },

  # input   : [batch_size, out_channels, out_length, model_out]
  # weight  : [out_channels, in_channels, kernel_length]
  #
  # output  : [batch_size, in_channels, in_length, model_out]
  get_gradient = function(input, weight) {

    # Since we have added the model_out dimension, strides and dilation need to
    # be extended by 1.

    stride <- c(self$stride, 1)

    # dilation is a number or a tuple of length 2
    dilation <- c(self$dilation, 1)

    out <- nnf_conv_transpose2d(input, weight$unsqueeze(4),
      bias = NULL,
      stride = stride,
      padding = 0,
      dilation = dilation
    )

    # If stride is > 1, it could happen that the reconstructed input after
    # padding (out) lost some dimensions, because multiple input shapes are
    # mapped to the same output shape. Therefore, we use padding with zeros to
    # fill in the missing irrelevant input values.
    lost_length <-
      self$input_dim[2] + self$padding[1] + self$padding[2] - dim(out)[3]

    out <- nnf_pad(out, pad = c(0, 0, 0, lost_length))
    # Now we have added the missing values such that
    # dim(out) = dim(padded_input)

    # Apply the inverse padding to obtain dim(out) = dim(input)
    out <- out[, , (self$padding[1] + 1):(dim(out)[3] - self$padding[2]), ]

    out
  },

  # output: list of two torch tensors with size
  #         [batch_size, out_channels, out_length] keys $pos and $neg
  get_pos_and_neg_outputs = function(input, use_bias = FALSE) {
    output <- NULL

    if (use_bias == TRUE) {
      b_pos <- self$b * (self$b > 0) * 0.5
      b_neg <- self$b * (self$b <= 0) * 0.5
    } else {
      b_pos <- NULL
      b_neg <- NULL
    }

    conv1d <- function(x, W, b) {
      x <- nnf_pad(x, pad = self$padding)
      out <- nnf_conv1d(x, W,
        bias = b,
        stride = self$stride,
        padding = 0,
        dilation = self$dilation
      )
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
  }
)
