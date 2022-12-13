###############################################################################
#                           2D-convolutional Layer
##############################################################################

conv2d_layer <- nn_module(
  classname = "Conv2D_Layer",
  inherit = InterpretingLayer,

  # weight  : [out_channels, in_channels, kernel_height, kernel_width]
  # bias    : [out_channels]
  # dim_in  : [in_channels, in_height, in_width]
  # dim_out : [out_channels, out_height, out_width]
  # stride  : [along width, along height] single integer or tuple
  # padding : [left, right, top, bottom], can be an integer or a
  #           four-dimensional tuple
  # dilation: [along width, along height] single integer or tuple
  initialize = function(weight,
                        bias,
                        dim_in,
                        dim_out,
                        stride = 1,
                        padding = c(0, 0, 0, 0),
                        dilation = 1,
                        activation_name = "linear",
                        dtype = "float") {

    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$in_channels <- dim(weight)[2]
    self$out_channels <- dim(weight)[1]
    self$kernel_size <- dim(weight)[-c(1, 2)]
    self$stride <- stride
    self$padding <- padding
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

  # x       : [batch_size, in_channels, in_height, in_width]
  #
  # output  : [batch_size, out_channels, out_height, out_width]
  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE, ...) {
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

  # x_ref   : Tensor of size [1, in_channels, in_height, in_width]
  #
  # output  : [1, out_channels, out_height, out_width]
  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE, ...) {
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

  #
  #   input   [batch_size, out_channels, out_height, out_width, model_out]
  #   weight  [out_channels, in_channels, kernel_height, kernel_width]
  #
  #   output  [batch_size, in_channels, in_height, in_width, model_out]
  #
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

  # input   [batch_size, in_channels, in_height, in_width]
  #
  # output$pos [batch_size, out_channels, out_height, out_width]
  # output$neg [batch_size, out_channels, out_height, out_width]
  get_pos_and_neg_outputs = function(input, use_bias = FALSE) {
    output <- NULL

    if (use_bias) {
      b_pos <- torch_clamp(self$b, min = 0)
      b_neg <- torch_clamp(self$b, max = 0)
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

    input_pos <- torch_clamp(input, min = 0)
    input_neg <- torch_clamp(input, max = 0)
    W_pos <- torch_clamp(self$W, min = 0)
    W_neg <- torch_clamp(self$W, max = 0)

    # input (+) x weight (+) and input (-) x weight (-)
    output$pos <-
      conv2d(input_pos, W_pos, b_pos) +
      conv2d(input_neg, W_neg, NULL)

    # input (+) x weight (-) and input (-) x weight (+)
    output$neg <-
      conv2d(input_pos, W_neg, b_neg) +
      conv2d(input_neg, W_pos, NULL)

    output
  }
)
