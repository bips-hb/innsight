
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

  # input   : [batch_size, out_channels, out_length, model_out]
  # weight  : [out_channels, in_channels, kernel_length]
  #
  # output  : [batch_size, in_channels, in_length, model_out]
  get_gradient = function(input, weight, ...) {

    # Since we have added the model_out dimension, strides and dilation need to
    # be extended by 1.
    out <- nnf_conv_transpose2d(input, weight$unsqueeze(4),
      bias = NULL,
      stride = c(self$stride, 1),
      padding = 0,
      dilation = c(self$dilation, 1)
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

    if (use_bias) {
      b_pos <- torch_clamp(self$b, min = 0)
      b_neg <- torch_clamp(self$b, max = 0)
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

    input_pos <- torch_clamp(input, min = 0)
    input_neg <- torch_clamp(input, max = 0)
    W_pos <- torch_clamp(self$W, min = 0)
    W_neg <- torch_clamp(self$W, max = 0)

    # input (+) x weight (+) and input (-) x weight (-)
    output$pos <-
      conv1d(input_pos, W_pos, b_pos) +
      conv1d(input_neg, W_neg, NULL)

    # input (+) x weight (-) and input (-) x weight (+)
    output$neg <-
      conv1d(input_pos, W_neg, b_neg) +
      conv1d(input_neg, W_pos, NULL)

    output
  }
)
