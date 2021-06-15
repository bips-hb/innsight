
#' @include Layer.R
#' @export
conv2d_layer <- torch::nn_module(
  classname = "Conv2D_Layer",
  inherit = Layer,

  #
  # weight: [out_channels, in_channels, kernel_height, kernel_width]
  # bias  : [out_channels]
  #
  initialize = function(weight,
                        bias,
                        dim_in,
                        dim_out,
                        stride = 1,
                        padding = 0,
                        dilation = 1,
                        activation_name = 'linear', ...) {

    # [in_channels, in_height, in_width]
    self$input_dim <- dim_in
    # [out_channels, out_height, out_width]
    self$output_dim <- dim_out
    self$in_channels <- dim(weight)[2]
    self$out_channels <- dim(weight)[1]
    self$kernel_size <- dim(weight)[-c(1,2)]

    self$W <- torch_tensor(weight, dtype = torch_float())
    self$b <- torch_tensor(bias, dtype = torch_float())

    self$stride <- stride
    self$padding <- padding
    self$dilation <- dilation

    self$get_activation(activation_name, ...)
  },

  #
  # x : Tensor [minibatch, in_channels, in_height, in_width]
  #
  forward = function(x) {
    self$input <- x
    self$preactivation <- torch::nnf_conv2d(x, self$W, self$b, self$stride, self$padding, self$dilation)
    if (self$activation_name == 'linear') {
      self$output <- self$preactivation
    }
    else {
      self$output <- self$activation_f(self$preactivation)
    }
    self$output
  },

  #
  # x_ref: Tensor of size [in_channels, in_height, in_width]
  #
  update_ref = function(x_ref) {

    self$input_ref <- x_ref
    self$preactivation_ref <- torch::nnf_conv2d(x_ref, self$W, self$b, self$stride, self$padding, self$dilation)
    if (self$activation_name == 'linear') {
      self$output_ref <- self$preactivation_ref
    }
    else {
      self$output_ref <- self$activation_f(self$preactivation_ref)
    }

    self$output_ref
  }
)
