
#' @include Layer.R
#' @export
dense_layer <- torch::nn_module(
  classname = "Dense_Layer",
  inherit = Layer,

  #
  # weight: [out_features, in_features]
  # bias  : [out_features]
  #
  initialize = function(weight, bias, activation_name, ...) {
    self$input_dim <- dim(weight)[2]
    self$output_dim <- dim(weight)[1]
    self$W <- torch::torch_tensor(weight, dtype = torch::torch_float())
    self$b <- torch::torch_tensor(bias, dtype = torch::torch_float())
    self$get_activation(activation_name, ...)
  },

  #
  # x: [num_batches, in_features]
  #
  forward = function(x) {

    self$input <- x
    self$preactivation <- torch::nnf_linear(x, self$W, self$b)

    if (self$activation_name == 'linear') {
      self$output <- self$preactivation
    }
    else {
      self$output <- self$activation_f(self$preactivation)
    }

    self$output
  },

  #
  # x_ref: Tensor of size [in_features]
  #
  update_ref = function(x_ref) {

    self$input_ref <- x_ref
    self$preactivation_ref <- torch::nnf_linear(x_ref, self$W, self$b)

    if (self$activation_name == 'linear') {
      self$output_ref <- self$preactivation_ref
    }
    else {
      self$output_ref <- self$activation_f(self$preactivation_ref)
    }

    self$output_ref
  }
)
