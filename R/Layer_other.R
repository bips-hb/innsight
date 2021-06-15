
#' @include Layer.R
#' @export
flatten_layer <- torch::nn_module(
  classname = "Flatten_Layer",
  inherit = Layer,

  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
  },

  #
  # x: [num_batches, features_1, ..., features_n]
  # out: [num_batches, features_1 * ... * features_n]
  #
  forward = function(x, channels_first = TRUE) {
      self$input <- x
      if (channels_first == FALSE) {
          x <- torch::torch_transpose(x, 2, -1)
      }
      self$preactivation <- torch::torch_flatten(x, start_dim = 2)
      self$output <- self$preactivation

      self$output
  },

  #
  # x_ref: Tensor of size [features_1, ...,  features_n]
  # out: Tensor of size [features_1 * ... * features_n]
  #
  update_ref = function(x_ref, channels_first = TRUE) {
      self$input_ref <- x_ref
      if (channels_first == FALSE) {
        x_ref <- torch::torch_transpose(x_ref, 2, -1)
      }
      self$preactivation_ref <- torch::torch_flatten(x_ref, start_dim = 1)
      self$output_ref <- self$preactivation_ref

      self$output_ref
  }
)
