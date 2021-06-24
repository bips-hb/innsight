
#' @include Layer.R
#' @export
flatten_layer <- torch::nn_module(
  classname = "Flatten_Layer",
  inherit = Layer,

  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$channels_first <- TRUE
  },

  #
  # x: [num_batches, features_1, ..., features_n]
  # out: [num_batches, features_1 * ... * features_n]
  #
  forward = function(x, channels_first = TRUE) {
      self$input <- x
      if (channels_first == FALSE) {
          x <- torch::torch_movedim(x, 2, -1)
          self$channels_first <- FALSE
      }
      else {
          self$channels_first <- TRUE
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
        x_ref <- torch::torch_movedim(x_ref, 2, -1)
      }
      self$preactivation_ref <- torch::torch_flatten(x_ref, start_dim = 2)
      self$output_ref <- self$preactivation_ref

      self$output_ref
  },

  # Arguments:
  #   output       : relevance score from the upper layer to the output, torch Tensor
  #                   : of size [batch_size, dim_out,  model_out]
  #
  #   input          : torch Tensor of size [batch_size, in_channels, * , model_out]
  #
  reshape_to_input = function(output) {
    batch_size <- dim(output)[1]
    model_out <- rev(dim(output))[1]

    if (self$channels_first == FALSE) {
      in_channels <- self$input_dim[1]
      in_dim <- c(self$input_dim[-1], in_channels)
      input <- output$reshape(c(batch_size, in_dim, model_out))
      input <- torch_movedim(input, length(self$input_dim) + 1, 2)
    }
    else {
      input <- output$reshape(c(batch_size, self$input_dim, model_out))
    }
    input
  }
)
