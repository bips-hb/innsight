
flatten_layer <- nn_module(
  classname = "Flatten_Layer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
  channels_first = NULL,
  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$channels_first <- TRUE
  },
  forward = function(x, channels_first = TRUE, save_input = TRUE,
                     save_output = TRUE) {
    if (save_input) {
      self$input <- x
    }
    if (channels_first == FALSE) {
      x <- torch_movedim(x, 2, -1)
      self$channels_first <- FALSE
    } else {
      self$channels_first <- TRUE
    }
    output <- torch_flatten(x, start_dim = 2)
    if (save_output) {
      self$output <- output
    }

    output
  },
  update_ref = function(x_ref, channels_first = TRUE, save_input = TRUE,
                        save_output = TRUE) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    if (channels_first == FALSE) {
      x_ref <- torch_movedim(x_ref, 2, -1)
    }
    output_ref <- torch_flatten(x_ref, start_dim = 2)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  # Arguments:
  #   output  : relevance score from the upper layer to the output, torchTensor
  #           : of size [batch_size, dim_out,  model_out]
  #
  #   input   : torch Tensor of size [batch_size, in_channels, * , model_out]
  #
  reshape_to_input = function(output) {
    batch_size <- dim(output)[1]
    model_out <- rev(dim(output))[1]

    if (self$channels_first == FALSE) {
      in_channels <- self$input_dim[1]
      in_dim <- c(self$input_dim[-1], in_channels)
      input <- output$reshape(c(batch_size, in_dim, model_out))
      input <- torch_movedim(input, length(self$input_dim) + 1, 2)
    } else {
      input <- output$reshape(c(batch_size, self$input_dim, model_out))
    }
    input
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
  }
)
