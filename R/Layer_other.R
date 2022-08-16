
###############################################################################
#      Super class for other layer types which are not based on a dense
#      or convolutional layers
###############################################################################

OtherLayer <- nn_module(
  classname = "OtherLayer",
  input_dim = NULL,
  output_dim = NULL,

  initialize = function() {
  },

  forward = function() {
  },

  reset = function() {
  },

  get_input_relevances = function(...) {
    self$reshape_to_input(...)
  },

  get_input_multiplier = function(...) {
    self$reshape_to_input(...)
  },

  get_gradient = function(...) {
    self$reshape_to_input(...)
  },

  reshape_to_input = function(rel_output, ...) {
    rel_output
  },

  set_dtype = function(dtype) {
  }
)


###############################################################################
#                               Flatten Layer
###############################################################################
flatten_layer <- nn_module(
  classname = "Flatten_Layer",
  inherit = OtherLayer,

  start_dim = NULL,
  end_dim = NULL,
  channels_first = NULL,

  initialize = function(dim_in, dim_out, start_dim = 2, end_dim = -1) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$start_dim <- start_dim
    self$end_dim <- end_dim
    self$channels_first <- TRUE
  },

  forward = function(x, channels_first = TRUE, ...) {
    if (channels_first == FALSE) {
      x <- torch_movedim(x, 2, -1)
    }
    self$channels_first <- channels_first

    output <- torch_flatten(x, start_dim = self$start_dim,
                            end_dim = self$end_dim)

    output
  },
  update_ref = function(x_ref, channels_first = TRUE, ...) {
    if (channels_first == FALSE) {
      x_ref <- torch_movedim(x_ref, 2, -1)
    }
    output_ref <- torch_flatten(x_ref, start_dim = self$start_dim,
                                end_dim = self$end_dim)

    output_ref
  },

  # Arguments:
  #   output  : relevance score from the upper layer to the output, torchTensor
  #           : of size [batch_size, dim_out,  model_out]
  #
  #   input   : torch Tensor of size [batch_size, in_channels, * , model_out]
  #
  reshape_to_input = function(rel_output, ...) {
    batch_size <- dim(rel_output)[1]
    model_out <- rev(dim(rel_output))[1]

    if (self$channels_first == FALSE) {
      in_channels <- self$input_dim[1]
      in_dim <- c(self$input_dim[-1], in_channels)
      rel_input <- rel_output$reshape(c(batch_size, in_dim, model_out))
      rel_input <- torch_movedim(rel_input, length(self$input_dim) + 1, 2)
    } else {
      rel_input <- rel_output$reshape(c(batch_size, self$input_dim, model_out))
    }

    rel_input
  }
)

###############################################################################
#                           Concatenate Layer
###############################################################################

concatenate_layer <- nn_module(
  classname = "Concatenate_Layer",
  inherit = OtherLayer,

  dim = NULL,

  initialize = function(dim, dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    if (dim == -1) {
      self$dim <- length(dim_in[[1]]) + 1 # add batch axis
    } else {
      self$dim <- dim
    }
  },

  forward = function(x, channels_first = FALSE, ...) {
    output <- torch_cat(x, dim = self$dim)

    output
  },

  update_ref = function(x_ref, channels_first = FALSE, ...) {
    output_ref <- torch_cat(x_ref, dim = self$dim)

    output_ref
  },


  reshape_to_input = function(rel_output, ...) {
    split_size <- lapply(self$input_dim, function(x) x[self$dim - 1])
    rel_input <- torch_split(rel_output, split_size, self$dim)

    rel_input
  }
)

###############################################################################
#                           Adding Layer
###############################################################################
add_layer <- nn_module(
  classname = "Adding_Layer",
  inherit = OtherLayer,

  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
  },

  forward = function(x, ...) {
    torch_stack(x, dim = length(self$input_dim))$sum(length(self$input_dim))
  },

  update_ref = function(x_ref, ...) {
    torch_stack(x_ref, dim = length(self$input_dim))$sum(length(self$input_dim))
  },

  reshape_to_input = function(rel_output, ...) {
    rep(list(rel_output / length(self$input_dim)), length(self$input_dim))
  }
)

###############################################################################
#                           Padding Layer
###############################################################################
padding_layer <- nn_module(
  classname = "Padding_Layer",
  inherit = OtherLayer,

  initialize = function(pad, dim_in, dim_out, mode = "constant", value = 0) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$pad <- pad
    self$mode <- mode
    self$value <- value

    num_dims <- length(dim_in)
    rev_dim_in <- rev(dim_in)
    if (num_dims == 1) {
      indices <- list(seq.int(from = pad[1] + 1, to = pad[1] + rev_dim_in[1]))
    } else if (num_dims == 2) {
      indices <- list(
        seq_len(rev_dim_in[2]),
        seq.int(from = pad[1] + 1, to = pad[1] + rev_dim_in[1])
      )
    } else {
      indices <- list(
        seq_len(rev_dim_in[3]),
        seq.int(from = pad[3] + 1, to = pad[3] + rev_dim_in[2]),
        seq.int(from = pad[1] + 1, to = pad[1] + rev_dim_in[1])
      )
    }
    self$rev_pad_idx <- indices
  },

  forward = function(x, ...) {
    nnf_pad(x, pad = self$pad, mode = self$mode, value = self$value)
  },

  update_ref = function(x_ref, ...) {
    nnf_pad(x_ref, pad = self$pad, mode = self$mode, value = self$value)
  },

  reshape_to_input = function(rel_output, ...) {
    dim <- rel_output$shape
    indices <- append(self$rev_pad_idx, list(seq_len(dim[1])), 0)
    indices <- append(indices, list(seq_len(rev(dim)[1])))
    args <- append(indices, list(x = rel_output), 0)
    args$drop <- FALSE

    do.call('[', args)
  }
)
###############################################################################
#                           Skipping Layer
###############################################################################
skipping_layer <- nn_module(
  classname = "Skipping_Layer",
  inherit = OtherLayer,

  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
  },

  forward = function(x, ...) {
    x
  },

  update_ref = function(x_ref, ...) {
    x_ref
  }
)
