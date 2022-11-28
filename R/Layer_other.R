
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
#                           BatchNorm Layer
###############################################################################
batchnorm_layer <- nn_module(
  classname = "BatchNorm_Layer",
  inherit = OtherLayer,

  initialize = function(num_features, eps, gamma, beta, run_mean, run_var,
                        dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$num_features <- num_features
    self$eps <- eps
    self$beta <- beta
    self$gamma <- gamma
    self$run_mean <- run_mean
    self$run_var <- run_var
  },

  forward = function(x, save_input = TRUE, save_output = TRUE, ...) {
    if (save_input) {
      self$input <- x
    }

    out <- nnf_batch_norm(x, self$run_mean, self$run_var, self$gamma, self$beta,
                          eps = self$eps)

    if (save_output) {
      self$output <- out
    }

    out
  },

  update_ref = function(x_ref, save_input = TRUE, save_output = TRUE, ...) {
    if (save_input) {
      self$input_ref <- x_ref
    }

    out <- nnf_batch_norm(x_ref, self$run_mean, self$run_var, self$gamma,
                          self$beta, eps = self$eps)

    if (save_output) {
      self$output_ref <- out
    }

    out
  },

  get_input_relevances = function(rel_output,
                                  rule_name = "simple",
                                  rule_param = NULL) {
    if (is.null(rule_param)) {
      if (rule_name == "epsilon") {
        rule_param <- 0.001
      } else if (rule_name == "alpha_beta") {
        rule_param <- 0.5
      }
    }
    new_shape <- c(1, self$num_features, rep(1, length(self$input_dim) - 1))

    if (rule_name == "simple") {
      z <- self$output
      reshaped_gamma <- torch_tensor(self$gamma)$reshape(new_shape)
      reshaped_run_var <- torch_tensor(self$run_var)$reshape(new_shape)
      weight <- reshaped_gamma / (reshaped_run_var + self$eps)^0.5
      z <- z #+ (z == 0) + 1e-6
      rel <- (weight * self$input / z)$unsqueeze(-1) * rel_output
    } else if (rule_name == "epsilon") {
      z <- self$output
      reshaped_gamma <- torch_tensor(self$gamma)$reshape(new_shape)
      reshaped_run_var <- torch_tensor(self$run_var)$reshape(new_shape)
      weight <- reshaped_gamma / (reshaped_run_var + self$eps)^0.5
      z <- z + rule_param * (torch_sgn(z) + (z == 0))
      rel <- (weight * self$input / z)$unsqueeze(-1) * rel_output
    } else if (rule_name == "alpha_beta") {
      fact <- torch_tensor(self$gamma / (self$run_var + self$eps)^0.5)$reshape(new_shape)
      bias <- torch_tensor(- self$gamma * self$run_mean /
                             (self$run_var + self$eps)^0.5 + self$beta)$reshape(new_shape)

      z_plus <- torch_maximum(fact * self$input, 0.0)
      z_plus <- z_plus + torch_eq(z_plus, 0) * 1e-6
      z_minus <- torch_minimum(fact * self$input, 0.0)
      z_minus <- z_minus - torch_eq(z_minus, 0) * 1e-6

      rel <- rule_param * z_plus / (z_plus + torch_maximum(bias, 0.0)) +
                (1 - rule_param) * z_minus / (z_minus + torch_minimum(bias, 0.0))
      rel <- rel$unsqueeze(-1) * rel_output
    }

    rel
  },

  get_input_multiplier = function(mult_output, ...) {
    self$get_gradient(mult_output)
  },

  get_gradient = function(grad_out, ...) {
    new_shape <- c(1, self$num_features, rep(1, length(self$input_dim) - 1))
    reshaped_gamma <- torch_tensor(self$gamma)$reshape(new_shape)
    reshaped_run_var <- torch_tensor(self$run_var)$reshape(new_shape)

    (reshaped_gamma / (reshaped_run_var + self$eps)^0.5)$unsqueeze(-1) * grad_out
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
    self$rev_pad_idx <- NULL

    if (!is.null(dim_in)) {
      self$calc_rev_pad_idx(dim_in)
    }
  },

  forward = function(x, ...) {
    if (is.null(self$rev_pad_idx)) self$calc_rev_pad_idx(x$shape[-1])

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
  },

  calc_rev_pad_idx = function(dim_in) {
    num_dims <- length(dim_in)
    rev_dim_in <- rev(dim_in)
    pad <- self$pad
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


###############################################################################
#                       Global Average Pooling
###############################################################################
global_avgpool_layer <- nn_module(
  classname = "GlobalAvgPooling",
  inherit = OtherLayer,

  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
  },

  forward = function(x, ...) {
    x$mean(dim = 2, keepdim = TRUE)
  },

  update_ref = function(x_ref, ...) {
    x_ref$mean(dim = 2, keepdim = TRUE)
  },

  reshape_to_input = function(rel_output, ...) {
    out_shape <- rel_output$shape
    expand_shape <- c(out_shape[1], self$input_dim, rev(out_shape)[1])
    rel_output$expand(expand_shape) / self$input_dim[1]
  }
)

###############################################################################
#                       Global Max Pooling
###############################################################################
global_maxpool_layer <- nn_module(
  classname = "GlobalMaxPooling",
  inherit = OtherLayer,

  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$mask <- NULL
  },

  forward = function(x, ...) {
    res <- x$max(dim = 2, keepdim = TRUE)
    self$mask <- (x == res[[1]])$unsqueeze(-1)

    res[[1]]
  },

  update_ref = function(x_ref, ...) {
    x_ref$max(dim = 2, keepdim = TRUE)[[1]]
  },

  reshape_to_input = function(rel_output, ...) {

    self$mask * rel_output
  }
)

