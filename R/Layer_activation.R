
activation_layer <- nn_module(
  classname = "Activation_Layer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,

  act_name = NULL,
  act_FUN = NULL,

  initialize = function(act_name, dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out

    act <- get_activation(act_name)
    self$act_name <- act$act_name
    self$act_FUN <- act$act_func
  },

  forward = function(x, save_input = FALSE, save_output = FALSE) {
    if (save_input) {
      self$input <- x
    }
    out <- self$act_FUN(x)
    if (save_output) {
      self$output <- out
    }

    out
  },

  update_ref = function(x_ref, save_input = FALSE, save_output = FALSE) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    out_ref <- self$act_FUN(x_ref)
    if (save_output) {
      self$output_ref <- out_ref
    }

    out_ref
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
  },

  set_dtype = function(dtype) {
    self$dtype <- dtype
  }
)

###############################################################################
#                                Utils
###############################################################################

get_activation <- function(act_name) {
  result <- NULL

  if (act_name == "relu") {
    act <- nnf_relu
  } else if (act_name == "leaky_relu") {
    act <- nnf_leaky_relu
  } else if (act_name == "softplus") {
    act <- nnf_softplus
  } else if (act_name %in% c("sigmoid", "logistic")) {
    act <- nnf_sigmoid
  } else if (act_name == "softmax") {
    act <- function(x) nnf_softmax(x, dim = -1)
  } else if (act_name == "tanh") {
    act <- torch_tanh
  } else if (act_name == "linear") {
    act <- function(x) x
  } else {
    stop(sprintf(
      "Activation function '%s' is not implementet yet!",
      act_name
    ))
  }

  result$act_func <- act
  result$act_name <- act_name

  result
}
