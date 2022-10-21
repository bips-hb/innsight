###############################################################################
#                   Super-class for Layers with weights
###############################################################################

InterpretingLayer <- nn_module(
  classname = "InterpretingLayer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  preactivation = NULL,
  preactivation_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
  activation_f = NULL,
  activation_name = NULL,
  W = NULL,
  b = NULL,
  dtype = NULL,

  initialize = function() {
  },

  forward = function() {
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$preactivation <- NULL
    self$preactivation_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
  },

  set_dtype = function(dtype) {
    if (dtype == "float") {
      self$W <- self$W$to(torch_float())
      self$b <- self$b$to(torch_float())
    } else if (dtype == "double") {
      self$W <- self$W$to(torch_double())
      self$b <- self$b$to(torch_double())
    } else {
      stop("Unknown argument for 'dtype' : '", dtype, "'. ",
           "Use 'float' or 'double' instead!")
    }
    self$dtype <- dtype
  },

  get_activation = function(act_name) {
    activation <- get_activation(act_name)

    self$activation_f <- activation$act_func
    self$activation_name <- activation$act_name
  },

  get_stabilizer = function() {
    # Get stabilizer
    if (self$dtype == "float") {
      eps <- 1e-6 # a bit larger than torch_finfo(torch_float())$eps
    } else {
      eps <- 1e-15 # a bit larger than torch_finfo(torch_double())$eps
    }

    eps
  }

)


###############################################################################
#                                Utils
###############################################################################

get_activation <- function(act_name) {
  result <- NULL

  if (act_name == "relu") {
    act <- nn_relu()
  } else if (act_name == "leaky_relu") {
    act <- nn_leaky_relu()
  } else if (act_name == "softplus") {
    act <- nn_softplus()
  } else if (act_name %in% c("sigmoid", "logistic")) {
    act <- nn_sigmoid()
  } else if (act_name == "softmax") {
    act <- nn_softmax(dim = -1)
  } else if (act_name == "tanh") {
    act <- nn_tanh()
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
