###############################################################################
#                   Super-class for Layers with weights
###############################################################################

InterpretingLayer <- nn_module(
  classname = "InterpretingLayer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
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
  }
)
