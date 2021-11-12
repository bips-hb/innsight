#'
#' A Neural Network Layer for Interpreting its Input
#'
#' Implementation of a layer of a neural network as a torch module, to be used
#' as a parent module to dense and convolutional layer modules. The main
#' difference with the pre-implemented modules in torch is that many values
#' are stored during the forward pass.
#'
#' @section Attributes:
#' \describe{
#'   \item{`self$input_dim`}{Dimension of the input without batch dimension}
#'   \item{`self$input`}{The last recorded input for this layer}
#'   \item{`self$input_ref`}{The last recorded reference input for this layer}
#'   \item{`preactivation`}{The last recoreded preactivation of this layer}
#'   \item{`preactivation_ref`}{The last recorded reference preactivation of
#'     this layer}
#'   \item{`self$output_dim`}{The dimension of the output of this layer}
#'   \item{`self$output`}{The last recorded output of this layer}
#'   \item{`self$output_ref`}{The last recored reference output of this layer}
#'   \item{`activation_f`}{The activation function of this layer implemented
#'     in torch}
#'   \item{`activation_name`}{The name of the activation function}
#' }
#'
#'
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

  get_activation = function(act_name) {
    activation <- get_activation(act_name)

    self$activation_f <- activation$act_func
    self$activation_name <- activation$act_name
  }
)


#
#         Layer utils
#


#
# @title Get activation function
# @name get_activation
# @description
# This function takes the name of an activation function as input and outputs
# the corresponding function.
# @param act_name The name of the activation function. Implemented functions
# are \emph{"relu"},\emph{"leaky_relu"},\emph{"softplus"},
# \emph{"sigmoid"/"logistic"},\emph{"tanh"},
# \emph{"linear"} and \emph{"softmax"}.
# @return Returns an object \code{result} with attributes \code{result$act}
# and \code{result$act_name}, the activation function and name respectively.
#
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
      "Activation function \"%s\" is not implementet yet!",
      act_name
    ))
  }

  result$act_func <- act
  result$act_name <- act_name

  result
}
