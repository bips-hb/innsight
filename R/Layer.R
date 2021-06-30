#'
#'Layer of a neural network
#'@description
#'Implementation of a layer of a neural network as a torch module, to be used as a
#'parent module to dense and convolutional layer modules.
#'
#' @name Layer
#' @export
Layer <- torch::nn_module(
  classname = "Layer",

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

  initialize = function() {},

  forward = function() {},

  get_activation = function(act_name) {

    activation <- get_activation(act_name)

    self$activation_f <- activation$act_func
    self$activation_name <- activation$act_name
  }
)


#
#         Layer utils
#


#'
#'@title Get activation function
#'@name get_activation
#'@description
#'This function takes the name of an activation function as input and outputs the
#'corresponding function.
#'@param act_name The name of the activation function. Implemented functions are \emph{"relu"},\emph{"leaky_relu"},\emph{"softplus"},\emph{"sigmoid"/"logistic"},\emph{"tanh"},
#'\emph{"linear"} and \emph{"softmax"}.
#'@return Returns an object \code{result} with attributes \code{result$act} and \code{result$act_name}, the activation function and name respectively.
#'
get_activation <- function(act_name) {

  result <- NULL

  if (act_name == 'relu')  {
    act <- torch::nn_relu()
  }
  else if (act_name == 'leaky_relu') {
    act <- torch::nn_leaky_relu()
  }
  else if (act_name == 'softplus') {
    act <- torch::nn_softplus()
  }
  else if (act_name %in%  c('sigmoid', 'logistic')) {
    act <- torch::nn_sigmoid()
  }
  else if (act_name == 'softmax') {
    act <- torch::nn_softmax(dim = -1)
  }
  else if (act_name == 'tanh') {
    act <- torch::nn_tanh()
  }
  else if (act_name == 'linear') {
    act <- function(x) x
  }
  else stop(sprintf("Activation function \"%s\" is not implementet yet!", act_name))

  result$act_func <- act
  result$act_name <- act_name

  result
}
