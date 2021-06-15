
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

  get_activation = function(act_name, ...) {

    activation <- get_activation(act_name, ...)

    self$activation_f <- activation$act_func
    self$activation_name <- activation$act_name
  }
)


#
#         Layer utils
#

get_activation <- function(act_name) {

  result <- NULL

  if (is.character(act_name)) {
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
  }
  else if (is.function(act_name)) {
    tryCatch({
      #self$act_dev <- Deriv::Deriv(act_name)
      result$act_func <- act_name
      result$act_name <- "custom"
    },
    error = function(error_message) {
      message("There was an error in the calculation of the derivative.")
      message(error_message)
    })
  }

  result
}



