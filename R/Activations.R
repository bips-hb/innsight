# Activations of common Neural Neural Networks
#

###-----------------------Activations-------------------------------------------

relu <- function(x) {
  ifelse(x >= 0, x, 0)
}

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

softplus <- function(x) {
  log( exp(x) + 1)
}

softmax <- function(x) {
  exp(x) / sum(exp(x))
}

linear <- function(x) {
  x
}

#' @title Get activation function by name
#' @name get_activation
#'
#' @description
#' This is a getter method for all implemented activation functions. Use one of
#' \code{"relu"}, \code{"softplus"}, \code{"sigmoid"} / \code{"logistic"},
#' \code{"softmax"}, \code{"tanh"}, \code{"linear"}.
#'
#' @param name Name of the activation function.
#'
#' @returns
#' Returns the activation function.
#'
#' @export
get_activation <- function(name) {
  if (name == "relu") return(relu)
  else if (name == "softplus") return(softplus)
  else if (name == "sigmoid" || name == "logistic") return(sigmoid)
  else if (name == "softmax") return(softmax)
  else if (name == "tanh") return(tanh)
  else if (name == "linear") return(linear)
  else stop(sprintf("Activation function \"%s\" is not implementet yet!", name))
}
