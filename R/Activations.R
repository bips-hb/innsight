# Activations of common Neural Neural Networks
#

###-----------------------Activations-------------------------------------------

relu <- function(x) {
  ifelse(x >= 0, x, 0)
}

relu_dev <- function(x) {
  r <- ifelse(x >= 0, 1, 0)
  diag(length(r)) * r
}

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

sigmoid_dev <- function(x) {
  r <- sigmoid(x) * (1 - sigmoid(x))
  diag(length(r)) * r
}

softplus <- function(x) {
  log( exp(x) + 1)
}

softplus_dev <- function(x) {
  r <- sigmoid(x)
  diag(length(r)) * r
}

softmax <- function(x) {
  exp(x) / sum(exp(x))
}

softmax_dev <- function(x) {
  s <- as.vector(softmax(x))
  t(diag(length(s)) * s - s^2)
}

linear <- function(x) {
  x
}

linear_dev <- function(x) {
  diag(length(x))
}

tanh_dev <- function(x) {
  r <- 1 - tanh(x)^2
  diag(length(r)) * r
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
#'

get_activation <- function(name) {
  if (name == "relu") return(relu)
  else if (name == "softplus") return(softplus)
  else if (name == "sigmoid" || name == "logistic") return(sigmoid)
  else if (name == "softmax") return(softmax)
  else if (name == "tanh") return(tanh)
  else if (name == "linear") return(linear)
  else stop(sprintf("Activation function \"%s\" is not implementet yet!", name))
}

#' @title Get deveritive of activation function by name
#' @name get_deveritive_activation
#'
#' @description
#' This is a getter method for all implemented deveritives of an activation functions. Use one of
#' \code{"relu"}, \code{"softplus"}, \code{"sigmoid"} / \code{"logistic"},
#' \code{"softmax"}, \code{"tanh"}, \code{"linear"}.
#'
#' @param name Name of the activation function.
#'
#' @returns
#' Returns the deveritive of the activation function.
#'
#' @export
#'
get_deveritive_activation <- function(name) {
  if (name == "relu") return(relu_dev)
  else if (name == "softplus") return(softplus_dev)
  else if (name == "sigmoid" || name == "logistic") return(sigmoid_dev)
  else if (name == "softmax") return(softmax_dev)
  else if (name == "tanh") return(tanh_dev)
  else if (name == "linear") return(linear_dev)
  else stop(sprintf("Deveritive of activation function \"%s\" is not implementet yet!", name))
}
