library(R6)

###------------------------Dense Layer------------------------------------------

#' Dense layer of a Neural Network
#'
#'
#' @description
#' Implementation of a regular densely-connected Neural Network layer as an R6 class,
#' where input, preactivation and output values of the last forward pass are stored
#' (same for a reference input, if this is needed). Applies
#' a linear transformation followed by an activation function \eqn{\sigma} to the incoming
#' data, i.e.
#' \deqn{y = \sigma (W^T x + b).}
#'
#' @examples
#' # Weight matrix of shape (number input features x number outputs)
#' # for example (3,5)
#' W <- matrix(c(1, -1, 2, 4, -1,
#'               2, -5, 3, 1, -2,
#'               3, 4, -1, -2, 4), nrow = 3, ncol = 5, byrow = TRUE)
#' # Bias vector of length (number outputs)
#' b <- c(-2, 1, 3, 4, -5)
#' # Activation function
#' sigmoid <- function(x) 1 / (1 + exp(-x))
#' # Create a new dense layer
#' my_layer <- Dense_Layer$new(weights = W, bias = b, activation = sigmoid)
#'
#' # example forward pass with no reference input
#' inputs <- c(-1,2,3)
#' my_layer$forward(x = inputs)
#'
#' # get preactivation of the last forward pass
#' my_layer$preactivation
#'
#' @export
#'


Dense_Layer <- R6Class("Dense_Layer",
public = list(

    #' @field type Type of the layer (in this case \code{"Dense"}).
    #' @field dim Dimension of this layer, i.e. \emph{(dim_in, dim_out)}.
    #' @field weights The weight matrix \eqn{W} of the dense layer with size
    #' \emph{(dim_in, dim_out)}.
    #' @field bias Bias vector \eqn{b} of the linear transformation. It's a vector of
    #' length \emph{dim_out}.
    #' @field activation Activation function \eqn{\sigma} to turn the linear
    #' transformation into a non-linear one.
    #' @field inputs Save the inputs from the last forward pass of this layer.
    #' If there was no call of method \href{#method-forward}{\code{Dense_Layer$forward}} yet
    #' then this value is \code{NULL}.
    #' @field preactivation Save the outputs of the linear transformation from
    #' the last forward pass of this layer. If there was no call of method
    #' \href{#method-forward}{\code{Dense_Layer$forward}} yet then this value is \code{NULL}.
    #' @field outputs Save the outputs of the whole layer from the last forward pass.
    #' If there was no call of method \href{#method-forward}{\code{Dense_Layer$forward}} yet
    #' then this value is \code{NULL}.
    #' @field inputs_ref Save the reference inputs from the last forward pass of this layer.
    #' If there was no call of method \href{#method-forward}{\code{Dense_Layer$forward}} yet
    #' then this value is \code{NULL}.
    #' @field preactivation_ref Save the reference outputs of the linear transformation from
    #' the last forward pass of this layer. If there was no call of method
    #' \href{#method-forward}{\code{Dense_Layer$forward}} yet then this value is \code{NULL}.
    #' @field outputs_ref Save the reference outputs of the whole layer from the last forward pass.
    #' If there was no call of method \href{#method-forward}{\code{Dense_Layer$forward}} yet
    #' then this value is \code{NULL}.

    type = NULL, # dense, conv
    dim = NULL, # c(input_dim, output_dim)
    weights = NULL, # input x output matrix
    bias = NULL, # vector (output_dim)
    activation = NULL,
    inputs = NULL, # vector (input_dim)
    preactivation = NULL, # vector (output_dim)
    outputs = NULL, # vector (output_dim)
    inputs_ref = NULL, # vector (input_dim)
    preactivation_ref = NULL, # vector (output_dim)
    outputs_ref = NULL, # vector (output_dim)

    #' @description
    #' Create a new instance of this class with given parameters.
    #'
    #' @param weights The weight matrix of dimension \emph{(dim_in , dim_out)} for the linear transformation.
    #' @param bias The bias vector of length \emph{dim_out} for the linear transformation.
    #' @param activation The activation function of the dense layer.
    #'
    #' @return A new instance of the R6 class 'Dense_Layer' with the given parameters.
    #'

    initialize = function(
      weights,
      bias,
      activation
      ) {
      self$type = "Dense"
      if (!is.matrix(weights)) {
        stop("Parameter 'weights' has to be a matrix of numbers.")
      }
      self$dim = dim(weights)
      self$weights = weights
      if (!is.vector(bias)) {
        stop("Parameter 'bias' has to be a vector of numbers.")
      } else if (length(bias) != ncol(weights)) {
        stop(sprintf("Missmatch between weight matrix and bias vector. Number of weight columns %s is unequal to the length of the bias vector %s!", ncol(weights), length(bias)))
      }
      self$bias = bias
      self$activation = activation
    },

    #' @description
    #' The forward method of the dense layer. This method calculates the linear
    #' transformation with the following non-linearity for the input \code{x} and,
    #' if needed, the reference value \code{x_ref}. All the intermediate values are
    #' stored in the class attributes: \code{inputs}, \code{preactivation},
    #' \code{outputs}, \code{inputs_ref}, \code{preactivation_ref},
    #' \code{outputs_ref}.
    #'
    #' @param x Input vector for this layer.
    #' @param x_ref Reference input vector for this layer. It is only needed for
    #' the DeepLift method (see \code{\link{func_deeplift}}) and can otherwise be set to \code{NULL}.
    #'
    #' @return A list of two vectors. The first one is the output of this
    #' layer for the input \code{x} and the second entry is the output for the
    #' reference input \code{x_ref}.
    #'

    forward = function(x, x_ref = NULL) {
      if (!is.vector(x) || !is.double(x)) {
        stop("Input value has to be a vector of numbers!")
      }
      else if (length(x) != self$dim[1]) {
        stop(sprintf("Wrong size of the input vector. Expected %s, your length %s .", self$dim[1], length(x)))
      }

      self$inputs_ref = x_ref
      if ( !is.null(x_ref) ) {
        if (!is.vector(x) || !is.double(x)) {
          stop("Reference input value has to be a vector of numbers!")
        }
        else if (length(x_ref) != self$dim[1]) {
          stop(sprintf("Wrong size of the input vector. Expected %s, your length %s .", self$dim[1], length(x_ref)))
        }
       self$preactivation_ref = as.vector(t(self$weights) %*% x_ref + self$bias)
       self$outputs_ref = self$activation(self$preactivation_ref)
      } else {
       self$outputs_ref = NULL
       self$preactivation_ref = NULL
      }

      self$inputs = x
      self$preactivation = as.vector(t(self$weights) %*% x + self$bias)
      self$outputs = self$activation(self$preactivation)

      list(out = self$outputs, out_ref = self$outputs_ref)

    }
)
)
