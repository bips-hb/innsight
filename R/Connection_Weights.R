

#' @title Connection Weights method
#' @name func_connection_weights
#'
#' @description
#' This function implements the \emph{Connection Weight} method investigated by
#' Olden et al. (2004) which results in a feature importance score for each input
#' variable. The basic idea is to multiply up all path weights for each
#' possible connection between an input feature and the output and then
#' calculate the sum over them. For a neural net with \eqn{3} hidden layers with weight
#' matrices \eqn{W_1}, \eqn{W_2} and \eqn{W_3} this method results in a simple
#' matrix multiplication
#' \deqn{W_1 \cdot W_2 \cdot W_3 }.
#'
#' @param layers List of layers of type \code{\link{Dense_Layer}}.
#' @param out_class If the given model is a classification model, this
#' parameter can be used to determine which class the importance score should be
#' calculated for. Use the default value \code{NULL} to return the importance
#' for all classes.
#'
#' @return Returns a vector of the length of the input features, which contains the
#' importance scores for each input variable.
#'
#' @examples
#' # create dense layers
#' W_1 <- matrix(rnorm(3*10), nrow = 3, ncol = 10)
#' b_1 <- rnorm(10)
#' W_2 <- matrix(rnorm(10*5), nrow = 10, ncol = 5)
#' b_2 <- rnorm(5)
#' W_3 <- matrix(rnorm(5*2), nrow = 5, ncol = 2)
#' b_3 <- rnorm(2)
#'
#' dense_1 <- Dense_Layer$new(W_1, b_1, get_activation("relu"))
#' dense_2 <- Dense_Layer$new(W_2, b_2, get_activation("relu"))
#' dense_3 <- Dense_Layer$new(W_3, b_3, get_activation("softmax"))
#'
#' # calculate importance scores for class 1
#' func_connection_weights(list(dense_1, dense_2, dense_3), out_class = 1)
#'
#' # calculate importance scores for all classes
#' func_connection_weights(list(dense_1, dense_2, dense_3), out_class = NULL)
#'
#' @seealso
#' \code{\link{Analyzer}}, \code{\link{func_deeplift}}, \code{\link{func_lrp}}
#'
#' @references
#' J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
#' quantifying variable importance in artificial neural networks using
#' simulated data.} Ecological Modelling 178, p. 389–397
#'
#'@export
#'

func_connection_weights <- function(layers, out_class = NULL){
  importance <- layers[[length(layers)]]$weights
  for (layer in rev(layers)[-1]) {
    importance <- layer$weights %*% importance
  }
  rownames(importance) <- paste0(rep("X", nrow(importance)), 1:nrow(importance))
  colnames(importance) <- paste0(rep("Y", ncol(importance)), 1:ncol(importance))
  if (is.null(out_class)) {
    return(importance)
  } else {
    if ( !(out_class %in% 1:ncol(importance) ) ) {
      stop(sprintf("Parameter 'out_class' has to be an integer value between 1 and %s! Your value: %s",
                   ncol(importance), out_class ))
    } else return(importance[, out_class])
  }
}





#lrp <- function(layers,
#                out_class = 1,
#                rule_name = "simple",
#                rule_param = NULL ) {
#
#  rule <- get_Rule(rule_name, rule_param)
#
#  # the output layer must be considered specially, i.e. without the softmax or
#  # sigmoid activation
#
#  rel <- layers[[length(layers)]]$outputs_linear #get output before softmax/sigmoid
#  if ( !(out_class %in% 1:length(rel)) ) {
#    stop(sprintf("Parameter 'out_class' has to be an integer value between 1 and %s! Your value: %s",
#                 length(rel), out_class ))
#  }
#  rel <- rel[out_class]
#  W <- matrix(layers[[length(layers)]]$weights[, out_class], ncol = 1)
#  b <- layers[[length(layers)]]$bias[out_class]
#  input <- as.vector(layers[[length(layers)]]$inputs)
#
#  rel <- rule(input, W, b, rel)
#
#  for (layer in rev(layers)[-1]) {
#    W <- layer$weights
#    b <- as.vector(layer$bias)
#    input <- as.vector(layer$inputs)
#
#    rel <- rule(input, W, b, rel)
#  }
#  rel
#}
