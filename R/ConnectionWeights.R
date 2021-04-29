
#' @title Connection Weights method
#' @name Connection_Weights
#'
#' @description
#' This function implements the \emph{Connection Weight} method investigated by
#' Olden et al. (2004) which results in a feature importance score for each input
#' variable. The basic idea is to multiply up all path weights for each
#' possible connection between an input feature and the output and then
#' calculate the sum over them. For a neural network with \eqn{3} hidden layers with weight
#' matrices \eqn{W_1}, \eqn{W_2} and \eqn{W_3} this method results in a simple
#' matrix multiplication
#' \deqn{W_1 * W_2 * W_3. }
#'
#' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
#'
#' @return It returns a matrix of shape \emph{(in, out)},
#' which contains the importance scores for each input variable to the
#' output predictions.
#'
#' @examples
#' library(neuralnet)
#' # train a neural net on the iris dataset and create analyzer
#' nn <- neuralnet(Species ~ ., iris, linear.output = FALSE,
#'                 hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
#' analyzer = Analyzer$new(nn)
#'
#' # calculate importance for all classes
#' result <- Connection_Weights(analyzer)
#' plot(result)
#'
#' # plot the importance in a ranked scale
#' plot(result, rank = TRUE)
#'
#' @seealso
#' \code{\link{Analyzer}}, [plot.ConnectionWeights]
#'
#'
#' @references
#' J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
#' quantifying variable importance in artificial neural networks using
#' simulated data.} Ecological Modelling 178, p. 389–397
#'
#'@export
#'

Connection_Weights <- function(analyzer){
  checkmate::assertClass(analyzer, "Analyzer")
  layers = analyzer$layers
  importance <- layers[[length(layers)]]$weights
  for (layer in rev(layers)[-1]) {
    importance <- layer$weights %*% importance
  }

  rownames(importance) <- analyzer$feature_names
  colnames(importance) <-analyzer$response_names

  class(importance) <- c("ConnectionWeights", class(importance))
  importance
}

#' @title Plot function for Connection Weight results
#' @name plot.ConnectionWeights
#' @description Plots the results of the \code{\link{Connection_Weights}} method.
#'
#' @param x A result of the \code{\link{Connection_Weights}} method.
#' @param rank If \code{TRUE}, importance scores are ranked.
#' @param scale Scale the importance scores to \eqn{[-1,1]}.
#' @param ... Other arguments passed on to methods. Not currently used.
#'
#' @return Returns a ggplot2 plot object.
#'
#' @examples
#' library(neuralnet)
#' nn <- neuralnet( Species ~ .,
#'                  iris, linear.output = FALSE,
#'                  hidden = c(10,8), act.fct = "tanh", rep = 1, threshold = 0.5)
#' # create an analyzer for this model
#' analyzer = Analyzer$new(nn)
#'
#' # calculate importance scores for all classes
#' result <- Connection_Weights(analyzer)
#'
#' # plot the results
#' plot(result)
#'
#' # plot the results with ranked importance scores
#' plot(result, rank = TRUE)
#'
#' # plot the results scaled to -1 to 1
#' plot(result, scale = TRUE)
#'
#' @seealso [Connection_Weights]
#' @export
#'
plot.ConnectionWeights <- function(x, rank = FALSE, scale = FALSE, ...) {
  if (is.vector(x)) {
    x <- matrix(x, ncol = 1)
    colnames(x) <- c("Y")
  }
  if (scale) {
    x <- x / max(abs(x))
  }
  if (rank) {
    x <- apply(x,2, rank) / nrow(x)
  }
  features <- rep(rownames(x), ncol(x))
  Features <- factor(features, levels = rownames(x))
  Class <- rep(colnames(x), each = nrow(x))
  Importance <- as.vector(x)
  ggplot2::ggplot(data.frame(Features, Class, Importance), ggplot2::aes(fill=Class, y=Importance, x=Features)) +
    ggplot2::geom_bar(position="dodge", stat="identity", color = "black", alpha = 0.6) +
    ggplot2::scale_fill_viridis_d() +
    ggplot2::ggtitle("Feature Importance with Connection Weights")
}
