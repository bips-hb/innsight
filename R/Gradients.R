
#' @title Calculate the Gradient
#' @name Gradient
#'
#' @description
#' This method computes the gradients of the outputs with respect to the input
#' variables, i.e. for all input variable \eqn{i} and output class \eqn{j}
#' \deqn{\frac{\partial f(x)_j}{\partial x_i}.}
#'
#' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
#' @param data Either a matrix or a data frame, where each row must describe an
#' input to the network.
#' @param times_input Multiplies the gradients with the input features. This
#' method is called 'Gradient x Inputs'.
#' @param out_class If the given model is a classification model, this
#' parameter can be used to determine which class the gradients should be
#' calculated for. Use the default value \code{NULL} to return the gradients
#' for all classes.
#' @param ignore_last_act Calculate the gradients of the output or the preactivation
#' of the output (default: \code{TRUE}).
#'
#' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
#' which contains the gradients for each input variable to the
#' output predictions. Otherwise returns a vector of the gradient
#' for each input variable for the given output class.
#'
#' @export
#'

Gradient <- function(analyzer, data, times_input = TRUE, out_class = NULL,
                     ignore_last_act = TRUE) {
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assertLogical(times_input)
  num_input_features <- analyzer$layers[[1]]$dim[1]
  checkmate::assert(checkmate::checkDataFrame(data, ncols = num_input_features),
                    checkmate::checkMatrix(data, ncols = num_input_features))

  num_inputs = nrow(data)
  gradients = vector(mode = "list", length = num_inputs)

  for (i in 1:num_inputs) {
    x <- as.vector(t(data[i,]))
    analyzer$update(x)
    gradients[[i]] <- calculate_gradients(analyzer, out_class, ignore_last_act) *
      (times_input * x + (1-times_input))
  }
  class(gradients) <- c("Gradient", class(gradients))
  attributes(gradients)$times_input <- times_input
  attributes(gradients)$name <- "Gradient"
  gradients
}

#' @title SmoothGrad method
#' @name SmoothGrad
#'
#' @description
#' SmoothGrad was introduced by D. Smilkov et al. (2017) and is an extension to
#' the classical [Gradient] method.
#' This method computes the gradients of the outputs with respect to the input
#' variables, i.e. for all input variable \eqn{i} and output class \eqn{j}
#' \deqn{\frac{\partial f(x)_j}{\partial x_i}.}
#'
#' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
#' @param data Either a matrix or a data frame, where each row must describe an
#' input to the network.
#' @param n Number of perturbations of the input vector (default: \eqn{50}).
#' @param noise_level Determines the standard deviation of the gaussian
#' perturbation, i.e. \eqn{\sigma = (\max(x) - \min(x)) *} \code{noise_level}.
#' @param times_input Multiplies the gradients with the input features. This
#' method is called 'Gradient x Inputs'.
#' @param out_class If the given model is a classification model, this
#' parameter can be used to determine which class the gradients should be
#' calculated for. Use the default value \code{NULL} to return the gradients
#' for all classes.
#' @param ignore_last_act Calculate the gradients of the output or the preactivation
#' of the output (default: \code{TRUE}).
#'
#' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
#' which contains the gradients for each input variable to the
#' output predictions. Otherwise returns a vector of the gradient
#' for each input variable for the given output class.
#'
#' @references
#' D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @export
#'

SmoothGrad <- function(analyzer, data, n= 50, noise_level = 0.3,
                       times_input = TRUE, out_class = NULL, ignore_last_act = TRUE) {
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assertLogical(times_input)
  checkmate::assertInt(n, lower = 1)
  checkmate::assertNumber(noise_level, lower = 0)
  num_input_features <- analyzer$layers[[1]]$dim[1]
  checkmate::assert(checkmate::checkDataFrame(data, ncols = num_input_features),
                    checkmate::checkMatrix(data, ncols = num_input_features))

  num_inputs = nrow(data)
  smooth_gradients = vector(mode = "list", length = num_inputs)

  for (i in 1:num_inputs) {
    x <- as.vector(t(data[i,]))
    sigma <- noise_level * (max(x) - min(x))
    smooth_grad <- 0
    for (j in 1:n) {
      input_pert <- x + stats::rnorm(length(x), mean = 0, sd = sigma)
      analyzer$update(input_pert)
      smooth_grad <- smooth_grad + calculate_gradients(analyzer, out_class, ignore_last_act)
    }
    smooth_gradients[[i]] <- ( smooth_grad / n ) * (times_input * x + (1-times_input))
  }
  class(smooth_gradients) <- c("Gradient", class(smooth_gradients))
  attributes(smooth_gradients)$times_input <- times_input
  attributes(smooth_gradients)$name <- "SmoothGrad"
  smooth_gradients
}

#' @title Plot function for Gradient results
#' @name plot.Gradient
#' @description Plots the results of the \code{\link{Gradient}} method.
#'
#' @param x A result of the \code{\link{Gradient}} method.
#' @param rank If \code{TRUE}, gradient values are ranked.
#' @param scale Scale the gradient values to \eqn{[-1,1]}.
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
#' # calculate contribution scores for the whole dataset and all classes
#' result <- Gradient(analyzer, data = iris[,-5])
#'
#' # plot the results
#' plot(result)
#'
#' # plot the results with ranked relevances
#' plot(result, rank = TRUE)
#'
#' # plot the results scaled to -1 to 1
#' plot(result, scale = TRUE)
#'
#' # you can also use the SmoothGrad method
#' result <- SmoothGrad(analyzer, data = iris[,-5])
#' plot(result, scale = TRUE)
#'
#' @seealso [Gradient]
#' @rdname plot.Gradient
#' @export
#'

plot.Gradient <- function(x, rank = FALSE, scale = FALSE, ...) {
  times_input <- attributes(x)$times_input
  name <- attributes(x)$name
  if (times_input) {
    subtitle <- "Input x Gradient"
  } else {
    subtitle <- ""
  }
  if (rank) {
    x <- lapply(x, function(z) apply(z,2, rank))
  }
  features = c()
  labels = c()
  gradient = c()
  for (i in 1:length(x)) {
    features <- c(features, rep(rownames(x[[i]]), ncol(x[[i]])))
    labels <- c(labels, rep(colnames(x[[i]]), each = nrow(x[[i]])))
    rel <- as.vector(x[[i]])
    if (scale) {
      rel <- rel / max(abs(rel))
    }
    gradient <- c(gradient, rel)
  }
  ggplot2::ggplot(data.frame(features, labels, gradient),
                  mapping = ggplot2::aes(x = features, y = gradient, fill = labels)) +
    ggplot2::geom_boxplot() +
    ggplot2::scale_fill_brewer(palette = "Reds") +
    ggplot2::ggtitle(sprintf("Feature Importance with %s", name), subtitle = subtitle)
}



calculate_gradients <- function(analyzer, out_class = NULL, ignore_last_act = TRUE) {
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assertInt(out_class, null.ok = TRUE, lower = 1, upper = rev(analyzer$layers)[[1]]$dim[2])
  checkmate::assertLogical(ignore_last_act)

  layers = analyzer$layers

  last_layer <- layers[[length(layers)]]
  if (ignore_last_act) {
    act_dev <- diag(length(last_layer$preactivation))
    gradient <- act_dev %*% t(last_layer$weights)
  } else {
    act_dev <- get_deveritive_activation(last_layer$activation_name)
    gradient <- act_dev(last_layer$preactivation) %*% t(last_layer$weights)
  }

  for (layer in rev(layers)[-1]) {
    act_dev <- get_deveritive_activation(layer$activation_name)
    gradient <- gradient %*% act_dev(layer$preactivation) %*% t(layer$weights)
  }
  gradient <- t(gradient)
  rownames(gradient) <- paste0(rep("X", nrow(gradient)), 1:nrow(gradient))
  colnames(gradient) <- paste0(rep("Y", ncol(gradient)), 1:ncol(gradient))
  if (!is.null(out_class)) {
    gradient <- as.matrix(gradient[, out_class])
    colnames(gradient) <- paste0("Y", out_class)
  }
  gradient
}
