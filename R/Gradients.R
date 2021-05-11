
#' @title Calculate the Gradient
#' @name Gradient
#'
#' @description
#' This method computes the gradients of the outputs with respect to the input
#' variables, i.e. for all input variable \eqn{i} and output class \eqn{j}
#' \deqn{d f(x)_j / d x_i.}
#'
#' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
#' @param data Either a matrix or a data frame, where each row must describe an
#' input to the network.
#' @param times_input Multiplies the gradients with the input features. This
#' method is called 'Gradient x Inputs'.
#' @param ignore_last_act Calculate the gradients of the output or the preactivation
#' of the output (default: \code{TRUE}).
#'
#' @return Returns an array of size \emph{(dim_in, dim_out, num_data)} which
#' contains the gradients for each input variable to the
#' output predictions for each element in the given data.
#'
#' @examples
#' library(neuralnet)
#' nn <- neuralnet( Species ~ .,
#'                  iris, linear.output = FALSE,
#'                  hidden = c(10,8), act.fct = "tanh", rep = 1, threshold = 0.5)
#' # create an analyzer for this model
#' analyzer = Analyzer$new(nn)
#'
#' # calculate gradients for the whole dataset
#' Gradient(analyzer, data = iris[,-5])
#'
#' @export
#'

Gradient <- function(analyzer, data, times_input = FALSE, ignore_last_act = TRUE) {

  # Check format of arguments
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assertLogical(times_input)
  dim_in <- analyzer$dim_in
  batch_size <- nrow(data)
  checkmate::assert(checkmate::checkDataFrame(data, ncols = dim_in),
                    checkmate::checkMatrix(data, ncols = dim_in))

  # Update the analyzer with the given data
  analyzer$update(as.matrix(data))

  # Calculate the gradients, returns an array of size [d_in, d_out, batch_size]
  gradients <- calculate_gradients(analyzer, ignore_last_act)

  if (times_input) {
    # Create multiplication array of size [d_in, d_out, batch_size] with
    # d_out-times stacked input values
    mult_array <- array(apply(as.matrix(data), 1,
                              function(x) rep(x, times = analyzer$dim_out)),
                        dim = dim(gradients))
    gradients <- gradients * mult_array
  }

  # Store class and other attributes
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
#' the classical [Gradient] method. It takes the mean of the gradients for \code{n}
#' perturbations of each data point, i.e. with \eqn{\epsilon ~ N(0,\sigma)}
#' \deqn{1/n \sum_n d f(x+ \epsilon)_j / d x_j.}
#'
#' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
#' @param data Either a matrix or a data frame, where each row must describe an
#' input to the network.
#' @param n Number of perturbations of the input vector (default: \eqn{50}).
#' @param noise_level Determines the standard deviation of the gaussian
#' perturbation, i.e. \eqn{\sigma = (max(x) - min(x)) *} \code{noise_level}.
#' @param times_input Multiplies the gradients with the input features. This
#' method is called 'Gradient x Inputs'.
#' @param ignore_last_act Calculate the gradients of the output or the preactivation
#' of the output (default: \code{TRUE}).
#'
#' @return Returns an array of size \emph{(dim_in, dim_out, num_data)} which
#' contains the smoothed gradients for each input variable to the
#' output predictions for each element in the given data.
#'
#' @references
#' D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @export
#'

SmoothGrad <- function(analyzer, data, n= 50, noise_level = 0.1,
                       times_input = FALSE, ignore_last_act = TRUE) {

  # Check format of arguments
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assertLogical(times_input)
  checkmate::assertInt(n, lower = 1)
  checkmate::assertNumber(noise_level, lower = 0)
  dim_in <- analyzer$layers[[1]]$dim[1]
  batch_size <- nrow(data)
  checkmate::assert(checkmate::checkDataFrame(data, ncols = dim_in),
                    checkmate::checkMatrix(data, ncols = dim_in))

  # Generate 'n' perturbations of the given data wrt the 'noise level' and add these
  # perturbed data points immediately after the original data point.
  # The resulting data matrix has a size of [batch_size * n, dim_in]
  #print("----------------------- The Data---------------------")
  noise_scale <- noise_level * (apply(as.matrix(data), 2, max) - apply(as.matrix(data), 2, min))
  #print(noise_scale)
  inputs <- matrix(rep(as.matrix(data), each = n), ncol = dim_in)
  noise <- matrix(stats::rnorm(batch_size * n * dim_in, mean = 0, sd = noise_scale), ncol = dim_in, byrow = TRUE)
  #print(inputs)

  #print("------------------------ The noise ------------------")
  #print(noise)

  input_pert <- inputs + noise
  #input_pert <- matrix(apply(as.matrix(data), 1,
  #                           function(x) {
  #                             x + stats::rnorm(dim_in*n,
  #                                              mean = 0,
  #                                              sd = noise_scale )
  #                             }
  #                           ), ncol = dim_in, byrow = TRUE)

  # Update the analyzer with the given data
  analyzer$update(input_pert)

  # Calculate the gradients, returns an array of size [d_in, d_out, batch_size * n]
  gradients <- calculate_gradients(analyzer, ignore_last_act)
  #print("---------------the gradients---------------")
  #analyzer$update(as.matrix(data))
  #print(calculate_gradients(analyzer, ignore_last_act))

  #print("---------------the gradients (stacked noise)-----------")
  #print(gradients)

  # In order to calculate the mean of all the n calculations for one data point,
  # we define a variable of shape (d_in, d_out, batch_size) with reasonable dimnames
  smooth_gradients <- array(NA,
                       dim = c(dim(gradients)[1:2], batch_size),
                       dimnames = list(dimnames(gradients)[[1]],
                                       dimnames(gradients)[[2]],
                                       dimnames(gradients)[[3]][1:batch_size]))

  # Reshape gradients to [d_in, d_out, batch_size, n]
  dim(gradients) <- c(dim(gradients)[1:2], n, batch_size)
  #print("------------------ Reshaped Gradients---------------")
  #print(gradients)

  # Calculate the mean over the fourth dimension and save the result in the variable
  # 'smooth_gradients'
  #print("----------------- In apply function ----------------")
  #apply(gradients, 4,
  #      function(x) {
  #        print("----Start-----")
  #        print(x[,,1])
  #        print(apply(x, c(1,2), mean))
  #        print("----End-------")
  #      })
  smooth_gradients[] <- array(apply(gradients, 4,
                                    function(x) apply(x, c(1,2), mean))
                              ,dim =dim(smooth_gradients))

  if (times_input) {
    # Create multiplication array of size [d_in, d_out, batch_size] with
    # d_out-times stacked input values
    dim_out <- dim(smooth_gradients)[2]
    mult_array <- array(apply(as.matrix(data), 1,
                              function(x) rep(x, times = dim_out)),
                        dim = dim(smooth_gradients))
    smooth_gradients <- smooth_gradients * mult_array
  }

  # Store class and other attributes
  class(smooth_gradients) <- c("Gradient", class(smooth_gradients))
  attributes(smooth_gradients)$times_input <- times_input
  attributes(smooth_gradients)$name <- "SmoothGrad"
  smooth_gradients
}

#' @title Plot function for Gradient results
#' @name plot.Gradient
#' @description Plots the results of the \code{\link{Gradient}} or
#' \code{\link{SmoothGrad}} method.
#'
#' @param x A result of the \code{\link{Gradient}} or \code{\link{SmoothGrad}}
#' method.
#' @param rank If \code{TRUE}, gradient values are ranked.
#' @param scale Scale the gradient values to the centered 90% of the calculated
#' values.
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
#' # cut the lower and upper 5% of the data and plot the result
#' plot(result, scale = TRUE)
#'
#' # you can also use the SmoothGrad method
#' result <- SmoothGrad(analyzer, data = iris[,-5])
#' plot(result, scale = TRUE)
#'
#' @seealso [Gradient], [SmoothGrad]
#' @rdname plot.Gradient
#' @export
#'

plot.Gradient <- function(x, rank = FALSE, scale = FALSE, ...) {
  # Get relevant attributes
  times_input <- attributes(x)$times_input
  name <- attributes(x)$name
  if (times_input) {
    subtitle <- "Input x Gradient"
  } else {
    subtitle <- ""
  }

  features = rep(dimnames(x)[[1]], dim(x)[2]*dim(x)[3])
  Class = rep(dimnames(x)[[2]], each = dim(x)[1], times = dim(x)[3])
  if (rank) {
    x[] <- apply(x, 3, function(z) apply(z,2, rank))
    y_min <- 1
    y_max <- dim(x)[1]
  } else if (scale) {
    y_min <- stats::quantile(x, 0.05)
    y_max <- stats::quantile(x, 0.95)
  } else {
    y_min <- min(x)
    y_max <- max(x)
  }
  Gradient = as.vector(x)
  Features <- factor(features, levels = dimnames(x)[[1]])
  ggplot2::ggplot(data.frame(Features, Class, Gradient),
                  mapping = ggplot2::aes(x = Features, y = Gradient, fill = Class)) +
    ggplot2::geom_boxplot(alpha = 0.6) +
    ggplot2::scale_fill_viridis_d() +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::ggtitle(sprintf("Feature Importance with %s", name), subtitle = subtitle)
}


calculate_gradients <- function(analyzer, ignore_last_act = TRUE) {

  # Check arguments
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assertLogical(ignore_last_act)

  layers = analyzer$layers
  last_layer <- layers[[length(layers)]]
  batch_size <- nrow(last_layer$preactivation)
  dim_out <- analyzer$dim_out
  dim_in <- analyzer$dim_in

  if (ignore_last_act) {
    # stacked identity matrices (size [dim_out * batch, dim_out]) to allow
    # batch-wise evaluation
    act_dev <- matrix(rep(diag(dim_out), times = batch_size),
                      nrow = dim_out * batch_size,
                      ncol = dim_out ,
                      byrow = TRUE)

    # transposed weight matrix has a size of [dim_out, h_in], hence the gradient
    # has a size of [dim_out * batch, h_in]
    gradient <- act_dev %*% t(last_layer$weights)
  } else {
    act_dev <- get_deveritive_activation(last_layer$activation_name)
    d_act <- matrix(rep(act_dev(last_layer$preactivation), each = dim_out), nrow = dim_out * batch_size)

    # gradient size [dim_out * batch, h_out]
    # act_dev size [dim_out * batch, h_out]
    # transposed weight matrix [h_out, h_in]
    gradient <- d_act %*% t(last_layer$weights)
    # new gradient has size [dim_out * batch, h_in]
  }
  for (layer in rev(layers)[-1]) {
    act_dev <- get_deveritive_activation(layer$activation_name)
    d_act <- matrix(rep(act_dev(layer$preactivation), each = dim_out), nrow = dim_out * batch_size)

    # gradient size [dim_out * batch, h_out]
    # act_dev size [dim_out * batch, h_out]
    # transposed weight matrix [h_out, h_in]
    gradient <- (gradient * d_act) %*% t(layer$weights)
    # new gradient has size [dim_out * batch, h_in]
  }
  gradient <- array(t(gradient), dim = c(dim_in, dim_out, batch_size))
  dimnames(gradient) <- list(analyzer$feature_names,
                             analyzer$response_names,
                             paste0(rep("B", batch_size), 1:batch_size))
  gradient
}

#library(neuralnet)
#library(ggplot2)
#data(iris)

#nn <- neuralnet(Species ~ ., iris, c(8,6,4), threshold = 0.1, linear.output = FALSE)

#ana <- Analyzer$new(nn)

#res_1 <- SmoothGrad(ana, data = iris[1:10,-5], n = 500, noise_level = 0.1, times_input = FALSE)
#res_2 <- Gradient(ana, iris[1:10,-5], times_input = FALSE)


#mean((res_1 - res_2)^2)

#res_1 <- as.vector(res_1)
#res_2 <- as.vector(res_2)

#r <- data.frame(Gradient = res_2, SmoothGrad = res_1)
#ggplot(r, aes(Gradient, SmoothGrad)) +
#  geom_point(alpha = 0.5)

