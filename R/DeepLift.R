
###---------------------------- DeepLift ---------------------------------------

#' @title Deep Learning Important FeaTures (DeepLIFT) method
#' @name DeepLift
#'
#' @description
#' This is an implementation of the \emph{Deep Learning Important FeaTures (DeepLIFT)}
#' algorithm introduced by Shrikumar et al. (2017). It's a local method for
#' interpreting a single element \eqn{x} of the dataset concerning a reference value \eqn{x'}
#' and returns the contribution of each input feature from the difference of the
#' output (\eqn{y=f(x)}) and reference output (\eqn{y'=f(x')}) prediction.
#' The basic idea of this method is to decompose the difference-from-reference
#' prediction with respect to the input features, i.e.
#' \deqn{\Delta y = y - y'  = \sum_i C(x_i).}
#' Compared to \emph{Layer-wise Relevance Propagation} (see \code{\link{LRP}}) is the
#' DeepLIFT method exact and not an approximation, so we get real contributions
#' of the input features to the difference-from-reference prediction. There are
#' two ways to handle activation functions: \emph{Rescale-Rule} and \emph{Reveal-Cancel-Rule}.
#'
#' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
#' @param data Either a matrix or a data frame, where each row must describe an
#' input to the network.
#' @param x_ref The reference input vector for the interpretation. You can also
#' pass an index from the given data set \code{data} which will be used as the
#' reference input. With the default value \code{NULL} you use an input of zeros.
#' @param rule_name Name of the applied rule to calculate the contributions. Use one
#' of \code{"rescale"} and \code{"revealcancel"}.
#'
#' @return
#' It returns an array of size \emph{(dim_in, dim_out, num_data)} which
#' contains the contribution scores for each input variable to the
#' output predictions for each element in the given data.
#'
#' @examples
#' library(neuralnet)
#' # train a NN and create an analyzer
#' nn <- neuralnet(Species ~ .,
#'                 iris, linear.output = FALSE,
#'                 hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
#' analyzer = Analyzer$new(nn)
#'
#' # calculate contributions for x_ref = 0 with rescale rule
#' result <- DeepLift(analyzer, iris[,-5])
#' plot(result)
#'
#' # calculate contributions for x_ref first datapoint with 'revealcancel' rule
#' result <- DeepLift(analyzer, iris[,-5], x_ref = 1, rule_name = "revealcancel")
#' plot(result)
#'
#' # compare class 'setosa' with 'virginica'
#' result <- DeepLift(analyzer, iris[iris$Species == "setosa",-5],
#'                    x_ref = colMeans(iris[iris$Species == "virginica",-5]),
#'                    rule_name = "revealcancel")
#' plot(result)
#'
#' @seealso
#' \code{\link{rescale_rule}}, \code{\link{reveal_cancel_rule}},
#' \code{\link{Analyzer}}
#'
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#'@export
#'

DeepLift <- function(analyzer, data, x_ref = NULL, rule_name = "rescale") {

  # Check arguments
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assert_choice(rule_name, c("rescale", "revealcancel"))
  dim_in <- analyzer$dim_in
  num_inputs = nrow(data)
  checkmate::assert(checkmate::checkDataFrame(data, ncols = dim_in),
                    checkmate::checkMatrix(data, ncols = dim_in))
  checkmate::assert(checkmate::checkDataFrame(x_ref, ncols = dim_in, nrows = 1),
                    checkmate::checkVector(x_ref, len = dim_in),
                    checkmate::checkInt(x_ref, lower = 1, upper = num_inputs, null.ok = TRUE))

  # Get the reference input
  # NULL: create an input vector of zeros
  if (is.null(x_ref)) {
    x_ref <- as.vector(t(data[1,])) * 0
  }
  # numeric: take the x_ref-th row of the given data as reference value
  else if (length(x_ref) == 1) {
    x_ref <- as.vector(t(data[x_ref,]))
  }
  # otherwise the argument 'x_ref' is the reference input
  else {
    x_ref <- as.vector(t(x_ref))
  }

  # Update the analyzer and set up some variables
  analyzer$update(as.matrix(data), x_ref)

  layers = analyzer$layers
  contributions = array(NA, dim = c(dim_in, analyzer$dim_out, num_inputs))
  dimnames(contributions) <- list(analyzer$feature_names,
                                  analyzer$response_names,
                                  paste0(rep("B", num_inputs), 1:num_inputs))

  # The last layer needs special treatment, since the activation function
  # will not be considered.
  last_layer <- layers[[length(layers)]]
  W_ll <- last_layer$weights

  for (i in 1:num_inputs) {
    delta_x <- last_layer$inputs[i,] - last_layer$inputs_ref

    # multiplier for linear part
    multiplier_plus <- W_ll * (W_ll * delta_x > 0) + 0.5 * W_ll * (delta_x == 0)
    multiplier_minus <- W_ll * (W_ll * delta_x < 0) + 0.5 * W_ll * (delta_x == 0)


    # continue with the remaining layers
    for (layer in rev(layers)[-1]) {
      W <- layer$weights
      delta_x <- layer$inputs[i,] - layer$inputs_ref

      # multiplier for linear part
      mult_x_plus <- W * (W * delta_x > 0) + 0.5 * W * (delta_x == 0)
      mult_x_minus <- W * (W * delta_x < 0) + 0.5 * W * (delta_x == 0)

      # multiplier for activation
      if (rule_name == "rescale") {
        multiplier <- rescale_rule(mult_x_plus, mult_x_minus,
                                   layer$outputs[i,] - layer$outputs_ref,
                                   layer$preactivation[i,] - layer$preactivation_ref)
      } else if (rule_name == "revealcancel") {
        multiplier <- reveal_cancel_rule(delta_x, mult_x_plus, mult_x_minus, layer)
      }

      # Combine multiplier with upper layer contributions
      multiplier_plus <- multiplier %*% multiplier_plus
      multiplier_minus <- multiplier %*% multiplier_minus
    }

    # Calculate the final contribution scores
    contributions[,,i] <-  (multiplier_plus + multiplier_minus) * delta_x
  }

  # Store class and other attributes
  class(contributions) <- c("DeepLift", class(contributions))
  attributes(contributions)$rule_name <- rule_name
  contributions
}

#' @title Plot function for DeepLift results
#' @name plot.DeepLift
#' @description Plots the results of the \code{\link{DeepLift}} method.
#'
#' @param x A result of the \code{\link{DeepLift}} method.
#' @param rank If \code{TRUE}, contribution scores are ranked.
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
#' result <- DeepLift(analyzer, data = iris[,-5])
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
#' @seealso [DeepLift]
#' @rdname plot.DeepLift
#' @export
#'

plot.DeepLift <- function(x, rank = FALSE, scale = FALSE, ...) {
  rule_name <- attributes(x)$rule_name
  subtitle = sprintf("%s-Rule", rule_name)
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
  Contribution = as.vector(x)
  Features <- factor(features, levels = dimnames(x)[[1]])
  ggplot2::ggplot(data.frame(Features, Class, Contribution),
                  mapping = ggplot2::aes(x = Features, y = Contribution, fill = Class), ...) +
    ggplot2::geom_boxplot(alpha = 0.6) +
    ggplot2::scale_fill_viridis_d() +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::ggtitle("Feature Importance with DeepLift", subtitle = subtitle)
}


#' @title DeepLIFT: Rescale-Rule
#' @name rescale_rule
#'
#' @description
#' Implementation of the \emph{Rescale Rule} introduced by Shrikumar et al. (2017)
#' to determine a multiplier for an activation function.
#' This rule defines the same multiplier for the negative and positive
#' contributions by the simple ratio between difference-from-reference preactivation
#' \eqn{\Delta x = x-x'} and difference-from-reference postactivation
#' \eqn{\Delta y = \sigma(x) - \sigma(x')}, i.e.
#' \deqn{m_{\Delta x^+ \Delta y^+} = m_{\Delta x^- \Delta y^-} = \frac{\Delta y}{\Delta x} = \frac{\sigma(x) - \sigma(x')}{x-x'}.}
#' Afterward the contribution \eqn{m_{\Delta x \Delta t}} is calculated and returned.
#'
#' @param mult_x_plus The multiplier from the upper positive difference-from-reference
#' value to the output, i.e. \eqn{m_{\Delta y^+ \Delta t}}
#' @param mult_x_minus The multiplier from the upper negative difference-from-reference
#' value to the output, i.e. \eqn{m_{\Delta y^- \Delta t}}.
#' @param delta_x Difference-from-reference of the preactivation.
#' @param delta_y Difference-from-reference of the postactivation.
#'
#' @return
#' Returns the multiplier from the difference-from-reference preactivation to
#' the output, i.e. \eqn{m_{\Delta x \Delta t}}.
#'
#' @seealso
#' \code{\link{reveal_cancel_rule}}, \code{\link{DeepLift}},
#' \code{\link{Analyzer}}
#'
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#'@export

rescale_rule <- function(mult_x_plus, mult_x_minus, delta_x, delta_y) {

  # add a numeric stabilizer
  multiplier_rescale <- delta_x / (delta_y + 1e-16 * ((delta_y >= 0)*2 -1 ))

  t(t(mult_x_plus + mult_x_minus) * multiplier_rescale)
}

#' @title DeepLIFT: Reveal-Cancel-Rule
#' @name reveal_cancel_rule
#'
#' @description
#' Implementation of the \emph{Reveal Cancel Rule} introduced by Shrikumar et al. (2017)
#' to determine a multiplier for an activation function.
#' This rule defines different multipliers for the negative and positive
#' contributions between difference-from-reference preactivation and postactivation.
#' The positive and negative part of the postactivation \eqn{\Delta y = \sigma(x) - \sigma(x')}
#' is given by
#' \deqn{\Delta y^+ := 0.5 ( \sigma(x' + \Delta x^+) - \sigma(x') + \sigma(x' + \Delta x^+ + \Delta x^-) - \sigma(x' + \Delta x^-) )}
#' \deqn{\Delta y^- := 0.5 ( \sigma(x' + \Delta x^-) - \sigma(x') + \sigma(x' + \Delta x^+ + \Delta x^-) - \sigma(x' + \Delta x^+) ).}
#' Hence the multiplier are
#' \deqn{m_{\Delta x^+ \Delta y^+} = \frac{\Delta y^+}{\Delta x^+}}
#' \deqn{m_{\Delta x^- \Delta y^-} = \frac{\Delta y^-}{\Delta x^-}.}
#' Afterward the contribution \eqn{m_{\Delta x \Delta t}} is calculated and returned.
#'
#' @param delta_x Difference-from-reference preactivation of the current layer.
#' @param mult_x_plus The multiplier from the upper positive difference-from-reference
#' value to the output, i.e. \eqn{m_{\Delta y^+ \Delta t}}
#' @param mult_x_minus The multiplier from the upper negative difference-from-reference
#' value to the output, i.e. \eqn{m_{\Delta y^- \Delta t}}.
#' @param layer The hidden layer of type \code{\link{Dense_Layer}}.
#'
#' @return
#' Returns the multiplier from the difference-from-reference preactivation to
#' the output, i.e. \eqn{m_{\Delta x \Delta t}}.
#'
#' @seealso
#' \code{\link{rescale_rule}}, \code{\link{DeepLift}},
#' \code{\link{Analyzer}}
#'
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#'@export

reveal_cancel_rule <- function(delta_x, mult_x_plus, mult_x_minus, layer) {
  delta_x_plus <- t(mult_x_plus) %*% delta_x
  delta_x_minus <- t(mult_x_minus) %*% delta_x
  x_ref <- layer$preactivation_ref
  act <- layer$activation

  delta_y_plus <- 0.5 * ( act(x_ref + delta_x_plus) - act(x_ref) +
                            act(x_ref + delta_x_plus + delta_x_minus) - act(x_ref + delta_x_minus))

  delta_y_minus <- 0.5 * ( act(x_ref + delta_x_minus) - act(x_ref) +
                             act(x_ref + delta_x_plus + delta_x_minus) - act(x_ref + delta_x_plus))
  multiplier_rc_rule_plus <- as.vector(delta_y_plus / (delta_x_plus + 1e-16))
  multiplier_rc_rule_minus <- as.vector(delta_y_minus / (delta_x_minus - 1e-16))

  t(t(mult_x_plus) * multiplier_rc_rule_plus) + t(t(mult_x_minus) * multiplier_rc_rule_minus)
}

