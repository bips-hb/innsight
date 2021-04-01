


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
#' @param x_ref The reference input vector for the interpretation.
#' @param rule_name Name of the applied rule to calculate the contributions. Use one
#' of \code{"rescale"} and \code{"revealcancel"}.
#' @param out_class If the given model is a classification model, this
#' parameter can be used to determine which class the contribution should be
#' calculated for. Use the default value \code{NULL} to return the contribution
#' for all classes.
#'
#' @return
#' It returns a list of matrices of shape \emph{(in, out)},
#' which contains the contribution scores for each input variable to the
#' output predictions or single output class (if \code{out_class} is not \code{NULL})
#' for every input in \code{data}.
#'
#' @examples
#' library(neuralnet)
#' # train a NN and create an analyzer
#' nn <- neuralnet(Species ~ .,
#'                 iris, linear.output = FALSE,
#'                 hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
#' analyzer = Analyzer$new(nn)
#'
#' # calculate contributions for all classes and x_ref = 0 with rescale rule
#' result <- DeepLift(analyzer, iris[,-5])
#' plot(result)
#'
#' # calculate contributions for class 1 and x_ref first datapoint with revealcancel rule
#' result <- DeepLift(analyzer, iris[,-5], x_ref = 1, rule_name = "revealcancel", out_class = 1)
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

DeepLift <- function(analyzer, data, x_ref = NULL, rule_name = "rescale",
                     out_class = NULL) {
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assert_choice(rule_name, c("rescale", "revealcancel"))
  num_input_features <- analyzer$layers[[1]]$dim[1]
  checkmate::assert(checkmate::checkDataFrame(data, ncols = num_input_features),
                    checkmate::checkMatrix(data, ncols = num_input_features))
  checkmate::assert(checkmate::checkDataFrame(x_ref, ncols = num_input_features, nrows = 1),
                    checkmate::checkVector(x_ref, len = num_input_features),
                    checkmate::checkInt(x_ref, lower = 1, upper = nrow(data), null.ok = TRUE))
  checkmate::assertInt(out_class, null.ok = TRUE, lower = 1, upper = rev(analyzer$layers)[[1]]$dim[2])

  if (is.null(x_ref)) {
    x_ref <- as.vector(t(data[1,])) * 0
  }
  else if (length(x_ref) == 1) {
    x_ref <- as.vector(t(data[x_ref,]))
  } else {
    x_ref <- as.vector(t(x_ref))
  }

  num_inputs = nrow(data)
  contributions = vector(mode = "list", length = num_inputs)

  for (i in 1:num_inputs) {
    analyzer$update(as.vector(t(data[i,])), x_ref)
    layers = analyzer$layers

    # The last layer needs special treatment, since the activation function
    # will not be considered.
    last_layer <- layers[[length(layers)]]

    W <- last_layer$weights
    delta_x <- last_layer$inputs - last_layer$inputs_ref

    # multiplier for linear part
    multiplier_plus <- W * (W * delta_x > 0) + 0.5 * W * (delta_x == 0)
    multiplier_minus <- W * (W * delta_x < 0) + 0.5 * W * (delta_x == 0)


    # continue with the remaining layers
    for (layer in rev(layers)[-1]) {
      W <- layer$weights
      delta_x <- layer$inputs - layer$inputs_ref

      # multiplier for linear part
      mult_x_plus <- W * (W * delta_x > 0) + 0.5 * W * (delta_x == 0)
      mult_x_minus <- W * (W * delta_x < 0) + 0.5 * W * (delta_x == 0)

      # multiplier for activation
      if (rule_name == "rescale") {
        multiplier <- rescale_rule(mult_x_plus, mult_x_minus, layer)
      } else if (rule_name == "revealcancel") {
        multiplier <- reveal_cancel_rule(delta_x, mult_x_plus, mult_x_minus, layer)
      }
      multiplier_plus <- multiplier %*% multiplier_plus
      multiplier_minus <- multiplier %*% multiplier_minus
    }

    # name the output
    contrib <- (multiplier_plus + multiplier_minus) * delta_x
    rownames(contrib) <- paste0(rep("X", nrow(contrib)), 1:nrow(contrib))
    colnames(contrib) <- paste0(rep("Y", ncol(contrib)), 1:ncol(contrib))
    if (!is.null(out_class)) {
      contrib <- as.matrix(contrib[, out_class])
      colnames(contrib) <- paste0("Y", out_class)
    }
    contributions[[i]] <- contrib
  }
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
#' @param scale Scale the contribution scores to \eqn{[-1,1]}.
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
#' # plot the results scaled to -1 to 1
#' plot(result, scale = TRUE)
#'
#' @seealso [DeepLift]
#' @rdname plot.DeepLift
#' @export
#'

plot.DeepLift <- function(x, rank = FALSE, scale = FALSE, ...) {
  rule_name <- attributes(x)$rule_name
  subtitle = sprintf("%s-Rule", rule_name)
  if (rank) {
    x <- lapply(x, function(z) apply(z,2, rank))
  }
  features = c()
  labels = c()
  contribution = c()
  for (i in 1:length(x)) {
    features <- c(features, rep(rownames(x[[i]]), ncol(x[[i]])))
    labels <- c(labels, rep(colnames(x[[i]]), each = nrow(x[[i]])))
    rel <- as.vector(x[[i]])
    if (scale) {
      rel <- rel / max(abs(rel))
    }
    contribution <- c(contribution, rel)
  }
  features <- factor(features, levels = rownames(x[[1]]))
  ggplot2::ggplot(data.frame(features, labels, contribution),
                  mapping = ggplot2::aes(x = features, y = contribution, fill = labels), ...) +
    ggplot2::geom_boxplot(alpha = 0.6) +
    ggplot2::scale_fill_viridis_d() +
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
#' @param layer The hidden layer of type \code{\link{Dense_Layer}}.
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

rescale_rule <- function(mult_x_plus, mult_x_minus, layer) {
  delta_x <- layer$outputs - layer$outputs_ref
  delta_y <- layer$preactivation - layer$preactivation_ref

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

