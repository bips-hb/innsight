


###---------------------------- DeepLift ---------------------------------------

#' @title Deep Learning Important FeaTures (DeepLIFT) method
#' @name func_deeplift
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
#' Compared to \emph{Layer-wise Relevance Propagation} (see \code{\link{func_lrp}}) is the
#' DeepLIFT method exact and not an approximation, so we get real contributions
#' of the input features to the difference-from-reference prediction. There are
#' two ways to handle activation functions: \emph{Rescale-Rule} and \emph{Reveal-Cancel-Rule}.
#'
#' @param layers List of layers of type \code{\link{Dense_Layer}}.
#' @param rule_name Name of the applied rule to calculate the contributions. Use one
#' of \code{"rescale"} and \code{"revealcancel"}.
#' @param out_class If the given model is a classification model, this
#' parameter can be used to determine which class the contribution should be
#' calculated for. Use the default value \code{NULL} to return the contribution
#' for all classes.
#'
#' @return
#' If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{in x out},
#' which contains the contribution values for each input variable to the
#' output predictions. Otherwise returns a vector of the contribution values
#' for each input variable for the given output class.
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
#' # do the forward pass to store all intermediate values
#' # NOTE: For DeepLift a reference input is required
#' inputs <- rnorm(3)
#' inputs_ref <- inputs * c(1,0.1, 0.01)
#' output <- dense_1$forward(inputs, inputs_ref)
#' output <- dense_2$forward(output$out, output$out_ref)
#' output <- dense_3$forward(output$out, output$out_ref)
#'
#' # print differenc-from-reference prediction before softmax activation
#' delta_out <- dense_3$preactivation - dense_3$preactivation_ref
#' delta_out
#'
#' # calculate contributions for class 1
#' contr <- func_deeplift(list(dense_1, dense_2, dense_3), out_class = 1)
#' contr
#' sum(contr) # same as delta_out[1]
#'
#' # calculate contributions for all classes
#' contr <- func_deeplift(list(dense_1, dense_2, dense_3))
#' contr
#' colSums(contr) # same as delta_out
#'
#' # calculate contributions for class 1 with Reveal-Cancel-Rule
#' contr <- func_deeplift(list(dense_1, dense_2, dense_3), rule_name = "revealcancel", out_class = 1)
#' contr
#' sum(contr) # same as delta_out[1]
#'
#' @seealso
#' \code{\link{rescale_rule}}, \code{\link{reveal_cancel_rule}},
#' \code{\link{Analyzer}}, \code{\link{func_lrp}}, \code{\link{func_connection_weights}}
#'
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#'@export


func_deeplift <- function(layers, rule_name = "rescale", out_class = NULL) {

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
    } else {
      stop(sprintf("The rule \"%s\" is not implemented yet. Use one \"rescale\" or \"revealcancel\"", rule_name))
    }
    multiplier_plus <- multiplier %*% multiplier_plus
    multiplier_minus <- multiplier %*% multiplier_minus
  }

  # name the output
  contrib <- (multiplier_plus + multiplier_minus) * delta_x
  rownames(contrib) <- paste0(rep("X", nrow(contrib)), 1:nrow(contrib))
  colnames(contrib) <- paste0(rep("Y", ncol(contrib)), 1:ncol(contrib))
  if (is.null(out_class)) {
    return(contrib)
  } else {
    if ( !(out_class %in% 1:ncol(contrib) ) ) {
      stop(sprintf("Parameter 'out_class' has to be an integer value between 1 and %s! Your value: %s",
                   ncol(contrib), out_class ))
    } else return(contrib[, out_class])
  }
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
#' \eqn{\Delta y = f(x) - f(x')}, i.e.
#' \deqn{m_{+ \to +} = m_{- \to -} = \frac{\Delta y}{\Delta x} = \frac{f(x) - f(x')}{x-x'}.}
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
#' \code{\link{reveal_cancel_rule}}, \code{\link{func_deeplift}},
#' \code{\link{Analyzer}}
#'
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#'@export

rescale_rule <- function(mult_x_plus, mult_x_minus, layer) {
  multiplier_rescale <- (layer$outputs - layer$outputs_ref) /
    (layer$preactivation - layer$preactivation_ref)

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
#' The positive and negative part of the postactivation \eqn{\Delta y = f(x) - f(x')}
#' is given by
#' \deqn{\Delta y^+ := 0.5 ( f(x' + \Delta x^+) - f(x') + f(x' + \Delta x^+ + \Delta x^-) - f(x' + \Delta x^-) )}
#' \deqn{\Delta y^- := 0.5 ( f(x' + \Delta x^-) - f(x') + f(x' + \Delta x^+ + \Delta x^-) - f(x' + \Delta x^+) ).}
#' Hence the multiplier are
#' \deqn{m_{+ \to +} = \frac{\Delta y^+}{\Delta x^+}}
#' \deqn{m_{- \to -} = \frac{\Delta y^-}{\Delta x^-}.}
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
#' \code{\link{rescale_rule}}, \code{\link{func_deeplift}},
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

#deeplift <- function(layers, rule = "rescale", out_class = 1) {
#
#  multiplier_plus = NULL
#  multiplier_minus = NULL
#
#  in_delta_x <- layers[[1]]$inputs - layers[[1]]$inputs_ref
#
#  i = 1
#  for (layer in layers) {
#    W <- layer$weights
#    delta_x <- layer$inputs - layer$inputs_ref
#
#    # multiplier for linear part
#    mult_x_plus <- W * (W * delta_x > 0) + 0.5 * W * (delta_x == 0)
#    mult_x_minus <- W * (W * delta_x < 0) + 0.5 * W * (delta_x == 0)
#
#    if (is.null(multiplier_plus)) {
#      multiplier_plus <- mult_x_plus
#      multiplier_minus <- mult_x_minus
#      multiplier <- multiplier_plus + multiplier_minus
#    } else {
#      multiplier <- multiplier_plus + multiplier_minus
#      multiplier_plus <-  multiplier %*% mult_x_plus
#      multiplier_minus <- multiplier %*% mult_x_minus
#      multiplier <- multiplier_plus + multiplier_minus
#    }
#
#    # multiplier activation
#
#    if (i < length(layers) && rule == "rescale") {
#      # Rescale rule
#      multiplier_rescale <- (layer$outputs - layer$outputs_ref) /
#        (layer$outputs_linear - layer$outputs_linear_ref)
#
#      multiplier_plus <- t(t(multiplier_plus) * multiplier_rescale)
#      multiplier_minus <- t(t(multiplier_minus) * multiplier_rescale)
#    } else if (i < length(layers) && rule == "revealcancel") {
#      # Reveal Cancel rule
#      delta_x_plus <- t(mult_x_plus) %*% delta_x
#      delta_x_minus <- t(mult_x_minus) %*% delta_x
#      x_ref <- layer$outputs_linear_ref
#      act <- layer$activation
#
#      delta_y_plus <- 0.5 * ( act(x_ref + delta_x_plus) - act(x_ref) +
#                                act(x_ref + delta_x_plus + delta_x_minus) - act(x_ref + delta_x_minus))
#
#      delta_y_minus <- 0.5 * ( act(x_ref + delta_x_minus) - act(x_ref) +
#                                 act(x_ref + delta_x_plus + delta_x_minus) - act(x_ref + delta_x_plus))
#      multiplier_rc_rule_plus <- as.vector(delta_y_plus / (delta_x_plus + 1e-16))
#      multiplier_rc_rule_minus <- as.vector(delta_y_minus / (delta_x_minus - 1e-16))
#
#      multiplier_plus <- t(t(multiplier_plus) * multiplier_rc_rule_plus)
#      multiplier_minus <- t(t(multiplier_minus) * multiplier_rc_rule_minus)
#    }
#    i <- i + 1
#  }
#  (multiplier_plus + multiplier_minus) * in_delta_x
#}
