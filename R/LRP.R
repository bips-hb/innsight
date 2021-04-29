#
# Layer-wise Relevance Propagation (LRP)
# "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation"
#       by S. Bach et al. (2015)
#
implemented_lrp_rules <- c("simple", "eps", "ab", "ww")



#' @title Layer-wise Relevance Propagation (LRP) method
#' @name LRP
#'
#' @description
#' This is an implementation of the \emph{Layer-wise Relevance Propagation (LRP)}
#' algorithm introduced by Bach et al. (2015). It's a local method for
#' interpreting a single element of the dataset and returns the relevance scores for
#' each input feature. The basic idea of this method is to decompose the
#' prediction score of the model with respect to the input features, i.e.
#' \deqn{f(x) = \sum_i R(x_i).}
#' Because of the bias vector, this decomposition is generally an approximation.
#' There exist several propagation rules to determine the relevance scores. In this
#' package are implemented: \code{\link{linear_simple_rule}},
#' \code{\link{linear_eps_rule}}, \code{\link{linear_ab_rule}},
#' \code{\link{linear_ww_rule}}.
#'
#' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
#' @param data Either a matrix or a data frame, where each row must describe an
#' input to the network.
#' @param rule_name The name of the rule, with which the relevance scores are
#' calculated. Implemented are \code{"simple"}, \code{"eps"}, \code{"ab"},
#' \code{"ww"} (default: \code{"simple"}).
#' @param rule_param The parameter of the selected rule. Note: Only the rules
#' \code{"eps"} and \code{"ab"} take use of the parameter. Use the default
#' value \code{NULL} for the default parameters ("eps" : \eqn{0.01}, "ab" : \eqn{0.5}).
#'
#' @return
#' It returns a list of matrices of shape \emph{(in, out)},
#' which contains the relevance scores for each input variable to the
#' output predictions or single output class (if \code{out_class} is not \code{NULL})
#' for every input in \code{data}.
#'
#' @examples
#' library(neuralnet)
#' data(iris)
#' nn <- neuralnet( Species ~ .,
#'                  iris, linear.output = FALSE,
#'                  hidden = c(10,8), act.fct = "tanh", rep = 1, threshold = 0.5)
#' # create an analyzer for this model
#' analyzer = Analyzer$new(nn)
#'
#' # calculate relevance scores for the whole dataset
#' result <- LRP(analyzer, data = iris[,-5], rule_name = "simple")
#' plot(result)
#'
#' # calculate relevance scores for the whole dataset with eps-rule (eps = 0.1)
#' result <- LRP(analyzer, data = iris[,-5], rule_name = "eps", rule_param = 0.1)
#' plot(result)
#'
#' @seealso
#' \code{\link{Analyzer}},
#' \code{\link{linear_simple_rule}},
#' \code{\link{linear_eps_rule}}, \code{\link{linear_ab_rule}},
#' \code{\link{linear_ww_rule}}, [plot.LRP]
#'
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46
#'
#' @export

LRP <- function(analyzer,
                data,
                rule_name = "simple",
                rule_param = NULL ) {

  # Check arguments and set default rule parameters
  checkmate::assertClass(analyzer, "Analyzer")
  checkmate::assert_choice(rule_name, implemented_lrp_rules)
  if (is.null(rule_param)) {
    if (rule_name == "eps"){
      rule_param = 0.01
    } else if (rule_name == "ab") {
      rule_param = 0.5
    }
  }
  rule <- get_Rule(rule_name, rule_param)
  dim_in <- analyzer$dim_in
  dim_out <- analyzer$dim_out
  num_inputs = nrow(data)

  checkmate::assert(checkmate::checkDataFrame(data, ncols = dim_in),
                    checkmate::checkMatrix(data, ncols = dim_in))

  # Update the analyzer and define some variables
  analyzer$update(as.matrix(data))

  layers = analyzer$layers
  relevances = array(NA, dim = c(dim_in, dim_out, num_inputs))
  dimnames(relevances) <- list(analyzer$feature_names,
                               analyzer$response_names,
                               paste0(rep("B", num_inputs), 1:num_inputs))

  # The last layer needs special treatment, since the activation function
  # will not be considered.
  last_layer <- layers[[length(layers)]]

  for (i in 1:num_inputs) {
      # last layer
      rel <- rule(last_layer$inputs[i,],
                  last_layer$weights,
                  last_layer$bias,
                  diag(dim_out) * last_layer$preactivation[i,])

      # other layers
      for (layer in rev(layers)[-1]) {

        rel <- rule(layer$inputs[i,],
                    layer$weights,
                    layer$bias,
                    rel)
      }
      relevances[,,i] <- rel
  }

  # Store class and other attributes
  class(relevances) <- c("LRP", class(relevances))
  attributes(relevances)$rule_name <- rule_name
  attributes(relevances)$rule_param <- rule_param
  relevances
}

#' @title Plot function for LRP results
#' @name plot.LRP
#' @description Plots the results of the \code{\link{LRP}} method.
#'
#' @param x A result of the \code{\link{LRP}} method.
#' @param rank If \code{TRUE}, relevance scores are ranked.
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
#' # calculate relevance scores for the whole dataset
#' result <- LRP(analyzer, data = iris[,-5], rule_name = "simple")
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
#' @seealso [LRP]
#' @rdname plot.LRP
#' @export
#'

plot.LRP <- function(x, rank = FALSE, scale = FALSE, ...) {
  rule_name <- attributes(x)$rule_name
  rule_param <- attributes(x)$rule_param
  if (rule_name %in% c("eps", "ab")) {
    subtitle = sprintf("%s-Rule (%s)", rule_name, rule_param)
  } else {
    subtitle = sprintf("%s-Rule", rule_name)
  }
  features = rep(dimnames(x)[[1]], dim(x)[2]*dim(x)[3])
  Class = rep(dimnames(x)[[2]], each = dim(x)[1], times = dim(x)[3])
  if (rank) {
    x[] <- apply(x, 3, function(z) apply(z,2, rank))
    y_min <- 0.9
    y_max <- dim(x)[2]+1 + 0.1
  } else if (scale) {
    y_min <- stats::quantile(x, 0.05)
    y_max <- stats::quantile(x, 0.95)
  } else {
    y_min <- min(x)
    y_max <- max(x)
  }
  Relevance = as.vector(x)
  Features <- factor(features, levels = dimnames(x)[[1]])
  ggplot2::ggplot(data.frame(Features, Class, Relevance),
                  mapping = ggplot2::aes(x = Features, y = Relevance, fill = Class), ...) +
    ggplot2::geom_boxplot(alpha = 0.6) +
    ggplot2::scale_fill_viridis_d() +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::ggtitle("Feature Importance with Layerwise Relevance Propagation", subtitle = subtitle)
}


###-------------------------------Linear Rules----------------------------------

#' @title LRP: Simple-Rule for a dense layer
#' @name linear_simple_rule
#'
#' @description
#' This is the simplest rule for the LRP method and is an implementation of eq. (56)
#' in Bach et al. (2015). Let \eqn{z_{ij}:= x_i w_{ij}} the preactivation of a hidden
#' dense layer between hidden neuron \eqn{i} and neuron \eqn{j} in the next hidden layer.
#' The relevance for a single connection is propagated by the ratio between local (\eqn{z_{ij}}) and
#' global (\eqn{z_j := b_j + \sum_{i'} z_{i'j}}) preactivation, i.e.
#' \deqn{R_{ij} = \frac{z_{ij}}{z_j} \cdot R_j.}
#' Then the relevance of the neuron \eqn{i} is given by the sum over the relevances
#' of all possible outgoing connections, i.e.
#' \deqn{R_i = \sum_j R_{ij}.}
#'
#' @param input The input vector with length \emph{dim_in} of the hidden layer.
#' @param weight The weight matrix with size \emph{(dim_in ,dim_out)} of the
#' hidden layer.
#' @param bias The bias vector with length \emph{dim_out} of the hidden layer.
#' @param relevance A matrix with the relevance scores from each neuron of the upper
#' hidden layer to each output neuron of the model.
#'
#' @return
#' Returns the relevance scores from each neuron in the lower hidden layer to each
#' output neuron of the model as a matrix.
#'
#' @seealso
#' \code{\link{linear_eps_rule}}, \code{\link{linear_ab_rule}},
#' \code{\link{linear_ww_rule}}, \code{\link{LRP}}
#'
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46
#'
#'@export

linear_simple_rule <- function(input, weight, bias, relevance){

  z_ij <- weight * input
  z_j <- colSums(z_ij) + bias


  # adding a small stabilizer in case z_j is zero
  z_ij %*% (relevance / ( z_j + 1e-16 * ((z_j >= 0)*2 -1 ) ) )# as.vector
}


#' @title LRP: Epsilon-Rule for a dense layer
#' @name linear_eps_rule
#'
#' @description
#' This is a variant of the LRP-rule \code{\link{linear_simple_rule}} with a
#' predefined stabilizer \eqn{\epsilon > 0} for the denominator and is an implementation
#' of eq. (58) in Bach et al. (2015). Let \eqn{z_{ij}:= x_i w_{ij}} the preactivation of a hidden
#' dense layer between hidden neuron \eqn{i} and neuron \eqn{j} in the next hidden layer.
#' The relevance for a single connection is propagated by the stabilized ratio between local (\eqn{z_{ij}}) and
#' global (\eqn{z_j := b_j + \sum_{i'} z_{i'j}}) preactivation, i.e.
#' \deqn{R_{ij} = \frac{z_{ij}}{z_j + \epsilon\ sgn(z_j)} \cdot R_j.}
#' Then the relevance of the neuron \eqn{i} is given by the sum over the relevances
#' of all possible outgoing connections, i.e.
#' \deqn{R_i = \sum_j R_{ij}.}
#'
#' @param input The input vector with length \emph{dim_in} of the hidden layer.
#' @param weight The weight matrix with size \emph{(dim_in ,dim_out)} of the
#' hidden layer.
#' @param bias The bias vector with length \emph{dim_out} of the hidden layer.
#' @param relevance A matrix with the relevance scores from each neuron of the upper
#' hidden layer to each output neuron of the model.
#' @param eps Value of the predefined stabilizer (default: \eqn{0.01}).
#'
#' @return
#' Returns the relevance scores from each neuron in the lower hidden layer to each
#' output neuron of the model as a matrix.
#'
#' @seealso
#' \code{\link{linear_simple_rule}}, \code{\link{linear_ab_rule}},
#' \code{\link{linear_ww_rule}}, \code{\link{LRP}}
#'
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46
#'
#'@export

linear_eps_rule <- function(input, weight, bias, relevance, eps = 0.01) {

  z_ij <- weight * input
  z_j <- colSums(z_ij) + bias

  z_ij %*% (relevance / ( z_j + eps * ((z_j >= 0)*2 -1 ) ) )
}

#' @title LRP: Alpha-Beta-Rule for a dense layer
#' @name linear_ab_rule
#'
#' @description
#' This is an alternative stabilizing method compared to the LRP-rule
#' \code{\link{linear_simple_rule}} that does not leak relevance of treating
#' negative and positive preactivation separately, i.e. we have a factor
#' \eqn{\alpha} only for the positive part and another one \eqn{\beta = 1 - \alpha} for the
#' negative part of the considered ratio between local and global preactivation:
#' \deqn{R_{ij} = ( \alpha \frac{z_{ij}^+}{z_j^+} + \beta \frac{z_{ij}^-}{z_j^-} ) \cdot R_j.}
#' Then the relevance of the neuron \eqn{i} is given by the sum over the relevances
#' of all possible outgoing connections, i.e.
#' \deqn{R_i = \sum_j R_{ij}.}
#'
#' @param input The input vector with length \emph{dim_in} of the hidden layer.
#' @param weight The weight matrix with size \emph{(dim_in ,dim_out)} of the
#' hidden layer.
#' @param bias The bias vector with length \emph{dim_out} of the hidden layer.
#' @param relevance A matrix with the relevance scores from each neuron of the upper
#' hidden layer to each output neuron of the model.
#' @param alpha Value of the factor for the positive contribution (default: \eqn{0.5}).
#'
#' @return
#' Returns the relevance scores from each neuron in the lower hidden layer to each
#' output neuron of the model as a matrix.
#'
#' @seealso
#' \code{\link{linear_simple_rule}}, \code{\link{linear_eps_rule}},
#' \code{\link{linear_ww_rule}}, \code{\link{LRP}}
#'
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46
#'
#'@export

linear_ab_rule <- function(input, weight, bias, relevance, alpha = 0.5) {

  z_ij <- weight * input
  z_plus <- colSums(relu(z_ij)) + relu(bias)

  z_minus <- colSums(-relu(-z_ij)) - relu(-bias)

  relu(z_ij) %*% (relevance * alpha / (z_plus + 1e-16)) - relu(-z_ij) %*% (relevance * (1-alpha) / (z_minus - 1e-16))
}

#' @title LRP: W²-Rule for a dense layer
#' @name linear_ww_rule
#'
#' @description
#' This propagation rule is independent of the input values and takes the ratio
#' between the squared weight \eqn{w_{ij}} and the sum of all squared weights to
#' neuron \eqn{j} (\eqn{\sum_{i'} w_{i'j}^2}). It's an implementation of eq. (12) in
#' Montavon et al. (2005), i.e.
#' \deqn{R_{ij} = \frac{w_{ij}^2}{\sum_k w_{kj}^2} \cdot R_j. }
#' Then the relevance of the neuron \eqn{i} is given by the sum over the relevances
#' of all possible outgoing connections, i.e.
#' \deqn{R_i = \sum_j R_{ij}.}
#'
#' @param weight The weight matrix with size \emph{(dim_in, dim_out)} of the
#' hidden layer.
#' @param relevance A matrix with the relevance scores from each neuron of the upper
#' hidden layer to each output neuron of the model.
#'
#' @return
#' Returns the relevance scores from each neuron in the lower hidden layer to each
#' output neuron of the model as a matrix.
#'
#' @seealso
#' \code{\link{linear_simple_rule}}, \code{\link{linear_eps_rule}},
#' \code{\link{linear_ab_rule}}, \code{\link{LRP}}
#'
#' @references
#' G. Montavon et al. (2015) \emph{Explaining nonLinear classification decisions
#' with deep taylor decomposition.} CoRR
#'
#'@export

linear_ww_rule <- function(weight, relevance) {

  ( t(t(weight^2) / colSums(weight^2)) ) %*% relevance
}

get_Rule <- function(rule_name, rule_param = NULL) {
  if (rule_name == "simple") {
    rule = linear_simple_rule
  }
  else if (rule_name == "eps") {
    if (is.null(rule_param)) rule_param = 0.01
    rule <- function(input, weight, bias, rel) linear_eps_rule(input, weight, bias, rel, eps = rule_param)
  }
  else if (rule_name == "ab") {
    if (is.null(rule_param)) rule_param = 0.5
    rule <- function(input, weight, bias, rel) linear_ab_rule(input, weight, bias, rel, alpha = rule_param)
  }
  else if (rule_name == "ww") {
    rule <- function(input, weight, bias, rel) linear_ww_rule(weight, rel)
  }
  else {
    stop(sprintf("The LRP-Rule \"%s\" is not implemented yet. Use one of \"%s\"",
                 rule_name, paste0(implemented_lrp_rules, collapse = "\", \"")))
  }
  rule
}
