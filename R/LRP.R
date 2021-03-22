#
# Layer-wise Relevance Propagation (LRP)
# "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation"
#       by S. Bach et al. (2015)
#

implemented_lrp_rules <- c("simple", "eps", "ab", "ww")



#' @title Layer-wise Relevance Propagation (LRP) method
#' @name func_lrp
#'
#' @description
#' This is an implementation of the \emph{Layer-wise Relevance Propagation (LRP)}
#' algorithm introduced by Bach et al. (2015). It's a local method for
#' interpreting a single element of the dataset and returns the relevance scores for
#' each input feature. The basic idea of this method is to decompose the
#' prediction score of the model with respect to the input features, i.e.
#' \deqn{f(x) \approx \sum_i R(x_i).}
#' Because of the bias vector, this decomposition is generally an approximation.
#' There exist several propagation rules to determine the relevance scores. In this
#' package are implemented: \code{\link{linear_simple_rule}},
#' \code{\link{linear_eps_rule}}, \code{\link{linear_ab_rule}},
#' \code{\link{linear_ww_rule}}.
#'
#' @param layers List of layers of type \code{\link{Dense_Layer}}.
#' @param out_class If the given model is a classification model, this
#' parameter can be used to determine which class the relevance scores should be
#' calculated for. Use the default value \code{NULL} to return the relevance
#' for all classes.
#' @param rule_name The name of the rule, with which the relevance scores are
#' calculated. Implemented are \code{"simple"}, \code{"eps"}, \code{"ab"},
#' \code{"ww"} (default: \code{"simple"}).
#' @param rule_param The parameter of the selected rule. Note: Only the rules
#' \code{"eps"} and \code{"ab"} take use of the parameter. Use the default
#' value \code{NULL} for the default parameters ("eps" : \eqn{0.01}, "ab" : \eqn{0.5}).
#'
#' @return
#' If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(#input features, #outputs)},
#' which contains the relevance scores for each input variable to the
#' output predictions. Otherwise it returns a vector of the relevance scores
#' for each input variable for the given output class.
#'
#' @examples
#' # create three dense layers
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
#' inputs <- rnorm(3)
#' output <- dense_1$forward(inputs)$out
#' output <- dense_2$forward(output)$out
#' output <- dense_3$forward(output)$out
#'
#' # calculate relevances for class 1 with simple-rule
#' func_lrp(list(dense_1, dense_2, dense_3), out_class = 1)
#'
#' # calculate relevances for all classes with simple-rule
#' func_lrp(list(dense_1, dense_2, dense_3))
#'
#' # calculate relevances for class 1 with eps-rule (eps = 0.1)
#' func_lrp(list(dense_1, dense_2, dense_3), out_class = 1, rule_name = "eps", rule_param = 0.1)
#'
#' @seealso
#' \code{\link{Analyzer}}, \code{\link{func_deeplift}}, \code{\link{func_connection_weights}},
#' \code{\link{linear_simple_rule}},
#' \code{\link{linear_eps_rule}}, \code{\link{linear_ab_rule}},
#' \code{\link{linear_ww_rule}}
#'
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46
#'
#' @export

func_lrp <- function(layers,
                out_class = NULL,
                rule_name = "simple",
                rule_param = NULL ) {

  rule <- get_Rule(rule_name, rule_param)

  # the output layer must be considered specially, i.e. without the softmax or
  # sigmoid activation

  last_layer <- layers[[length(layers)]]
  rel <- last_layer$preactivation #get output before softmax/sigmoid
  W <- last_layer$weights
  b <- last_layer$bias
  input <- last_layer$inputs

  rel <- rule(input, W, b, diag(length(rel)) * rel)

  for (layer in rev(layers)[-1]) {
    W <- layer$weights
    b <- as.vector(layer$bias)
    input <- as.vector(layer$inputs)

    rel <- rule(input, W, b, rel)
  }
  rownames(rel) <- paste0(rep("X", nrow(rel)), 1:nrow(rel))
  colnames(rel) <- paste0(rep("Y", ncol(rel)), 1:ncol(rel))
  if (is.null(out_class)) {
    return(rel)
  } else {
    if ( !(out_class %in% 1:ncol(rel) ) ) {
      stop(sprintf("Parameter 'out_class' has to be an integer value between 1 and %s! Your value: %s",
                   ncol(rel), out_class ))
    } else return(rel[, out_class])
  }
}




###-------------------------------Linear Rules----------------------------------

#' @title Simple-LRP-Rule for a dense layer
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
#' \code{\link{linear_ww_rule}}, \code{\link{func_lrp}}, \code{\link{Analyzer}}
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


#' @title Epsilon-LRP-Rule for a dense layer
#' @name linear_eps_rule
#'
#' @description
#' This is a variant of the LRP-rule \code{\link{linear_simple_rule}} with a
#' predefined stabilizer \eqn{\varepsilon > 0} for the denominator and is an implementation
#' of eq. (58) in Bach et al. (2015). Let \eqn{z_{ij}:= x_i w_{ij}} the preactivation of a hidden
#' dense layer between hidden neuron \eqn{i} and neuron \eqn{j} in the next hidden layer.
#' The relevance for a single connection is propagated by the stabilized ratio between local (\eqn{z_{ij}}) and
#' global (\eqn{z_j := b_j + \sum_{i'} z_{i'j}}) preactivation, i.e.
#' \deqn{R_{ij} = \frac{z_{ij}}{z_j + \varepsilon\ sgn(z_j)} \cdot R_j.}
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
#' \code{\link{linear_ww_rule}}, \code{\link{func_lrp}}, \code{\link{Analyzer}}
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

#' @title Alpha-Beta-LRP-Rule for a dense layer
#' @name linear_ab_rule
#'
#' @description
#' This is an alternative stabilizing method compared to the LRP-rule
#' \code{\link{linear_simple_rule}} that does not leak relevance of treating
#' negative and positive preactivation separately, i.e. we have a factor
#' \eqn{\alpha} only for the positive part and another one \eqn{\beta = 1 - \alpha} for the
#' negative part of the considered ratio between local and global preactivation:
#' \deqn{R_{ij} = ( \alpha\ \frac{z_{ij}^+}{z_j^+} + \beta\ \frac{z_{ij}^-}{z_j^-} ) \cdot R_j.}
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
#' \code{\link{linear_ww_rule}}, \code{\link{func_lrp}}, \code{\link{Analyzer}}
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

#' @title W²-LRP-Rule for a dense layer
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
#' \code{\link{linear_ab_rule}}, \code{\link{func_lrp}}, \code{\link{Analyzer}}
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
