#' @title Layer-wise Relevance Propagation (LRP)
#'
#' @description
#' This is an implementation of the \emph{Layer-wise Relevance Propagation
#' (LRP)} algorithm introduced by Bach et al. (2015). It's a local method for
#' interpreting a single element of the dataset and calculates the relevance
#' scores for each input feature to the model output. The basic idea of this
#' method is to decompose the prediction score of the model with respect to
#' the input features, i.e.
#' \deqn{f(x) = \sum_i R(x_i).}
#' Because of the bias vector that absorbs some relevance, this decomposition
#' is generally an approximation. There exist several propagation rules to
#' determine the relevance scores. In this package are implemented: simple
#' rule ("simple"), \eqn{\varepsilon}-rule ("epsilon") and
#' \eqn{\alpha}-\eqn{\beta}-rule ("alpha_beta").
#'
#' @template examples-LRP
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-ignore_last_act
#' @template param-x_ref
#' @template param-dtype
#' @template param-output_idx
#' @template param-verbose
#' @template param-winner_takes_all
#'
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10,
#' p. 1-46
#'
#' @family methods
#' @export
LRP <- R6Class(
  classname = "LRP",
  inherit = InterpretingMethod,
  public = list(
    #' @field rule_name The name of the rule with which the relevance scores
    #' are calculated. Implemented are `"simple"`, `"epsilon"`, `"alpha_beta"`.
    #' However, this value
    #' can also be a named list that assigns one of these three rules to each
    #' implemented layer type separately. e.g.
    #' `list(Dense_Layer = "simple", Conv2D_Layer = "alpha_beta")`.
    #' Layers not specified in this list then use the default value `"simple"`.
    #' The implemented layer types are:\cr
    #' * 'Dense_Layer', 'Conv1D_Layer', 'Conv2D_Layer', 'BatchNorm_Layer',
    #' 'AvgPool1D_Layer', 'AvgPool2D_Layer', 'MaxPool1D_Layer' and
    #' 'MaxPool2D_Layer'
    #' @field rule_param The parameter of the selected rule. Similar to the
    #' argument `rule_name`, this can also be a named list that assigns a
    #' rule parameter to each layer type.
    rule_name = NULL,
    rule_param = NULL,


    #' @description
    #' Create a new instance of the *LRP* method. When initialized,
    #' the method is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param rule_name The name of the rule with which the relevance scores
    #' are calculated. Implemented are `"simple"`, `"epsilon"`, `"alpha_beta"`.
    #' You can pass one of the above characters to apply this rule to all
    #' possible layers. However, this value can also be a named list that
    #' assigns one of these three rules to each
    #' implemented layer type separately. e.g.
    #' `list(Dense_Layer = "simple", Conv2D_Layer = "alpha_beta")`.
    #' Layers not specified in this list then use the default value `"simple"`.
    #' The implemented layer types are:\cr
    #' * 'Dense_Layer', 'Conv1D_Layer', 'Conv2D_Layer', 'BatchNorm_Layer',
    #' 'AvgPool1D_Layer', 'AvgPool2D_Layer', 'MaxPool1D_Layer' and
    #' 'MaxPool2D_Layer'
    #' @param rule_param The parameter of the selected rule. Note: Only the
    #' rules \code{"epsilon"} and \code{"alpha_beta"} take use of the
    #' parameter. Use the default value \code{NULL} for the default parameters
    #' ("epsilon" : \eqn{0.01}, "alpha_beta" : \eqn{0.5}). Similar to the
    #' argument `rule_name`, this can also be a named list that assigns a
    #' rule parameter to each layer type. If the layer type is not specified
    #' in the named list, the default parameters will be used.
    #'
    #' @return A new instance of the R6 class `'LRP'`.
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          rule_name = "simple",
                          rule_param = NULL,
                          winner_takes_all = TRUE,
                          verbose = interactive(),
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, winner_takes_all, verbose, dtype)

      layer_names_with_rule <- c(
        "Dense_Layer", "Conv1D_Layer", "Conv2D_Layer", "BatchNorm_Layer",
        "AvgPool1D_Layer", "AvgPool2D_Layer", "MaxPool1D_Layer",
        "MaxPool2D_Layer")

      assert(
        checkChoice(rule_name, c("simple", "epsilon", "alpha_beta")),
        checkList(rule_name, types = "character", names = "named")
      )
      if (is.list(rule_name)) {
        for (name in names(rule_name)) {
          assertSubset(name, layer_names_with_rule,
                       .var.name = "names(rule_name)")
          assertChoice(rule_name[[name]],
                       c("simple", "epsilon", "alpha_beta", "pass"))
        }
      }
      self$rule_name <- rule_name

      assert(
        checkNumber(rule_param, null.ok = TRUE),
        checkList(rule_param, types = "numeric", names = "named")
      )
      if (is.list(rule_param)) {
        for (name in names(rule_param)) {
          assertSubset(name, layer_names_with_rule,
                       .var.name = "names(rule_param)")
        }
      }
      self$rule_param <- rule_param

      self$converter$model$forward(self$data,
        channels_first = self$channels_first,
        save_input = TRUE,
        save_preactivation = TRUE,
        save_output = TRUE,
        save_last_layer = TRUE
      )

      self$result <- private$run("LRP")
    }
  )
)

#'
#' @importFrom graphics boxplot
#' @exportS3Method
#'
boxplot.LRP <- function(x, ...) {
  x$boxplot(...)
}
