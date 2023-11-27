#' @title Layer-wise relevance propagation (LRP)
#'
#' @description
#' This is an implementation of the \emph{layer-wise relevance propagation
#' (LRP)} algorithm introduced by Bach et al. (2015). It's a local method for
#' interpreting a single element of the dataset and calculates the relevance
#' scores for each input feature to the model output. The basic idea of this
#' method is to decompose the prediction score of the model with respect to
#' the input features, i.e.,
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
#' @template param-output_label
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
    #' @field rule_name (`character(1)` or `list`)\cr
    #' The name of the rule with which the relevance scores
    #' are calculated. Implemented are `"simple"`, `"epsilon"`, `"alpha_beta"`
    #' (and `"pass"` but only for 'BatchNorm_Layer'). However, this value
    #' can also be a named list that assigns one of these three rules to each
    #' implemented layer type separately, e.g.,
    #' `list(Dense_Layer = "simple", Conv2D_Layer = "alpha_beta")`.
    #' Layers not specified in this list then use the default value `"simple"`.
    #' The implemented layer types are:
    #' \tabular{lll}{
    #'  \eqn{\cdot} 'Dense_Layer' \tab \eqn{\cdot} 'Conv1D_Layer' \tab \eqn{\cdot} 'Conv2D_Layer'\cr
    #'  \eqn{\cdot} 'BatchNorm_Layer' \tab \eqn{\cdot} 'AvgPool1D_Layer' \tab \eqn{\cdot} 'AvgPool2D_Layer'\cr
    #'  \eqn{\cdot} 'MaxPool1D_Layer' \tab \eqn{\cdot} 'MaxPool2D_Layer' \tab
    #' }
    #' @field rule_param (`numeric` or `list`)\cr
    #' The parameter of the selected rule. Similar to the
    #' argument `rule_name`, this can also be a named list that assigns a
    #' rule parameter to each layer type.\cr
    rule_name = NULL,
    rule_param = NULL,


    #' @description
    #' Create a new instance of the `LRP` R6 class. When initialized,
    #' the method *LRP* is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param rule_name (`character(1)` or `list`)\cr
    #' The name of the rule with which the relevance scores
    #' are calculated. Implemented are `"simple"`, `"epsilon"`, `"alpha_beta"`.
    #' You can pass one of the above characters to apply this rule to all
    #' possible layers. However, this value can also be a named list that
    #' assigns one of these three rules to each
    #' implemented layer type separately, e.g.,
    #' `list(Dense_Layer = "simple", Conv2D_Layer = "alpha_beta")`.
    #' Layers not specified in this list then use the default value `"simple"`.
    #' The implemented layer types are:\cr
    #' \tabular{lll}{
    #'  \eqn{\cdot} 'Dense_Layer' \tab \eqn{\cdot} 'Conv1D_Layer' \tab \eqn{\cdot} 'Conv2D_Layer'\cr
    #'  \eqn{\cdot} 'BatchNorm_Layer' \tab \eqn{\cdot} 'AvgPool1D_Layer' \tab \eqn{\cdot} 'AvgPool2D_Layer'\cr
    #'  \eqn{\cdot} 'MaxPool1D_Layer' \tab \eqn{\cdot} 'MaxPool2D_Layer' \tab
    #' }
    #' *Note:* For normalization layers like 'BatchNorm_Layer', the rule
    #' `"pass"` is implemented as well, which ignores such layers in the
    #' backward pass.\cr
    #' @param rule_param (`numeric(1)` or `list`)\cr
    #' The parameter of the selected rule. Note: Only the
    #' rules \code{"epsilon"} and \code{"alpha_beta"} take use of the
    #' parameter. Use the default value \code{NULL} for the default parameters
    #' ("epsilon" : \eqn{0.01}, "alpha_beta" : \eqn{0.5}). Similar to the
    #' argument `rule_name`, this can also be a named list that assigns a
    #' rule parameter to each layer type. If the layer type is not specified
    #' in the named list, the default parameters will be used.\cr
    #'
    #' @return A new instance of the R6 class `LRP`.
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          output_label = NULL,
                          ignore_last_act = TRUE,
                          rule_name = "simple",
                          rule_param = NULL,
                          winner_takes_all = TRUE,
                          verbose = interactive(),
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx, output_label,
                       ignore_last_act, winner_takes_all, verbose, dtype)

      layer_names_with_rule <- c(
        "Dense_Layer", "Conv1D_Layer", "Conv2D_Layer", "BatchNorm_Layer",
        "AvgPool1D_Layer", "AvgPool2D_Layer", "MaxPool1D_Layer",
        "MaxPool2D_Layer")

      cli_check(c(
        checkChoice(rule_name, c("simple", "epsilon", "alpha_beta")),
        checkList(rule_name, types = "character", names = "named")
      ), "rule_name")
      if (is.list(rule_name)) {
        for (name in names(rule_name)) {
          cli_check(checkSubset(name, layer_names_with_rule),
                    "names(rule_name)")
          cli_check(checkChoice(rule_name[[name]],
                                c("simple", "epsilon", "alpha_beta", "pass")),
                    paste0("rule_name[['", name, "']]"))
        }
      }
      self$rule_name <- rule_name

      cli_check(c(
        checkNumber(rule_param, null.ok = TRUE),
        checkList(rule_param, types = "numeric", names = "named")
      ), "rule_param")
      if (is.list(rule_param)) {
        for (name in names(rule_param)) {
          cli_check(checkSubset(name, layer_names_with_rule),
                    "names(rule_param)")
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
  ),

  private = list(
    print_method_specific = function() {
      i <- cli_ul()
      if (!is.list(self$rule_name) & !is.list(self$rule_param)) {
        cli_li(paste0("{.field rule_name}: '", self$rule_name, "'"))
        if (is.null(self$rule_param)) {
          if (self$rule_name == "simple") arg <- 0
          else if (self$rule_name == "epsilon") arg <- 0.01
          else if (self$rule_name == "alpha_beta") arg <- 0.5
          cli_li(paste0("{.field rule_param}: ", arg," (default)"))
        } else {
          cli_li(paste0("{.field rule_param}: ", self$rule_param))
        }
      } else if (!is.list(self$rule_name)) {
        cli_li(paste0("{.field rule_name}: '", self$rule_name, "'"))
        cli_li("{.field rule_param}:")
        items <- lapply(self$rule_name, function(s) paste0("'", s, "'"))
        items <- append(items, list("other layers" = "NULL (default values)"))
        names(items) <- paste0(symbol$line, " ", names(items))
        cli_dl(items)
      } else if (!is.list(self$rule_param)) {
        cli_li("{.field rule_name}:")
        items <- lapply(self$rule_name, function(s) paste0("'", s, "'"))
        items <- append(items, list("other layers" = "'simple' (default value)"))
        names(items) <- paste0(symbol$line, " ", names(items))
        cli_dl(items)
        cli_li(paste0("{.field rule_param}: ", self$rule_param))
      } else {
        cli_li("{.field rule_name}:")
        items <- lapply(self$rule_name, function(s) paste0("'", s, "'"))
        items <- append(items, list("other layers" = "'simple' (default value)"))
        names(items) <- paste0(symbol$line, " ", names(items))
        cli_dl(items)

        cli_li("{.field rule_param}:")
        items <- self$rule_param
        items <- append(items, list("other layers" = "NULL (default values)"))
        names(items) <- paste0(symbol$line, " ", names(items))
        cli_dl(items)
      }

      cli_li(paste0("{.field winner_takes_all}: ", self$winner_takes_all))
      cli_end(id = i)
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
