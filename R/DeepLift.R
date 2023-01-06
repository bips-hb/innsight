#' @title Deep Learning Important FeaTures (DeepLift)
#'
#' @description
#' This is an implementation of the \emph{Deep Learning Important FeaTures
#' (DeepLift)} algorithm introduced by Shrikumar et al. (2017). It's a local
#' method for interpreting a single element \eqn{x} of the dataset concerning
#' a reference value \eqn{x'} and returns the contribution of each input
#' feature from the difference of the output (\eqn{y=f(x)}) and reference
#' output (\eqn{y'=f(x')}) prediction. The basic idea of this method is to
#' decompose the difference-from-reference prediction with respect to the
#' input features, i.e.
#' \deqn{\Delta y = y - y'  = \sum_i C(x_i).}
#' Compared to \emph{Layer-wise Relevance Propagation} (see [LRP]), the
#' DeepLift method is an exact decomposition and not an approximation, so we
#' get real contributions of the input features to the
#' difference-from-reference prediction. There are two ways to handle
#' activation functions: *Rescale-rule* (`'rescale'`) and
#' *RevealCancel-rule* (`'reveal_cancel'`).
#'
#' @template param-converter
#' @template param-data
#' @template param-output_idx
#' @template param-channels_first
#' @template param-ignore_last_act
#' @template param-x_ref
#' @template param-dtype
#' @template param-verbose
#' @template param-winner_takes_all
#' @template field-x_ref
#' @template examples_DeepLift
#'
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#' @family methods
#' @export
DeepLift <- R6Class(
  classname = "DeepLift",
  inherit = InterpretingMethod,
  public = list(

    #' @field rule_name Name of the applied rule to calculate the contributions.
    #' Either `'rescale'` or `'reveal_cancel'`.
    rule_name = NULL,
    x_ref = NULL,

    #' @description
    #' Create a new instance of the *DeepLift* method. When initialized,
    #' the method is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param rule_name Name of the applied rule to calculate the
    #' contributions. Use either `'rescale'` or `'reveal_cancel'`.
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          rule_name = "rescale",
                          x_ref = NULL,
                          winner_takes_all = TRUE,
                          verbose = interactive(),
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, winner_takes_all, verbose, dtype)

      assertChoice(rule_name, c("rescale", "reveal_cancel"))
      self$rule_name <- rule_name

      if (is.null(x_ref)) {
          x_ref <- lapply(lapply(self$data, dim),
                          function(x) array(0, dim = c(1, x[-1])))
      }
      self$x_ref <- private$test_data(x_ref, name = "x_ref")

      self$converter$model$forward(self$data,
        channels_first = self$channels_first,
        save_input = TRUE,
        save_preactivation = TRUE,
        save_output = TRUE
      )
      self$converter$model$update_ref(self$x_ref,
        channels_first = self$channels_first,
        save_input = TRUE,
        save_preactivation = TRUE,
        save_output = TRUE
      )


      self$result <- private$run("DeepLift")
    }
  )
)


#'
#' @importFrom graphics boxplot
#' @exportS3Method
#'
boxplot.DeepLift <- function(x, ...) {
  x$boxplot(...)
}
