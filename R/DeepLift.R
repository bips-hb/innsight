#' @title Deep Learning Important Features (DeepLift)
#'
#' @description
#' This is an implementation of the \emph{Deep Learning Important Features
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
#' activation functions: *rescale* rule (`'rescale'`) and
#' *reveal-cancel* rule (`'reveal_cancel'`).
#'
#' @template param-converter
#' @template param-data
#' @template param-output_idx
#' @template param-channels_first
#' @template param-ignore_last_act
#' @template param-x_ref
#' @template param-dtype
#' @template param-verbose
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

    #' @field rule_name (`character(1)`)\cr
    #' Name of the applied rule to calculate the contributions.
    #' Either `'rescale'` or `'reveal_cancel'`.\cr
    rule_name = NULL,
    x_ref = NULL,

    #' @description
    #' Create a new instance of the *DeepLift* method. When initialized,
    #' the method is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param rule_name (`character(1)`)\cr
    #' Name of the applied rule to calculate the
    #' contributions. Use either `'rescale'` or `'reveal_cancel'`. \cr
    #' @param winner_takes_all (`logical(1)`)\cr
    #' This logical argument is only relevant for MaxPooling
    #' layers and is otherwise ignored. With this layer type, it is possible that
    #' the position of the maximum values in the pooling kernel of the normal input
    #' \eqn{x} and the reference input \eqn{x'} may not match, which leads to a
    #' violation of the summation-to-delta property. To overcome this problem,
    #' another variant is implemented, which treats a MaxPooling layer as an
    #' AveragePooling layer in the backward pass only leading to an equally
    #' distribution of the upper-layer contribution to the lower layer.\cr
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

      cli_check(checkChoice(rule_name, c("rescale", "reveal_cancel")),
                "rule_name")
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
  ),

  private = list(
    print_method_specific = function() {
      i <- cli_ul()
      cli_li(paste0("{.field rule_name}: '", self$rule_name, "'"))
      cli_li(paste0("{.field winner_takes_all}: ", self$winner_takes_all))
      all_zeros <- all(unlist(lapply(self$x_ref,
                                     function(x) all(as_array(x) == 0))))
      if (all_zeros) {
        s <- "zeros"
      } else {
        values <- unlist(lapply(self$x_ref, as_array))
        s <- paste0("mean: ", mean(values), " (q1: ", quantile(values, 0.25),
                    ", q3: ", quantile(values, 0.75), ")")
      }
      cli_li(paste0("{.field x_ref}: ", s))
      cli_end(id = i)
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
