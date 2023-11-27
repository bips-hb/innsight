#' Connection weights method
#'
#' @description
#' This class implements the *Connection weights* method investigated by
#' Olden et al. (2004), which results in a relevance score for each input
#' variable. The basic idea is to multiply all path weights for each
#' possible connection between an input feature and the output node and then
#' calculate the sum over them. Besides, it is originally a global
#' interpretation method and independent of the input data. For a neural
#' network with \eqn{3} hidden layers with weight matrices \eqn{W_1},
#' \eqn{W_2} and \eqn{W_3}, this method results in a simple matrix
#' multiplication independent of the activation functions in between:
#' \deqn{W_1 * W_2 * W_3.}
#'
#' In this package, we extended this method to a local method inspired by the
#' method *Gradient\eqn{\times}Input* (see [`Gradient`]). Hence, the local variant is
#' simply the point-wise product of the global *Connection weights* method and
#' the input data. You can use this variant by setting the `times_input`
#' argument to `TRUE` and providing input data.
#'
#' @template examples-ConnectionWeights
#' @template param-converter
#' @template param-data-optional
#' @template param-channels_first
#' @template param-dtype
#' @template param-output_idx
#' @template param-output_label
#' @template param-verbose
#'
#' @references
#' * J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
#'  quantifying variable importance in artificial neural networks using
#'  simulated data.} Ecological Modelling 178, p. 389–397
#'
#' @family methods
#' @export
ConnectionWeights <- R6Class(
  classname = "ConnectionWeights",
  inherit = InterpretingMethod,
  public = list(
    #' @field times_input (`logical(1)`)\cr
    #' This logical value indicates whether the results from
    #' the *Connection weights* method were multiplied by the provided input
    #' data or not. Thus, this value specifies whether the original global
    #' variant of the method or the local one was applied. If the value is
    #' `TRUE`, then data is provided in the field `data`.
    times_input = NULL,

    #' @description
    #' Create a new instance of the class `ConnectionWeights`. When
    #' initialized, the method is applied and the results
    #' are stored in the field `result`.
    #'
    #' @param times_input (`logical(1)`)\cr
    #' Multiplies the results with the input features.
    #' This variant turns the global *Connection weights* method into a local
    #' one. Default: `FALSE`.\cr
    initialize = function(converter,
                          data = NULL,
                          output_idx = NULL,
                          output_label = NULL,
                          channels_first = TRUE,
                          times_input = FALSE,
                          verbose = interactive(),
                          dtype = "float") {
      cli_check(checkClass(converter, "Converter"), "converter")
      self$converter <- converter

      cli_check(checkLogical(channels_first), "channels_first")
      self$channels_first <- channels_first

      cli_check(checkLogical(times_input), "times_input")
      self$times_input <- times_input

      cli_check(checkLogical(verbose), "verbose")
      self$verbose <- verbose

      cli_check(checkChoice(dtype, c("float", "double")), "dtype")
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      # Check output indices and labels
      outputs <- check_output_idx(output_idx, converter$output_dim,
                                  output_label, converter$output_names)
      self$output_idx <- outputs[[1]]
      self$output_label <- outputs[[2]]

      if (times_input & is.null(data)) {
        stopf(
          "If you want to use the {.emph ConnectionWeights} method with the ",
          "{.arg times_input} argument, you must also specify {.arg data}! "
        )
      } else if (times_input) {
        self$data <- private$test_data(data)
      } else {
        if (!is.null(data)) {
          messagef(
            "If {.arg times_input} = FALSE, then the method {.emph ConnectionWeights} ",
            "is a global method and independent of the data. Therefore, the ",
            "argument {.arg data} will be ignored."
          )
        }
        # Set only a single data index
        self$data <- list(torch_tensor(1))
      }

      self$ignore_last_act <- FALSE

      result <- private$run("Connection-Weights")

      if (self$times_input) {
        result <- calc_times_input(result, self$data)
      }

      self$result <- result
    }
  ),
  private = list(
    print_method_specific = function() {
      i <- cli_ul()
      if (self$times_input) {
        cli_li(paste0("{.field times_input}:  TRUE (",
                      symbol$arrow_right,
                      " local {.emph ConnectionWeights} method)"))
      } else {
        cli_li(paste0("{.field times_input}:  FALSE (",
                      symbol$arrow_right,
                      " global {.emph ConnectionWeights} method)"))
      }
      cli_end(id = i)
    }
  )
)


#' @importFrom graphics boxplot
#' @exportS3Method
boxplot.ConnectionWeights <- function(x, ...) {
  x$boxplot(...)
}
