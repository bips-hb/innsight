#' Connection Weights
#'
#' @description
#' This class implements the *Connection Weights* method investigated by
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
#' method *Gradient x Input* (see [Gradient]). Hence, the local variant is
#' simply the point-wise product of the global *Connection Weights* method and
#' the input data. You can use this variant by setting the `times_input`
#' argument to `TRUE` and providing input data.
#'
#' @template examples-ConnectionWeights
#' @template param-converter
#' @template param-data-optional
#' @template param-channels_first
#' @template param-dtype
#' @template param-output_idx
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
    #' @field times_input This logical value indicates whether the results from
    #' the *Connection Weights* method were multiplied by the provided input
    #' data or not. Thus, this value specifies whether the original global
    #' variant of the method or the local one was applied. If the value is
    #' `TRUE`, then data is provided in the field `data`.
    times_input = NULL,

    #' @description
    #' Create a new instance of the *Connection Weights* method. When
    #' initialized, the method is applied to the given data and the results
    #' are stored in the field `result`.
    #'
    #' @param times_input Multiplies the results with the input features.
    #' This variant tuns the global *Connection Weights* method into a local
    #' one. Default: `FALSE`.
    initialize = function(converter,
                          data = NULL,
                          output_idx = NULL,
                          channels_first = TRUE,
                          times_input = FALSE,
                          verbose = interactive(),
                          dtype = "float") {
      assertClass(converter, "Converter")
      self$converter <- converter

      assertLogical(channels_first)
      self$channels_first <- channels_first

      assertLogical(times_input)
      self$times_input <- times_input

      assertLogical(verbose)
      self$verbose <- verbose

      assertChoice(dtype, c("float", "double"))
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      # Check output indices
      self$output_idx <- check_output_idx(output_idx, converter$output_dim)

      if (times_input & is.null(data)) {
        stopf(
          "If you want to use the ConnectionWeights method with the ",
          "'times_input' argument, you must also specify 'data'! ",
          call = "ConnectionWeights$new(...)"
        )
      } else if (times_input) {
        self$data <- private$test_data(data)
      } else {
        if (!is.null(data)) {
          messagef(
            "If 'times_input' = FALSE, then the method 'ConnectionWeights' ",
            "is a global method and independent of the data. Therefore, the ",
            "argument 'data' will be ignored."
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
  )
)


#' @importFrom graphics boxplot
#' @exportS3Method
boxplot.ConnectionWeights <- function(x, ...) {
  x$boxplot(...)
}
