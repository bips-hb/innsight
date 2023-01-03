#' Connection Weights
#'
#' @description
#' This class implements the \emph{Connection Weights} method investigated by
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
#' simply the pointwise product of the global *Connection Weights* method and
#' the input data. You can use this variant by setting the `times_input`
#' argument to `TRUE` and providing input data.
#'
#' @template examples-ConnectionWeights
#' @template param-converter
#' @template param-data-optional
#' @template param-channels_first
#' @template param-dtype
#' @template param-preprocess_FUN
#' @template param-as_plotly
#' @template param-aggr_channels
#' @template param-ref_data_idx
#' @template param-individual_data_idx
#' @template param-individual_max
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
    #' @param output_idx These indices specify the output nodes for which
    #' the method is to be applied. In order to allow models with multiple
    #' output layers, there are the following possibilities to select
    #' the indices of the output nodes in the individual output layers:
    #' \itemize{
    #'   \item A `vector` of indices: If the model has only one output layer,
    #'   the values correspond to the indices of the output nodes, e.g.
    #'   `c(1,3,4)` for the first, third and fourth output node. If there are
    #'   multiple output layers, the indices of the output nodes from the first
    #'   output layer are considered.
    #'   \item A `list` of `vectors` of indices: If the method is to be
    #'   applied to output nodes from different layers, a list can be passed
    #'   that specifies the desired indices of the output nodes for each
    #'   output layer. Unwanted output layers have the entry `NULL` instead of
    #'   a vector of indices, e.g. `list(NULL, c(1,3))` for the first and
    #'   third output node in the second output layer.
    #'   \item `NULL` (default): The method is applied to all output nodes in
    #'   the first output layer but is limited to the first ten as the
    #'   calculations become more computationally expensive for more output
    #'   nodes.
    #' }
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

      assert_logical(channels_first)
      self$channels_first <- channels_first

      assert_logical(times_input)
      self$times_input <- times_input

      assertLogical(verbose)
      self$verbose <- verbose

      assertChoice(dtype, c("float", "double"))
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      # Check output indices
      self$output_idx <- check_output_idx(output_idx, converter$output_dim)

      if (times_input & is.null(data)) {
        stop(
          "If you want to use the ConnectionWeights method with the ",
          "'times_input' argument, you must also specify 'data'!"
        )
      } else if (times_input) {
        self$data <- private$test_data(data)
      } else {
        if (!is.null(data)) {
          message(
            "If 'times_input' = FALSE, then the method ",
            "'ConnectionWeights' ",
            "is a global method and independent of the data. ",
            "Therefore, the argument 'data' will be ignored."
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
    },

    #' @description
    #' This method visualizes the result of the *Connection Weights*
    #' method and enables a visual in-depth investigation with the help
    #' of the S4 classes [`innsight_ggplot2`] and [`innsight_plotly`].\cr
    #' If the local *Connection Weights* method was applied, you can use the
    #' argument `data_idx` to select the data points in the given
    #' data for the plot. In addition, the individual output nodes for the plot
    #' can be selected with the argument `output_idx`. The different results
    #' for the selected data points and outputs are visualized using the
    #' ggplot2-based S4 class `innsight_ggplot2`. You can also use the
    #' `as_plotly` argument to generate an interactive plot with
    #' `innsight_plotly` based on the plot function [plotly::plot_ly]. For
    #' more information and the whole bunch of possibilities,
    #' see [`innsight_ggplot2`] and [`innsight_plotly`].\cr
    #' \cr
    #' **Note:**
    #' 1. For the interactive plotly-based plots, the suggested package
    #' `plotly` is required.
    #' 2. The ggplot2-based plots for models with multiple input layers are
    #' a bit more complex, therefore the suggested packages `'grid'`,
    #' `'gridExtra'` and `'gtable'` must be installed in your R session.
    #'
    #' @param data_idx An integer vector containing the numbers of the data
    #' points whose result is to be plotted, e.g. `c(1,3)` for the first
    #' and third data point in the given data. Default: `1`. This argument
    #' is only relevant for the local *Connection Weights* method and
    #' otherwise ignored.
    #' @param output_idx The indices of the output nodes for which the results
    #' is to be plotted. This can be either a `vector` of indices or a `list`
    #' of vectors of indices but must be a subset of the indices for which the
    #' results were calculated, i.e. a subset of `output_idx` from the
    #' initialization `new()` (see argument `output_idx` in method `new()` of
    #' this R6 class for details). By default (`NULL`), the smallest index
    #' of all calculated output nodes and output layers is used.
    #'
    #' @return
    #' Returns either an [`innsight_ggplot2`] (`as_plotly = FALSE`) or an
    #' [`innsight_plotly`] (`as_plotly = TRUE`) object with the plotted
    #' individual results.
    plot = function(data_idx = 1,
                    aggr_channels = "sum",
                    output_idx = NULL,
                    preprocess_FUN = identity,
                    as_plotly = FALSE) {
      if (!self$times_input) {
        if (!identical(data_idx, 1)) {
          message(paste0(
            "Without the 'times_input' argument, the method ",
            "'ConnectionWeights'",
            " is a global method, therefore no individual data instances ",
            "can be plotted. Your argument 'data_idx': c(",
            paste(data_idx, collapse = ", "), ")\n",
            "The argument 'data_idx' will be ignored in the following!"
          ))
        }
        data_idx <- 1
        self$data <- list(array(0, dim = c(1, 1)))
        no_data <- TRUE
      } else {
        no_data <- FALSE
      }

      private$plot(
        data_idx, output_idx, aggr_channels,
        as_plotly, "Relative Importance", no_data
      )
    },

    #' @description
    #' This method visualizes the results of the local *Connection Weights*
    #' method summarized as
    #' boxplots and enables a visual in-depth investigation of the global
    #' behavior with the help of the S4 classes [`innsight_ggplot2`] and
    #' [`innsight_plotly`].\cr
    #' You can use the argument `output_idx` to select the individual output
    #' nodes for the plot. For tabular and 1D data, boxplots are created in
    #' which a reference value can be selected from the data using the
    #' `ref_data_idx` argument. For images, only the pixel-wise median is
    #' visualized due to the complexity. The plot is generated using the
    #' ggplot2-based S4 class `innsight_ggplot2`. You can also use the
    #' `as_plotly` argument to generate an interactive plot with
    #' `innsight_plotly` based on the plot function [plotly::plot_ly]. For
    #' more information and the whole bunch of possibilities, see
    #' [`innsight_ggplot2`] and [`innsight_plotly`].\cr \cr
    #' **Note:**
    #' 1. This method can only be used for the local *Connection Weights*
    #' method, i.e. if `times_input` is `TRUE` and `data` is provided.
    #' 2. For the interactive plotly-based plots, the suggested package
    #' `plotly` is required.
    #' 3. The ggplot2-based plots for models with multiple input layers are
    #' a bit more complex, therefore the suggested packages `'grid'`,
    #' `'gridExtra'` and `'gtable'` must be installed in your R session.
    #'
    #' @param output_idx The indices of the output nodes for which the
    #' results is to be plotted. This can be either a `vector` of indices or
    #' a `list` of vectors of indices but must be a subset of the indices for
    #' which the results were calculated, i.e. a subset of `output_idx` from
    #' the initialization `new()` (see argument `output_idx` in method `new()`
    #' of this R6 class for details). By default (`NULL`), the smallest index
    #' of all calculated output nodes and output layers is used.
    #' @param data_idx By default ("all"), all available data points are used
    #' to calculate the boxplot information. However, this parameter can be
    #' used to select a subset of them by passing the indices. E.g. with
    #' `c(1:10, 25, 26)` only the first 10 data points and
    #' the 25th and 26th are used to calculate the boxplots.
    #'
    #' @return
    #' Returns either an [`innsight_ggplot2`] (`as_plotly = FALSE`) or an
    #' [`innsight_plotly`] (`as_plotly = TRUE`) object with the plotted
    #' summarized results.
    boxplot = function(output_idx = NULL,
                       data_idx = "all",
                       ref_data_idx = NULL,
                       aggr_channels = "norm",
                       preprocess_FUN = abs,
                       as_plotly = FALSE,
                       individual_data_idx = NULL,
                       individual_max = 20) {
      if (!self$times_input) {
        stop("\n[innsight] ERROR in boxplot for 'ConnectionWeights':\n",
          "Only if the result of the Connection-Weights method is ",
          "multiplied by the data ('times_input' = TRUE), it is a local ",
          "method and only then boxplots can be generated over multiple ",
          "instances. Thus, the argument 'data' must be specified and ",
          "'times_input = TRUE' when applying the 'ConnectionWeights$new' ",
          "method.",
          call. = FALSE
        )
      }

      private$boxplot(
        output_idx, data_idx, ref_data_idx, aggr_channels,
        preprocess_FUN, as_plotly, individual_data_idx,
        individual_max, "Relative Importance"
      )
    }
  )
)


#' @importFrom graphics boxplot
#' @exportS3Method
boxplot.ConnectionWeights <- function(x, ...) {
  x$boxplot(...)
}
