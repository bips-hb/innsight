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
#' activation functions: *Rescale-Rule* (`'rescale'`) and
#' *RevealCancel-Rule* (`'reveal_cancel'`).
#'
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-ignore_last_act
#' @template param-x_ref
#' @template param-dtype
#' @template param-aggr_channels
#' @template param-as_plotly
#' @template param-ref_data_idx
#' @template param-preprocess_FUN
#' @template param-individual_data_idx
#' @template param-individual_max
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
    #' @param rule_name Name of the applied rule to calculate the
    #' contributions. Use either `'rescale'` or `'reveal_cancel'`.
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          rule_name = "rescale",
                          x_ref = NULL,
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, dtype)

      assertChoice(rule_name, c("rescale", "reveal_cancel"))
      self$rule_name <- rule_name

      if (is.null(x_ref)) {
          x_ref <- lapply(lapply(self$data, dim),
                          function(x) array(0, dim = c(1,x[-1])))
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
    },

    #' @description
    #' This method visualizes the result of individual data points of the
    #' selected method and enables a visual in-depth investigation with the help
    #' of the S4 classes [`innsight_ggplot2`] and [`innsight_plotly`].\cr
    #' You can use the argument `data_idx` to select the data points in the given
    #' data for the plot. In addition, the individual output nodes for the plot
    #' can be selected with the argument `output_idx`. The different results for
    #' the selected data points and outputs are visualized using the ggplot2-based
    #' S4 class `innsight_ggplot2`. You can also use the `as_plotly` argument to
    #' generate an interactive plot with `innsight_plotly` based on the
    #' plot function [plotly::plot_ly]. For more information and the whole bunch
    #' of possibilities, see [`innsight_ggplot2`] and [`innsight_plotly`].\cr
    #' \cr
    #' **Note:**
    #' 1. For the interactive plotly-based plots, the suggested package `plotly`
    #' is required.
    #' 2. The ggplot2-based plots for models with multiple input layers are a bit
    #' more complex, therefore the suggested packages `'grid'`, `'gridExtra'`
    #' and `'gtable'` must be installed in your R session.
    #'
    #' @param data_idx An integer vector containing the numbers of the data
    #' points whose result is to be plotted, e.g. `c(1,3)` for the first
    #' and third data point in the given data. Default: `1`.
    #' @param output_idx The indices of the output nodes for which the results is
    #' to be plotted. This can be either a `vector` of indices or a `list` of
    #' vectors of indices but must be a subset of the indices for which the
    #' results were calculated, i.e. a subset of `output_idx` from the
    #' initialization `new()` (see argument `output_idx` in method `new()` of this
    #' R6 class for details). By default (`NULL`), the smallest index of all
    #' calculated output nodes and output layers is used.
    #'
    #' @return
    #' Returns either an [`innsight_ggplot2`] (`as_plotly = FALSE`) or an
    #' [`innsight_plotly`] (`as_plotly = TRUE`) object with the plotted
    #' individual results.
    plot = function(data_idx = 1,
                    output_idx = NULL,
                    aggr_channels = 'sum',
                    as_plotly = FALSE) {

      private$plot(data_idx, output_idx, aggr_channels,
                   as_plotly, "Contribution")
    },

    #' @description
    #' This method visualizes the results of the selected method summarized as
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
    #' 1. For the interactive plotly-based plots, the suggested package `plotly`
    #' is required.
    #' 2. The ggplot2-based plots for models with multiple input layers are a bit
    #' more complex, therefore the suggested packages `'grid'`, `'gridExtra'`
    #' and `'gtable'` must be installed in your R session.
    #'
    #' @param output_idx The indices of the output nodes for which the results is
    #' to be plotted. This can be either a `vector` of indices or a `list` of
    #' vectors of indices but must be a subset of the indices for which the
    #' results were calculated, i.e. a subset of `output_idx` from the
    #' initialization `new()` (see argument `output_idx` in method `new()` of this
    #' R6 class for details). By default (`NULL`), the smallest index of all
    #' calculated output nodes and output layers is used.
    #' @param data_idx By default ("all"), all available data points are used to
    #' calculate the boxplot information. However, this parameter can be used
    #' to select a subset of them by passing the indices. E.g. with
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
                       aggr_channels = 'sum',
                       preprocess_FUN = abs,
                       as_plotly = FALSE,
                       individual_data_idx = NULL,
                       individual_max = 20) {
      private$boxplot(output_idx, data_idx, ref_data_idx, aggr_channels,
                      preprocess_FUN, as_plotly, individual_data_idx,
                      individual_max, "Contribution")
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
