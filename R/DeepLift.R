
#' @title Deep Learning Important FeaTures (DeepLift) Method
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
#' @examplesIf torch::torch_is_installed()
#' #----------------------- Example 1: Torch ----------------------------------
#' library(torch)
#'
#' # Create nn_sequential model and data
#' model <- nn_sequential(
#'   nn_linear(5, 12),
#'   nn_relu(),
#'   nn_linear(12, 2),
#'   nn_softmax(dim = 2)
#' )
#' data <- torch_randn(25, 5)
#' ref <- torch_randn(1, 5)
#'
#' # Create Converter
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # Apply method DeepLift
#' deeplift <- DeepLift$new(converter, data, x_ref = ref)
#'
#' # Print the result as a torch tensor for first two data points
#' deeplift$get_result("torch.tensor")[1:2]
#'
#' # Plot the result for both classes
#' plot(deeplift, output_idx = 1:2)
#'
#' # Plot the boxplot of all datapoints
#' boxplot(deeplift, output_idx = 1:2)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' library(neuralnet)
#' data(iris)
#'
#' # Train a neural network
#' nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(3, 2), act.fct = "tanh", rep = 1
#' )
#'
#' # Convert the model
#' converter <- Converter$new(nn)
#'
#' # Apply DeepLift with rescale-rule and a reference input of the feature
#' # means
#' x_ref <- matrix(colMeans(iris[, c(3, 4)]), nrow = 1)
#' deeplift_rescale <- DeepLift$new(converter, iris[, c(3, 4)], x_ref = x_ref)
#'
#' # Get the result as a dataframe and show first 5 rows
#' deeplift_rescale$get_result(type = "data.frame")[1:5, ]
#'
#' # Plot the result for the first datapoint in the data
#' plot(deeplift_rescale, data_idx = 1)
#'
#' # Plot the result as boxplots
#' boxplot(deeplift_rescale)
#'
#' # ------------------------- Example 3: Keras -------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))
#'
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_2d(
#'       input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
#'       activation = "softplus", padding = "valid"
#'     ) %>%
#'     layer_conv_2d(
#'       kernel_size = 8, filters = 4, activation = "tanh",
#'       padding = "same"
#'     ) %>%
#'     layer_conv_2d(
#'       kernel_size = 4, filters = 2, activation = "relu",
#'       padding = "valid"
#'     ) %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 2, activation = "softmax")
#'
#'   # Convert the model
#'   converter <- Converter$new(model)
#'
#'   # Apply the DeepLift method with reveal-cancel rule
#'   deeplift_revcancel <- DeepLift$new(converter, data,
#'     channels_first = FALSE,
#'     rule_name = "reveal_cancel"
#'   )
#'
#'   # Plot the result for the first image and both classes
#'   plot(deeplift_revcancel, output_idx = 1:2)
#'
#'   # Plot the result as boxplots for first class
#'   boxplot(deeplift_revcancel, output_idx = 1)
#'
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'   boxplot(deeplift_revcancel, as_plotly = TRUE)
#' }
#'
#' # ------------------------- Advanced: Plotly -------------------------------
#' # If you want to create an interactive plot of your results with custom
#' # changes, you can take use of the method plotly::ggplotly
#' library(ggplot2)
#' library(neuralnet)
#' library(plotly)
#' data(iris)
#'
#' nn <- neuralnet(Species ~ .,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
#' )
#' # create an converter for this model
#' converter <- Converter$new(nn)
#'
#' # create new instance of 'DeepLift'
#' deeplift <- DeepLift$new(converter, iris[, -5])
#'
#' # Get the ggplot and add your changes
#' p <- plot(deeplift, output_idx = 1, data_idx = 1:2) +
#'   theme_bw() +
#'   scale_fill_gradient2(low = "green", mid = "black", high = "blue")
#'
#' # Now apply the method plotly::ggplotly with argument tooltip = "text"
#' plotly::ggplotly(p, tooltip = "text")
#'
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#' @export
#'

DeepLift <- R6Class(
  classname = "DeepLift",
  inherit = InterpretingMethod,
  public = list(

    #' @field x_ref The reference input of size *(1, dim_in)* for the
    #' interpretation.
    #' @field rule_name Name of the applied rule to calculate the contributions
    #' for the non-linear part of a neural network layer. Either
    #' \code{"rescale"} or \code{"reveal_cancel"}.
    #'
    rule_name = NULL,
    x_ref = NULL,


    #' @description
    #' Create a new instance of the DeepLift method.
    #'
    #' @param converter An instance of the R6 class \code{\link{Converter}}.
    #' @param data The data for which the contribution scores are to be
    #' calculated. It has to be an array or array-like format of size
    #' *(batch_size, dim_in)*.
    #' @param channels_first The format of the given date, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param ignore_last_act Set this boolean value to include the last
    #' activation, or not (default: `TRUE`). In some cases, the last activation
    #' leads to a saturation problem.
    #' @param dtype The data type for the calculations. Use
    #' either `'float'` for [torch::torch_float] or `'double'` for
    #' [torch::torch_double].
    #' @param x_ref The reference input of size *(1, dim_in)* for the
    #' interpretation. With the default value \code{NULL} you use an input
    #' of zeros.
    #' @param rule_name Name of the applied rule to calculate the
    #' contributions. Use one of `'rescale'` and `'reveal_cancel'`.
    #' @param output_idx This vector determines for which outputs the method
    #' will be applied. By default (`NULL`), all outputs (but limited to the
    #' first 10) are considered.
    #'
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
        x_ref <- array(0, dim = c(1, dim(data)[-1]))
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


      self$result <- private$run()
    },


    #' @description
    #' This method visualizes the result of the selected method in a
    #' [ggplot2::ggplot]. You can use the argument `data_idx` to select
    #' the data points in the given data for the plot. In addition, the
    #' individual output nodes for the plot can be selected with the argument
    #' `output_idx`. The different results for the selected data points and
    #' outputs are visualized using the method [ggplot2::facet_grid].
    #' You can also use the `as_plotly` argument to generate an interactive
    #' plot based on the plot function [plotly::plot_ly].
    #'
    #' @param data_idx An integer vector containing the numbers of the data
    #' points whose result is to be plotted, e.g. `c(1,3)` for the first
    #' and third data point in the given data. Default: `c(1)`.
    #' @param output_idx An integer vector containing the numbers of the
    #' output indices whose result is to be plotted, e.g. `c(1,4)` for the
    #' first and fourth model output. But this vector must be included in the
    #' vector `output_idx` from the initialization, otherwise, no results were
    #' calculated for this output node and can not be plotted. By default
    #' (`NULL`), the smallest index of all calculated output nodes is used.
    #' @param aggr_channels Pass one of `'norm'`, `'sum'`, `'mean'` or a
    #' custom function to aggregate the channels, e.g. the maximum
    #' ([base::max]) or minimum ([base::min]) over the channels or only
    #' individual channels with `function(x) x[1]`. By default (`'sum'`),
    #' the sum of all channels is used.\cr
    #' **Note:** This argument is used only for 2D and 3D inputs.
    #' @param as_plotly This boolean value (default: `FALSE`) can be used to
    #' create an interactive plot based on the library `plotly`. This function
    #' takes use of [plotly::ggplotly], hence make sure that the suggested
    #' package `plotly` is installed in your R session.\cr
    #' **Advanced:** You can first
    #' output the results as a ggplot (`as_plotly = FALSE`) and then make
    #' custom changes to the plot, e.g. other theme or other fill color. Then
    #' you can manually call the function `ggplotly` to get an interactive
    #' plotly plot.
    #'
    #' @return
    #' Returns either a [ggplot2::ggplot] (`as_plotly = FALSE`) or a
    #' [plotly::plot_ly] (`as_plotly = TRUE`) with the plotted results.
    #'
    #'
    plot = function(data_idx = 1,
                    output_idx = NULL,
                    aggr_channels = 'sum',
                    as_plotly = FALSE) {

      private$plot(data_idx, output_idx, aggr_channels,
                   as_plotly, "Contribution")
    },

    #'
    #' @description
    #' This function visualizes the results of this method in a boxplot, where
    #' the type of visualization depends on the input dimension of the data.
    #' By default a [ggplot2::ggplot] is returned, but with the argument
    #' `as_plotly` an interactive [plotly::plot_ly] plot can be created,
    #' which however requires a successful installation of the package
    #' `plotly`.
    #'
    #'
    #' @param preprocess_FUN This function is applied to the method's result
    #' before calculating the boxplots. Since positive and negative values
    #' often cancel each other out, the absolute value (`abs`) is used by
    #' default. But you can also use the raw data (`identity`) to see the
    #' results' orientation, the squared data (`function(x) x^2`) to weight
    #' the outliers higher or any other function.
    #' @param data_idx By default ("all"), all available data is used to
    #' calculate the boxplot information. However, this parameter can be used
    #' to select a subset of them by passing the indices. E.g. with
    #' `data_idx = c(1:10, 25, 26)` only the first `10` data points and
    #' the 25th and 26th are used to calculate the boxplots.
    #' @param output_idx An integer vector containing the numbers of the
    #' output indices whose result is to be plotted, e.g. `c(1,4)` for the
    #' first and fourth model output. But this vector must be included in the
    #' vector `output_idx` from the initialization, otherwise, no results were
    #' calculated for this output node and can not be plotted. By default
    #' (`NULL`), the smallest index of all calculated output nodes is used.
    #' @param ref_data_idx This integer number determines the index for the
    #' reference data point. In addition to the boxplots, it is displayed in
    #' red color and is used to compare an individual result with the summary
    #' statistics provided by the boxplot. With the default value (`NULL`)
    #' no individual data point is plotted. This index can be chosen with
    #' respect to all available data, even if only a subset is selected with
    #' argument `data_idx`.\cr
    #' **Note:** Because of the complexity of 3D inputs, this argument is used
    #' only for 1D and 2D inputs and disregarded for 3D inputs.
    #' @param aggr_channels Pass one of `'norm'`, `'sum'`, `'mean'` or a
    #' custom function to aggregate the channels, e.g. the maximum
    #' ([base::max]) or minimum ([base::min]) over the channels or only
    #' individual channels with `function(x) x[1]`. By default (`'norm'`),
    #' the Euclidean norm of all channels is used.\cr
    #' **Note:** This argument is used only for 2D and 3D inputs.
    #' @param as_plotly This boolean value (default: `FALSE`) can be used to
    #' create an interactive plot based on the library `plotly` instead of
    #' `ggplot2`. Make sure that the suggested package `plotly` is installed
    #' in your R session.
    #' @param individual_data_idx Only relevant for a `plotly` plot with input
    #' dimension `1` or `2`! This integer vector of data indices determines
    #' the available data points in a dropdown menu, which are drawn in
    #' individually analogous to `ref_data_idx` only for more data points.
    #' With the default value `NULL` the first `individual_max` data points
    #' are used.\cr
    #' **Note:** If `ref_data_idx` is specified, this data point will be
    #' added to those from `individual_data_idx` in the dropdown menu.
    #' @param individual_max Only relevant for a `plotly` plot with input
    #' dimension `1` or `2`! This integer determines the maximum number of
    #' individual data points in the dropdown menu without counting
    #' `ref_data_idx`. This means that if `individual_data_idx` has more
    #' than `individual_max` indices, only the first `individual_max` will
    #' be used. A too high number can significantly increase the runtime.
    #'
    #' @return
    #' Returns either a [ggplot2::ggplot] (`as_plotly = FALSE`) or a
    #' [plotly::plot_ly] (`as_plotly = TRUE`) with the boxplots.
    #'
    boxplot = function(output_idx = NULL,
                       data_idx = "all",
                       ref_data_idx = NULL,
                       aggr_channels = 'norm',
                       preprocess_FUN = abs,
                       as_plotly = FALSE,
                       individual_data_idx = NULL,
                       individual_max = 20) {
      private$boxplot(output_idx, data_idx, ref_data_idx, aggr_channels,
                      preprocess_FUN, as_plotly, individual_data_idx,
                      individual_max, "Contribution")
    }
  ),
  private = list(
    run = function() {
      rev_layers <- rev(self$converter$model$modules_list)
      last_layer <- rev_layers[[1]]
      rev_layers <- rev_layers[-1]

      mul <- torch_diag_embed(torch_ones_like(last_layer$output))

      mul <- mul[,,self$output_idx, drop = FALSE]

      message("Backward pass 'DeepLift':")
      # Define Progressbar
      pb <- txtProgressBar(min = 0, max = length(rev_layers) + 1, style = 3)
      i <- 0

      if (self$ignore_last_act &&
        !("Flatten_Layer" %in% last_layer$".classes")) {
        mul <- last_layer$get_input_multiplier(mul,
          rule_name = "ignore_last_act"
        )
      } else {
        mul <- last_layer$get_input_multiplier(mul, self$rule_name)
      }
      last_layer$reset()

      i <- i + 1
      setTxtProgressBar(pb, i)

      # other layers
      for (layer in rev_layers) {
        if ("Flatten_Layer" %in% layer$".classes") {
          mul <- layer$reshape_to_input(mul)
        } else {
          mul <- layer$get_input_multiplier(mul, self$rule_name)
        }
        layer$reset()

        i <- i + 1
        setTxtProgressBar(pb, i)
      }
      if (!self$channels_first) {
        mul <- torch_movedim(mul, 2, length(dim(mul)) - 1)
      }
      x_diff <- (self$data - self$x_ref)$unsqueeze(-1)

      close(pb)

      mul * x_diff
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
