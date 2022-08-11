#' Connection Weights Method
#'
#' @description
#' This class implements the \emph{Connection Weights} method investigated by
#' Olden et al. (2004), which results in a feature relevance score for each input
#' variable. The basic idea is to multiply all path weights for each
#' possible connection between an input feature and the output node and then
#' calculate the sum over them. Besides, it is originally a global
#' interpretation method and independent of the input data. For a neural
#' network with \eqn{3} hidden layers with weight matrices \eqn{W_1},
#' \eqn{W_2} and \eqn{W_3}, this method results in a simple matrix multiplication
#' independent of the activation functions in between:
#' \deqn{W_1 * W_2 * W_3.}
#'
#' In this package, we extended this method to a local method inspired by the
#' method *Gradient x Input* (see [Gradient]). Hence, the local variant is simply
#' the pointwise product of the global *Connection Weights* method and the
#' input data. You can use this variant by setting the `times_input` argument
#' to `TRUE` and providing input data.
#'
#' @field times_input This logical value indicates whether the results from the
#' *Connection Weights* method were multiplied by the provided input data or not.
#' Thus, this value specifies whether the original global variant of the
#' method or the local one was applied. If the value is `TRUE`, then
#' data is provided in the field `data`.
#'
#' @examplesIf torch::torch_is_installed()
#' #----------------------- Example 1: Torch ----------------------------------
#' library(torch)
#'
#' # Create nn_sequential model
#' model <- nn_sequential(
#'   nn_linear(5, 12),
#'   nn_relu(),
#'   nn_linear(12, 1),
#'   nn_sigmoid()
#' )
#'
#' # Create Converter with input names
#' converter <- Converter$new(model,
#'   input_dim = c(5),
#'   input_names = list(c("Car", "Cat", "Dog", "Plane", "Horse"))
#' )
#'
#' # Apply method Connection Weights
#' cw <- ConnectionWeights$new(converter)
#'
#' # Print the result as a data.frame
#' cw$get_result("data.frame")
#'
#' # Plot the result
#' plot(cw)
#'
#' #----------------------- Example 2: Neuralnet ------------------------------
#' library(neuralnet)
#' data(iris)
#'
#' # Train a Neural Network
#' nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(3, 2), act.fct = "tanh", rep = 1
#' )
#'
#' # Convert the trained model
#' converter <- Converter$new(nn)
#'
#' # Apply the Connection Weights method
#' cw <- ConnectionWeights$new(converter)
#'
#' # Get the result as a torch tensor
#' cw$get_result(type = "torch.tensor")
#'
#' # Plot the result
#' plot(cw)
#'
#' #----------------------- Example 3: Keras ----------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   # Define a model
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_1d(
#'       input_shape = c(64, 3), kernel_size = 16, filters = 8,
#'       activation = "softplus"
#'     ) %>%
#'     layer_conv_1d(kernel_size = 16, filters = 4, activation = "tanh") %>%
#'     layer_conv_1d(kernel_size = 16, filters = 2, activation = "relu") %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 2, activation = "softmax")
#'
#'   # Convert the model
#'   converter <- Converter$new(model)
#'
#'   # Apply the Connection Weights method
#'   cw <- ConnectionWeights$new(converter)
#'
#'   # Get the result as data.frame
#'   cw$get_result(type = "data.frame")
#'
#'   # Plot the result for all classes
#'   plot(cw, output_idx = 1:2)
#' }
#'
#' # ------------------------- Advanced: Plotly -------------------------------
#' # If you want to create an interactive plot of your results with custom
#' # changes, you can take use of the method plotly::ggplotly
#' library(ggplot2)
#' library(plotly)
#' library(neuralnet)
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
#' # create new instance of 'LRP'
#' cw <- ConnectionWeights$new(converter)
#'
#' library(plotly)
#'
#' # Get the ggplot and add your changes
#' p <- plot(cw, output_idx = 1) +
#'   theme_bw() +
#'   scale_fill_gradient2(low = "green", mid = "black", high = "blue")
#'
#' # Now apply the method plotly::ggplotly with argument tooltip = "text"
#' plotly::ggplotly(p, tooltip = "text")
#'
#' @references
#' * J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
#'  quantifying variable importance in artificial neural networks using
#'  simulated data.} Ecological Modelling 178, p. 389–397
#'
#' @export
ConnectionWeights <- R6Class(
  classname = "ConnectionWeights",
  inherit = InterpretingMethod,
  public = list(
    times_input = NULL,

    #' @param converter The converter of class [Converter] with the stored and
    #' torch-converted model.
    #' @param data The data for which the contribution scores are to be
    #' calculated. It has to be an array or array-like format of size
    #' *(batch_size, dim_in)*. This argument is only relevant if
    #' `times_input` is `TRUE`, otherwise it will be ignored because it is a
    #' locale (i.e. explanation for each data point individually) method only
    #' in this case.
    #' @param output_idx This vector determines for which output indices the
    #' method will be applied. By default (`NULL`), all outputs (but limited
    #' to the first 10) are considered.
    #' @param channels_first The data format of the result, i.e. channels on
    #' last dimension (`FALSE`) or on the first dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param times_input Multiplies the results with the input features.
    #' This variant tuns the global Connection Weights method into a local one.
    #' Default: `FALSE`.
    #' @param dtype The data type for the calculations. Use
    #' either `'float'` for [torch::torch_float] or `'double'` for
    #' [torch::torch_double].
    #'
    initialize = function(converter,
                          data = NULL,
                          output_idx = NULL,
                          channels_first = TRUE,
                          times_input = FALSE,
                          dtype = "float") {

      assertClass(converter, "Converter")
      self$converter <- converter

      assert_logical(channels_first)
      self$channels_first <- channels_first

      assert_logical(times_input)
      self$times_input <- times_input

      assertChoice(dtype, c("float", "double"))
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      # Check output indices
      self$output_idx <- check_output_idx(output_idx, converter$output_dim)

      if (times_input & is.null(data)) {
        stop("If you want to use the ConnectionWeights method with the ",
             "'times_input' argument, you must also specify 'data'!")
      } else if (times_input) {
        self$data <- private$test_data(data)
      } else {
        if (!is.null(data)) {
          message("If 'times_input' = FALSE, then the method Connection-Weights ",
                  "is a global method and independent of the data. ",
                  "Therefore, the argument 'data' will be ignored.")
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

    #'
    #' @description
    #' This method visualizes the result of the *ConnectionWeights* method in a
    #' [ggplot2::ggplot]. You can use the argument `output_idx` to select
    #' individual output nodes for the plot. The different results for the
    #' selected outputs are visualized using the method [ggplot2::facet_grid].
    #' You can also use the `as_plotly` argument to generate an interactive
    #' plot based on the plot function [plotly::plot_ly].
    #'
    #' @param data_idx to do!
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
    #' @param preprocess_FUN This function is applied to the method's result
    #' before generating the plot. By default, the identity function
    #' (`identity`) is used.
    #'
    #' @return
    #' Returns either a [ggplot2::ggplot] (`as_plotly = FALSE`) or a
    #' [plotly::plot_ly] object (`as_plotly = TRUE`) with the plotted results.
    #'
    plot = function(data_idx = 1,
                    aggr_channels = 'sum',
                    output_idx = NULL,
                    preprocess_FUN = identity,
                    as_plotly = FALSE) {

      if (!self$times_input) {
        if (!identical(data_idx, 1)) {
          message(paste0(
            "Without the 'times_input' argument, the method 'Connection-Weights'",
            " is a global method, therefore no individual data instances ",
            "can be plotted. Your argument 'data_idx': c(",
            paste(data_idx, collapse = ", "), ")\n",
            "The argument 'data_idx' will be ignored in the following!"))
        }
        data_idx <- 1
        self$data <- list(array(0, dim = c(1,1)))
        no_data <- TRUE
      } else {
        no_data <- FALSE
      }

      private$plot(data_idx, output_idx, aggr_channels,
                   as_plotly, "Relative Importance", no_data)
    },


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
    #'
    boxplot = function(output_idx = NULL,
                       data_idx = "all",
                       ref_data_idx = NULL,
                       aggr_channels = 'norm',
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
             "method.", call. = FALSE)
      }

      private$boxplot(output_idx, data_idx, ref_data_idx, aggr_channels,
                      preprocess_FUN, as_plotly, individual_data_idx,
                      individual_max, "Relative Importance")
    }
  )
)


#' @importFrom graphics boxplot
#' @exportS3Method
boxplot.ConnectionWeights <- function(x, ...) {
  x$boxplot(...)
}
