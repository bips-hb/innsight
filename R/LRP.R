
#' @title Layer-wise Relevance Propagation (LRP) Method
#' @name LRP
#'
#' @description
#' This is an implementation of the \emph{Layer-wise Relevance Propagation
#' (LRP)} algorithm introduced by Bach et al. (2015). It's a local method for
#' interpreting a single element of the dataset and calculates the relevance
#' scores for each input feature. The basic idea of this method is to decompose
#' the prediction score of the model with respect to the input features, i.e.
#' \deqn{f(x) = \sum_i R(x_i).}
#' Because of the bias vector, this decomposition is generally an approximation.
#' There exist several propagation rules to determine the relevance scores.
#' In this package are implemented: simple rule ("simple"), epsilon rule
#' ("epsilon") and alpha-beta rule ("alpha_beta").
#'
#' @examplesIf torch::torch_is_installed()
#'  #----------------------- Example 1: Torch ----------------------------------
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
#'
#' # Create Converter
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # Apply method LRP with simple rule (default)
#' lrp <- LRP$new(converter, data)
#'
#' # Print the result as an array
#' lrp$get_result()
#'
#' # Plot the result for both classes
#' plot(lrp, classes = 1:2)
#'
#' # Plot the boxplot of all datapoints without preprocess function
#' boxplot(lrp, classes = 1:2, preprocess_FUN = identity)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' library(neuralnet)
#' data(iris)
#' nn <- neuralnet(Species ~ .,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
#' )
#' # create an converter for this model
#' converter <- Converter$new(nn)
#'
#' # create new instance of 'LRP'
#' lrp <- LRP$new(converter, iris[, -5], rule_name = "simple")
#'
#' # get the result as an array
#' lrp$get_result()
#'
#' # get the result as a torch tensor
#' lrp$get_result(type = "torch.tensor")
#'
#' # use the alpha-beta rule with alpha = 2
#' lrp <- LRP$new(converter, iris[, -5],
#'   rule_name = "alpha_beta",
#'   rule_param = 2
#' )
#'
#' # include the last activation into the calculation
#' lrp <- LRP$new(converter, iris[, -5],
#'   rule_name = "alpha_beta",
#'   rule_param = 2,
#'   ignore_last_act = FALSE
#' )
#'
#' # Plot the result for all classes
#' plot(lrp, class = 1:3)
#'
#' # Plot the Boxplot for the first class
#' boxplot(lrp)
#'
#' # You can also create an interactive plot with plotly.
#' # This is a suggested package, so make sure that it is installed
#' library(plotly)
#'
#' # Result as boxplots
#' boxplot(lrp, as_plotly = TRUE)
#'
#' # Result of the second data point
#' plot(lrp, datapoint = 2, as_plotly = TRUE)
#'
#' # ------------------------- Example 3: Keras -------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   data <- array(rnorm(10 * 60 * 3), dim = c(10, 60, 3))
#'
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_1d(
#'       input_shape = c(60, 3), kernel_size = 8, filters = 8,
#'       activation = "softplus", padding = "valid"
#'     ) %>%
#'     layer_conv_1d(
#'       kernel_size = 8, filters = 4, activation = "tanh",
#'       padding = "same"
#'     ) %>%
#'     layer_conv_1d(
#'       kernel_size = 4, filters = 2, activation = "relu",
#'       padding = "valid"
#'     ) %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 3, activation = "softmax")
#'
#'   # Convert the model
#'   converter <- Converter$new(model)
#'
#'   # Apply the LRP method with the epsilon rule and eps = 0.1
#'   lrp_eps <- LRP$new(converter, data,
#'     channels_first = FALSE,
#'     rule_name = "epsilon",
#'     rule_param = 0.1
#'   )
#'
#'   # Plot the result for the first datapoint and all classes
#'   plot(lrp_eps, classes = 1:3)
#'
#'   # Plot the result as boxplots for first two classes
#'   boxplot(lrp_eps, classes = 1:2)
#'
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'
#'   # Result as boxplots
#'   boxplot(lrp_eps, as_plotly = TRUE)
#'
#'   # Result of the second data point
#'   plot(lrp_eps, datapoint = 2, as_plotly = TRUE)
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
#' lrp <- LRP$new(converter, iris[, -5])
#'
#' library(plotly)
#'
#' # Get the ggplot and add your changes
#' p <- plot(lrp, classes = 1, datapoint = 1:2) +
#'   theme_bw() +
#'   scale_fill_gradient2(low = "green", mid = "black", high = "blue")
#'
#' # Now apply the method plotly::ggplotly with argument tooltip = "text"
#' plotly::ggplotly(p, tooltip = "text")
#'
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10,
#' p. 1-46
#'
#' @export

LRP <- R6Class(
  classname = "LRP",
  inherit = InterpretingMethod,
  public = list(
    #' @field rule_name The name of the rule, with which the relevance scores
    #' are calculated. Implemented are \code{"simple"}, \code{"epsilon"},
    #' \code{"alpha_beta"} (default: \code{"simple"}).
    #' @field rule_param The parameter of the selected rule.
    #'
    rule_name = NULL,
    rule_param = NULL,


    #' @description
    #' Create a new instance of the LRP-Method.
    #'
    #' @param converter An instance of the R6 class \code{\link{Converter}}.
    #' @param data The data for which the relevance scores are to be
    #' calculated. It has to be an array or array-like format of size
    #' (batch_size, dim_in).
    #' @param rule_name The name of the rule, with which the relevance scores
    #' are calculated. Implemented are \code{"simple"}, \code{"epsilon"},
    #' \code{"alpha_beta"} (default: \code{"simple"}).
    #' @param rule_param The parameter of the selected rule. Note: Only the
    #' rules \code{"epsilon"} and \code{"alpha_beta"} take use of the
    #' parameter. Use the default value \code{NULL} for the default parameters
    #' ("epsilon" : \eqn{0.01}, "alpha_beta" : \eqn{0.5}).
    #' @param channels_first Set the data format of the given data. Internally
    #' the format `channels_first` is used, therefore the format of the given
    #' data is required. Also use the default value `TRUE` if no convolutional
    #' layers are used.
    #' @param ignore_last_act Set this boolean value to include the last
    #' activation, or not (default: `TRUE`). In some cases, the last activation
    #' leads to a saturation problem.
    #' @param dtype The data type for the calculations. Use either `'float'` or
    #' `'double'`.
    #'
    #' @return A new instance of the R6 class `'LRP'`.
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          rule_name = "simple",
                          rule_param = NULL,
                          ignore_last_act = TRUE,
                          dtype = "float") {
      super$initialize(converter, data, channels_first, dtype, ignore_last_act)

      assertChoice(rule_name, c("simple", "epsilon", "alpha_beta"))
      self$rule_name <- rule_name

      assertNumber(rule_param, null.ok = TRUE)
      self$rule_param <- rule_param

      self$converter$model$forward(self$data,
        channels_first = self$channels_first
      )

      self$result <- private$run()
    },

    #'
    #' @description
    #' This method visualizes the result of the selected method in a
    #' [ggplot2::ggplot]. You can use the argument `datapoint` to select
    #' the data points in the given data for the plot. In addition, the
    #' individual classes for the plot can be selected with the argument
    #' `classes`. The different results for the selected data points and
    #' classes are visualized using the method [ggplot2::facet_grid].
    #'
    #' @param datapoint An integer vector containing the numbers of the data
    #' points whose result is to be plotted, e.g. `c(1,3)` for the first
    #' and third data point in the given data. Default: `c(1)`.
    #' @param classes An integer vector containing the numbers of the classes
    #' whose result is to be plotted, e.g. `c(1,4)` for the first and fourth
    #' class. Default: `c(1)`.
    #' @param aggr_channels Pass a function to aggregate the channels. The
    #' default function is [base::sum], but you can pass an arbitrary function.
    #' For example, the maximum `max` or minimum `min` over the channels or
    #' only individual channels with `function(x) x[1]`.
    #' @param as_plotly This boolean value (default: `FALSE`) can be used to
    #' create an interactive plot based on the library `plotly`. This function
    #' takes use of [plotly::ggplotly], hence make sure that the suggested
    #' package `plotly` is installed in your R session. Advanced: You can first
    #' output the results as a ggplot (`as_plotly = FALSE`) and then make
    #' custom changes to the plot, e.g. other theme or other fill color. Then
    #' you can manually call the function `ggplotly` to get an interactive
    #' plotly plot.
    #'
    #' @return
    #' Returns either a [ggplot2::ggplot] (`as_plotly = FALSE`) or a
    #' [plotly::plot_ly] (`as_plotly = TRUE`) with the plotted results.
    #'
    plot = function(datapoint = 1,
                    classes = 1,
                    aggr_channels = sum,
                    as_plotly = FALSE) {

      private$plot(datapoint, classes, aggr_channels,
                   as_plotly, "Relevance")
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
    #' @param preprocess_FUN This function is applied to the method's result
    #' before calculating the boxplots. Since positive and negative values
    #' often cancel each other out, the absolute value (`abs`) is used by
    #' default. But you can also use the raw data (`function(x) x`) to see the
    #' results' orientation, the squared data (`function(x) x^2`) to weight
    #' the outliers higher or any other function.
    #' @param boxplot_data By default ("all"), all available data is used to
    #' calculate the boxplot information. However, this parameter can be used
    #' to select a subset of them by passing the indices. E.g. with
    #' `boxplot_data = c(1:10, 25, 26)` only the first `10` data points and
    #' the 25th and 26th are used to calculate the boxplots.
    #' @param classes An integer vector containing the numbers of the classes
    #' whose result is to be plotted, e.g. `c(1,4)` for the first and fourth
    #' class. Default: `c(1)`.
    #' @param ref_datapoint This integer number determines the index for the
    #' reference data point. In addition to the boxplots, it is displayed in
    #' red color and is used to compare an individual result with the summary
    #' statistics provided by the boxplot. With the default value (`NULL`)
    #' no individual data point is plotted. This index can be chosen with
    #' respect to all available data, even if only a subset is selected with
    #' argument `boxplot_data`.\cr
    #' **Note:** Because of the complexity of 3D inputs, this argument is used
    #' only for 1D and 2D inputs and disregarded for 3D inputs.
    #' @param aggr_channels Pass a function to aggregate the channels. The
    #' default function is [base::mean], but you can pass an arbitrary
    #' function. For example, the maximum `max` or minimum `min` over the
    #' channels or only individual channels with `function(x) x[1]`.\cr
    #' **Note:** This function is used only for 2D and 3D inputs.
    #' @param as_plotly This boolean value (default: `FALSE`) can be used to
    #' create an interactive plot based on the library `plotly` instead of
    #' `ggplot2`. Make sure that the suggested package `plotly` is installed
    #' in your R session.
    #' @param individual_data Only relevant for a `plotly` plot with input
    #' dimension `1` or `2`! This integer vector of data indices determines
    #' the available data points in a dropdown menu, which are drawn in
    #' individually analogous to `ref_datapoint` only for more data points.
    #' With the default value `NULL` the first `individual_max` data points
    #' are used.\cr
    #' **Note:** If `ref_datapoint` is specified, this data point will be
    #' added to those from `individual_data` in the dropdown menu.
    #' @param individual_max Only relevant for a `plotly` plot with input
    #' dimension `1` or `2`! This integer determines the maximum number of
    #' individual data points in the dropdown menu without counting
    #' `ref_datapoint`. This means that if `individual_data` has more
    #' than `individual_max` indices, only the first `individual_max` will
    #' be used. Too high a number can significantly increase the runtime.
    #'
    #'
    boxplot = function(boxplot_data = "all",
                       classes = 1,
                       ref_datapoint = NULL,
                       aggr_channels = mean,
                       preprocess_FUN = abs,
                       as_plotly = FALSE,
                       individual_data = NULL,
                       individual_max = 20) {
      private$boxplot(preprocess_FUN, boxplot_data, classes, ref_datapoint,
                      aggr_channels, individual_data,
                      individual_max, as_plotly, "Relevance")
    }
  ),
  private = list(
    run = function() {
      rev_layers <- rev(self$converter$model$modules_list)
      last_layer <- rev_layers[[1]]

      if (self$ignore_last_act) {
        rel <- torch_diag_embed(last_layer$preactivation)
      } else {
        rel <- torch_diag_embed(last_layer$output)

        # For probabilistic output we need to subtract 0.5, such that
        # 0 means no relevance
        if (last_layer$activation_name %in%
          c("softmax", "sigmoid", "logistic")) {
          rel <- rel - 0.5
        }
      }

      # other layers
      for (layer in rev_layers) {
        if ("Flatten_Layer" %in% layer$".classes") {
          rel <- layer$reshape_to_input(rel)
        } else {
          rel <- layer$get_input_relevances(
            rel,
            self$rule_name,
            self$rule_param
          )
        }
      }
      if (!self$channels_first) {
        rel <- torch_movedim(rel, 2, length(dim(rel)) - 1)
      }


      rel
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
