#'
#' @title Superclass for Gradient-based Interpretation Methods
#' @description Superclass for gradient-based interpretation methods. This
#' class inherits from [InterpretingMethod]. It summarizes all implemented
#' gradient-based methods and provides a private function to calculate the
#' gradients w.r.t. to the input for given data. Implemented are:
#'
#' - Normal Gradients and Gradient x Input ([Gradient])
#' - SmoothGrad and SmoothGrad x Input ([SmoothGrad])
#'
GradientBased <- R6::R6Class(
  classname = "GradientBased",
  inherit = InterpretingMethod,
  public = list(

    #' @field times_input Multiplies the gradients with the input features.
    #' This method is called 'Gradient x Input'.
    times_input = NULL,

    #' @description
    #' Create a new instance of this class.
    #'
    #' @param converter The converter of class [Converter] with the stored and
    #' torch-converted model.
    #' @param data The given data in an array-like format to be interpreted
    #' with the selected gradient-based method.
    #' @param channels_first The format of the given date, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data (either `'float'` or `'double'`).
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not. In some cases, the last
    #' activation leads to a saturation problem.
    #' @param times_input Multiplies the gradients with the input features. This
    #' method is called 'Gradient x Input'.
    #' @param output_idx This vector determines for which outputs the method
    #' will be applied. By default (`NULL`), all outputs (but limited to the
    #' first 10) are considered.
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = TRUE,
                          dtype = "float"
                          ) {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, dtype)

      assert_logical(times_input)
      self$times_input <- times_input
    },

    #'
    #' @description
    #' This method visualizes the result of the selected method in a
    #' [ggplot2::ggplot]. You can use the argument `data_idx` to select
    #' the data points in the given data for the plot. In addition, the
    #' individual classes for the plot can be selected with the argument
    #' `output_idx`. The different results for the selected data points and
    #' classes are visualized using the method [ggplot2::facet_grid].
    #'
    #' @param data_idx An integer vector containing the numbers of the data
    #' points whose result is to be plotted, e.g. `c(1,3)` for the first
    #' and third data point in the given data. Default: `c(1)`.
    #' @param output_idx An integer vector containing the numbers of the classes
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
    plot = function(data_idx = 1,
                    output_idx = c(),
                    aggr_channels = sum,
                    as_plotly = FALSE) {

      private$plot(data_idx, output_idx, aggr_channels,
                   as_plotly, "Gradient")
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
    #' @param data_idx By default ("all"), all available data is used to
    #' calculate the boxplot information. However, this parameter can be used
    #' to select a subset of them by passing the indices. E.g. with
    #' `data_idx = c(1:10, 25, 26)` only the first `10` data points and
    #' the 25th and 26th are used to calculate the boxplots.
    #' @param output_idx An integer vector containing the numbers of the classes
    #' whose result is to be plotted, e.g. `c(1,4)` for the first and fourth
    #' class. Default: `c(1)`.
    #' @param ref_data_idx This integer number determines the index for the
    #' reference data point. In addition to the boxplots, it is displayed in
    #' red color and is used to compare an individual result with the summary
    #' statistics provided by the boxplot. With the default value (`NULL`)
    #' no individual data point is plotted. This index can be chosen with
    #' respect to all available data, even if only a subset is selected with
    #' argument `data_idx`.\cr
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
    #' be used. Too high a number can significantly increase the runtime.
    #'
    #'
    boxplot = function(output_idx = c(),
                       data_idx = "all",
                       ref_data_idx = NULL,
                       aggr_channels = sum,
                       preprocess_FUN = abs,
                       as_plotly = FALSE,
                       individual_data_idx = NULL,
                       individual_max = 20) {
      private$boxplot(output_idx, data_idx, ref_data_idx, aggr_channels,
                      preprocess_FUN, as_plotly, individual_data_idx,
                      individual_max, "Gradient")
    }
  ),
  private = list(
    calculate_gradients = function(input) {
      input$requires_grad <- TRUE

      out <- self$converter$model(input,
                                  channels_first = self$channels_first,
                                  save_input = FALSE,
                                  save_preactivation = FALSE,
                                  save_output = FALSE,
                                  save_last_layer = TRUE)


      if (self$ignore_last_act) {
        output <- rev(self$converter$model$modules_list)[[1]]$preactivation
      } else {
        output <- out
      }

      # Implemented is only the case where the output is one-dimensional
      assertTRUE(length(dim(output)) == 2)
      out_sum <- sum(output, dim = 1)

      # Define Progressbar
      pb <- txtProgressBar(min = 0, max = length(self$output_idx), style = 3)

      res <- lapply(seq_len(length(self$output_idx)), function(i) {
        setTxtProgressBar(pb, i)
        autograd_grad(out_sum[self$output_idx[i]], input, retain_graph = TRUE)[[1]]
      })

      close(pb)

      torch_stack(res, dim = -1)
    }
  )
)


#'
#' @importFrom graphics boxplot
#' @exportS3Method
#'
boxplot.GradientBased <- function(x, ...) {
  x$boxplot(...)
}


#' @title Gradient Method
#' @name Gradient
#'
#' @description
#' This method computes the gradients (also known as Vanilla Gradients) of
#' the outputs with respect to the input variables, i.e. for all input
#' variable \eqn{i} and output class \eqn{j}
#' \deqn{d f(x)_j / d x_i.}
#' If the argument `times_input` is `TRUE`, the gradients are multiplied by
#' the respective input value (Gradient x Input), i.e.
#' \deqn{x_i * d f(x)_j / d x_i.}
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
#'
#' # Create Converter with input and output names
#' converter <- Converter$new(model,
#'   input_dim = c(5),
#'   input_names = list(c("Car", "Cat", "Dog", "Plane", "Horse")),
#'   output_names = list(c("Buy it!", "Don't buy it!"))
#' )
#'
#' # Calculate the Gradients
#' grad <- Gradient$new(converter, data)
#'
#' # Print the result as a data.frame
#' grad$get_result("data.frame")
#'
#' # Plot the result for both classes
#' plot(grad, output_idx = 1:2)
#'
#' # Plot the boxplot of all datapoints
#' boxplot(grad, output_idx = 1:2)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' library(neuralnet)
#' data(iris)
#'
#' # Train a neural network
#' nn <- neuralnet(Species ~ ., iris,
#'   linear.output = FALSE,
#'   hidden = c(10, 5),
#'   act.fct = "logistic",
#'   rep = 1
#' )
#'
#' # Convert the trained model
#' converter <- Converter$new(nn)
#'
#' # Calculate the gradients
#' gradient <- Gradient$new(converter, iris[, -5], times_input = TRUE)
#'
#' # Plot the result for the first and 60th data point and all classes
#' plot(gradient, data_idx = c(1, 60), output_idx = 1:3)
#'
#' # Calculate Gradients x Input and do not ignore the last activation
#' gradient <- Gradient$new(converter, iris[, -5], ignore_last_act = FALSE)
#'
#' # Plot the result again
#' plot(gradient, data_idx = c(1, 60), output_idx = 1:3)
#'
#' # ------------------------- Example 3: Keras -------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   data <- array(rnorm(64 * 60 * 3), dim = c(64, 60, 3))
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
#'   # Apply the Gradient method
#'   gradient <- Gradient$new(converter, data, channels_first = FALSE)
#'
#'   # Plot the result for the first datapoint and all classes
#'   plot(gradient, output_idx = 1:3)
#'
#'   # Plot the result as boxplots for first two classes
#'   boxplot(gradient, output_idx = 1:2)
#'
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'
#'   # Result as boxplots
#'   boxplot(gradient, as_plotly = TRUE)
#'
#'   # Result of the second data point
#'   plot(gradient, data_idx = 2, as_plotly = TRUE)
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
#' # create new instance of 'Gradient'
#' gradient <- Gradient$new(converter, iris[, -5])
#'
#' library(plotly)
#'
#' # Get the ggplot and add your changes
#' p <- plot(gradient, output_idx = 1, data_idx = 1:2) +
#'   theme_bw() +
#'   scale_fill_gradient2(low = "green", mid = "black", high = "blue")
#'
#' # Now apply the method plotly::ggplotly with argument tooltip = "text"
#' plotly::ggplotly(p, tooltip = "text")
#'
#' @export
#'

Gradient <- R6::R6Class(
  classname = "Gradient",
  inherit = GradientBased,
  public = list(

    #' @description
    #' Create a new instance of this class.
    #'
    #' @param converter The converter of class [Converter] with the stored
    #' and torch-converted model.
    #' @param data The given data in an array-like format to be interpreted
    #' with the this method.
    #' @param channels_first The format of the given data, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data (either `'float'` or `'double'`).
    #' Default: `'float'`.
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not. In some cases, the
    #' last activation leads to a saturation problem. Default: `TRUE`.
    #' @param times_input Multiplies the gradients with the input features.
    #' This method is called 'Gradient x Input'. Default: `TRUE`.
    #' @param output_idx This vector determines for which outputs the method
    #' will be applied. By default (`NULL`), all outputs (but limited to the
    #' first 10) are considered.
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = TRUE,
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, times_input, dtype)

      self$result <- private$run()
      self$converter$model$reset()
    }
  ),
  private = list(
    run = function() {
      message("Backwardpass 'Gradient':")
      gradients <- private$calculate_gradients(self$data)

      if (self$times_input) {
        gradients <- gradients * self$data$unsqueeze(-1)
      }

      gradients
    }
  )
)


#' @title SmoothGrad Method
#' @name SmoothGrad
#'
#' @description
#' SmoothGrad was introduced by D. Smilkov et al. (2017) and is an extension to
#' the classical [Gradient] method. It takes the mean of the gradients for
#' \code{n} perturbations of each data point, i.e. with
#' \eqn{\epsilon ~ N(0,\sigma)}
#' \deqn{1/n \sum_n d f(x+ \epsilon)_j / d x_j.}
#'
#' @references
#' D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @examplesIf torch::torch_is_installed()
#' # ------------------------- Example 1: Torch -------------------------------
#' library(torch)
#'
#' # Create nn_sequential model and data
#' model <- nn_sequential(
#'   nn_linear(5, 10),
#'   nn_relu(),
#'   nn_linear(10, 2),
#'   nn_sigmoid()
#' )
#' data <- torch_randn(25, 5)
#'
#' # Create Converter
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # Calculate the smoothed Gradients
#' smoothgrad <- SmoothGrad$new(converter, data)
#'
#' # Print the result as a data.frame
#' smoothgrad$get_result("data.frame")
#'
#' # Plot the result for both classes
#' plot(smoothgrad, output_idx = 1:2)
#'
#' # Plot the boxplot of all datapoints
#' boxplot(smoothgrad, output_idx = 1:2)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' library(neuralnet)
#' data(iris)
#'
#' # Train a neural network
#' nn <- neuralnet(Species ~ ., iris,
#'   linear.output = FALSE,
#'   hidden = c(10, 5),
#'   act.fct = "logistic",
#'   rep = 1
#' )
#'
#' # Convert the trained model
#' converter <- Converter$new(nn)
#'
#' # Calculate the smoothed gradients
#' smoothgrad <- SmoothGrad$new(converter, iris[, -5], times_input = FALSE)
#'
#' # Plot the result for the first and 60th data point and all classes
#' plot(smoothgrad, data_idx = c(1, 60), output_idx = 1:3)
#'
#' # Calculate SmoothGrad x Input and do not ignore the last activation
#' smoothgrad <- SmoothGrad$new(converter, iris[, -5], ignore_last_act = FALSE)
#'
#' # Plot the result again
#' plot(smoothgrad, data_idx = c(1, 60), output_idx = 1:3)
#'
#' # ------------------------- Example 3: Keras -------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   data <- array(rnorm(64 * 60 * 3), dim = c(64, 60, 3))
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
#'   # Apply the SmoothGrad method
#'   smoothgrad <- SmoothGrad$new(converter, data, channels_first = FALSE)
#'
#'   # Plot the result for the first datapoint and all classes
#'   plot(smoothgrad, output_idx = 1:3)
#'
#'   # Plot the result as boxplots for first two classes
#'   boxplot(smoothgrad, output_idx = 1:2)
#'
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'
#'   # Result as boxplots
#'   boxplot(smoothgrad, as_plotly = TRUE)
#'
#'   # Result of the second data point
#'   plot(smoothgrad, data_idx = 2, as_plotly = TRUE)
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
#' # create new instance of 'SmoothGrad'
#' smoothgrad <- SmoothGrad$new(converter, iris[, -5])
#'
#' library(plotly)
#'
#' # Get the ggplot and add your changes
#' p <- plot(smoothgrad, output_idx = 1, data_idx = 1:2) +
#'   theme_bw() +
#'   scale_fill_gradient2(low = "green", mid = "black", high = "blue")
#'
#' # Now apply the method plotly::ggplotly with argument tooltip = "text"
#' plotly::ggplotly(p, tooltip = "text")
#'
#' @export
#'
SmoothGrad <- R6Class(
  classname = "SmoothGrad",
  inherit = GradientBased,
  public = list(

    #' @field n Number of perturbations of the input data (default: \eqn{50}).
    #' @field noise_level The standard deviation of the gaussian
    #' perturbation, i.e. \eqn{\sigma = (max(x) - min(x)) *} `noise_level`.
    #'
    n = NULL,
    noise_level = NULL,

    #' @description
    #' Create a new instance of this class.
    #'
    #' @param converter The converter of class [Converter] with the stored and
    #' torch-converted model.
    #' @param data The given data in an array-like format to be interpreted
    #' with the this method.
    #' @param channels_first The format of the given data, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data (either `'float'` or `'double'`).
    #' Default: `'float'`.
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not. In some cases, the last
    #' activation leads to a saturation problem. Default: `TRUE`.
    #' @param times_input Multiplies the smoothed gradients with the input
    #' features. This method is called 'SmoothGrad x Input'.
    #' @param n Number of perturbations of the input data (default: \eqn{50}).
    #' @param noise_level Determines the standard deviation of the gaussian
    #' perturbation, i.e. \eqn{\sigma = (max(x) - min(x)) *} `noise_level`.
    #' @param output_idx This vector determines for which outputs the method
    #' will be applied. By default (`NULL`), all outputs (but limited to the
    #' first 10) are considered.
    #'
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = TRUE,
                          n = 50,
                          noise_level = 0.1,
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, times_input, dtype)

      assertInt(n, lower = 1)
      assertNumber(noise_level, lower = 0)
      self$n <- n
      self$noise_level <- noise_level

      self$result <- private$run()
      self$converter$model$reset()
    }
  ),
  private = list(
    run = function() {
      data <-
        torch_repeat_interleave(
          self$data,
          repeats = torch_tensor(self$n, dtype = torch_long()),
          dim = 1
        )

      noise_scale <- self$noise_level * (max(data) - min(data))

      noise <- torch_randn_like(data) * noise_scale

      message("Backwardpass 'SmoothGrad':")
      gradients <- private$calculate_gradients(data + noise)

      smoothgrads <-
        torch_stack(lapply(gradients$chunk(dim(self$data)[1]),
          FUN = torch_mean,
          dim = 1
        ),
        dim = 1
        )

      if (self$times_input) {
        smoothgrads <- smoothgrads * self$data$unsqueeze(-1)
      }

      smoothgrads
    }
  )
)
