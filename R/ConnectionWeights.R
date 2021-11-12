#' Connection Weights Method
#'
#' @description
#' This class implements the \emph{Connection Weight} method investigated by
#' Olden et al. (2004) which results in a feature relevance score for each input
#' variable. The basic idea is to multiply up all path weights for each
#' possible connection between an input feature and the output and then
#' calculate the sum over them. Besides, it is a global interpretation method
#' and independent of the input data. For a neural network with \eqn{3} hidden
#' layers with weight matrices \eqn{W_1}, \eqn{W_2} and \eqn{W_3} this method
#' results in a simple matrix multiplication
#' \deqn{W_1 * W_2 * W_3. }
#'
#'
#' @field converter The converter of class [Converter] with the stored and
#' torch-converted model.
#' @field channels_first The data format of the result, i.e. channels on
#' last dimension (`FALSE`) or on the first dimension (`TRUE`). If the
#' data has no channels, use the default value `TRUE`.
#' @field dtype The type of the data and parameters (either `'float'`
#' for [torch::torch_float] or `'double'` for [torch::torch_double]).
#' @field result The methods result as a torch tensor of size
#' (dim_in, dim_out).
#' @field output_idx This vector determines for which outputs the method
#' will be applied
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
  public = list(
    converter = NULL,
    channels_first = NULL,
    dtype = NULL,
    result = NULL,
    output_idx = NULL,

    #' @param converter The converter of class [Converter] with the stored and
    #' torch-converted model.
    #' @param output_idx This vector determines for which output indices the
    #' method will be applied. By default (`NULL`), all outputs (but limited
    #' to the first 10) are considered.
    #' @param channels_first The data format of the result, i.e. channels on
    #' last dimension (`FALSE`) or on the first dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data and parameters
    #' (either `'float'` or `'double'`).
    #'
    initialize = function(converter,
                          output_idx = NULL,
                          channels_first = TRUE,
                          dtype = "float") {
      assertClass(converter, "Converter")
      self$converter <- converter

      assertIntegerish(output_idx, null.ok = TRUE, lower = 1,
                       upper = converter$model_dict$output_dim)


      if (is.null(output_idx)) output_idx <-
        1:min(converter$model_dict$output_dim, 10)
      self$output_idx <- output_idx

      assert_logical(channels_first)
      self$channels_first <- channels_first

      assertChoice(dtype, c("float", "double"))
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      self$result <- private$run()
    },

    #'
    #' @description
    #' This function returns the result of the Connection Weights method either
    #' as an array (`array`), a torch tensor (`torch.tensor`) of size
    #' (dim_in, dim_out) or a data.frame (`data.frame`).
    #'
    #' @param type The data format of the result. Use one of `'array'`,
    #' `'torch.tensor'` or `'data.frame'` (default: `'array'`).
    #'
    #' @return The result of this method for the given data in the chosen
    #' format.
    #'
    get_result = function(type = "array") {
      assertChoice(type,
                   c("array", "data.frame", "torch.tensor", "torch_tensor"))

      result <- self$result
      if (type == "array") {
        result <- as.array(result)
      } else if (type == "data.frame") {
        result <- private$get_dataframe()
      }

      result
    },

    #'
    #' @description
    #' This method visualizes the result of the ConnectionWeight method in a
    #' [ggplot2::ggplot]. You can use the argument `classes` to select
    #' the classes for the plot. By default a [ggplot2::ggplot] is returned,
    #' but with the argument `as_plotly` an interactive [plotly::plot_ly] plot
    #' can be created, which however requires a successful installation of
    #' the package `plotly`.
    #'
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
    #' @param preprocess_FUN This function is applied to the method's result
    #' before generating the plot. By default, the identity function
    #' (`identity`) is used.
    #'
    #' @return
    #' Returns either a [ggplot2::ggplot] (`as_plotly = FALSE`) or a
    #' [plotly::plot_ly] object (`as_plotly = TRUE`) with the plotted results.
    #'
    plot = function(output_idx = c(),
                    aggr_channels = sum,
                    preprocess_FUN = identity,
                    as_plotly = FALSE) {

      assertSubset(output_idx, self$output_idx)
      assert(
        checkFunction(aggr_channels),
        checkChoice(aggr_channels, c("norm", "sum", "mean"))
      )
      assertFunction(preprocess_FUN)
      assertLogical(as_plotly)

      if (length(output_idx) == 0) {
        classes <- self$output_idx[1]
        classes_idx <- 1
      } else {
        classes <- output_idx
        classes_idx <- match(classes, self$output_idx)
      }

      if (!is.function(aggr_channels)) {
        if (aggr_channels == "norm") {
          aggr_channels <- function(x) sum(x^2)^0.5
        } else if (aggr_channels == "sum") {
          aggr_channels <- sum
        } else if (aggr_channels == "mean") {
          aggr_channels <- mean
        }
      }

      l <- length(dim(self$result))
      output_names <- unlist(self$converter$model_dict$output_names)[classes]
      input_names <- self$converter$model_dict$input_names
      result <- preprocess_FUN(self$result)

      # 1D Input
      if (l == 2) {
        result <- result[,classes_idx, drop = FALSE]$unsqueeze(1)
        p <- plot_1d_input(result, "Relative Importance", "data_1",
                           input_names, output_names, TRUE, TRUE)
        dynamicTicks <- FALSE
      }
      # 2D Input
      else if (l == 3) {
        result <- as_array(result[, , classes_idx, drop = FALSE]$unsqueeze(1))
        if (self$channels_first) {
          dims <- c(1, 3, 4)
          d <- 2
        } else {
          dims <- c(1, 2, 4)
          d <- 3
        }

        # Summarize the channels by function 'aggr_channels'
        result <- torch_tensor(apply(result, dims, aggr_channels))$unsqueeze(d)
        input_names[[1]] <- c("aggr")
        p <- plot_2d_input(result, "Relative Importance", "data_1", input_names,
                           output_names, self$channels_first, TRUE)
        dynamicTicks <- TRUE
      }
      # 3D Input
      else if (l == 4) {
        result <- as_array(result[, , , classes_idx, drop = FALSE]$unsqueeze(1))
        if (self$channels_first) {
          dims <- c(1, 3, 4, 5)
          d <- 2
        } else {
          dims <- c(1, 2, 3, 5)
          d <- 4
        }

        # Summarize the channels by function 'aggr_channels'
        result <- torch_tensor(apply(result, dims, aggr_channels))$unsqueeze(d)
        input_names[[1]] <- c("aggr")
        p <- plot_3d_input(result, "Relative Importance", "data_1", input_names,
                           output_names, self$channels_first, TRUE)
        dynamicTicks <- TRUE
      }

      if (as_plotly) {
        if (!requireNamespace("plotly", quietly = FALSE)) {
          stop("Please install the 'plotly' package if you want to create an
         interactive plot.")
        }
        p <- plotly::ggplotly(p, tooltip = "text", dynamicTicks = dynamicTicks)
        p <- plotly::layout(p,
                            xaxis = list(rangemode = "tozero"),
                            yaxis = list(rangemode = "tozero"))
      }
      p
    }
  ),
  private = list(
    run = function() {
      if (self$dtype == "double") {
        grad <-
          torch_tensor(diag(self$converter$model_dict$output_dim),
            dtype = torch_double()
          )$unsqueeze(1)
      } else {
        grad <-
          torch_tensor(diag(self$converter$model_dict$output_dim),
            dtype = torch_float()
          )$unsqueeze(1)
      }

      index <- torch_tensor(self$output_idx, dtype = torch_long())
      grad <- grad[,,index, drop = FALSE]

      layers <- rev(self$converter$model$modules_list)
      message("Backwardpass 'ConnectionWeights':")
      # Define Progressbar
      pb <- txtProgressBar(min = 0, max = length(layers), style = 3)
      i <- 0

      for (layer in layers) {
        if ("Flatten_Layer" %in% layer$".classes") {
          grad <- layer$reshape_to_input(grad)
        } else {
          grad <- layer$get_gradient(grad, layer$W)
        }

        i <- i + 1
        setTxtProgressBar(pb, i)
      }
      if (!self$channels_first) {
        grad <- torch_movedim(grad, 2, length(dim(grad)) - 1)
      }
      close(pb)

      grad$squeeze(1)
    },
    get_dataframe = function() {
      result <- as.array(self$result)
      input_names <- self$converter$model_dict$input_names
      class <- unlist(self$converter$model_dict$output_names)[self$output_idx]

      if (length(input_names) == 1) {
        df <- expand.grid(
          feature = input_names[[1]],
          class = class
        )
      }
      # input (channels, signal_length)
      else if (length(input_names) == 2) {
        if (self$channels_first) {
          df <- expand.grid(
            channel = input_names[[1]],
            feature_l = input_names[[2]],
            class = class
          )
        } else {
          df <- expand.grid(
            feature_l = input_names[[2]],
            channel = input_names[[1]],
            class = class
          )
        }
      } else if (length(input_names) == 3) {
        if (self$channels_first) {
          df <- expand.grid(
            channel = input_names[[1]],
            feature_h = input_names[[2]],
            feature_w = input_names[[3]],
            class = class
          )
        } else {
          df <- expand.grid(
            feature_h = input_names[[2]],
            feature_w = input_names[[3]],
            channel = input_names[[1]],
            class = class
          )
        }
      }
      df$value <- as.vector(result)
      df
    }
  )
)
