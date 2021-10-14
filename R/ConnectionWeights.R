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
#'
#' @examples
#'
#' # We need libtorch to be installed
#' if (!torch::torch_is_installed()) {
#'   torch::install_torch()
#' }
#'
#' #----------------------- Example 1: Neuralnet ------------------------------
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
#' #----------------------- Example 2: Keras ----------------------------------
#'
#' # Define a model
#' model <- keras_model_sequential()
#' model %>%
#'   layer_conv_1d(
#'     input_shape = c(64, 3), kernel_size = 16, filters = 8,
#'     activation = "softplus"
#'   ) %>%
#'   layer_conv_1d(kernel_size = 16, filters = 4, activation = "tanh") %>%
#'   layer_conv_1d(kernel_size = 16, filters = 2, activation = "relu") %>%
#'   layer_flatten() %>%
#'   layer_dense(units = 64, activation = "relu") %>%
#'   layer_dense(units = 2, activation = "softmax")
#'
#' # Convert the model
#' converter <- Converter$new(model)
#'
#' # Apply the Connection Weights method
#' cw <- ConnectionWeights$new(converter)
#'
#' # Get the result as data.frame
#' cw$get_result(type = "data.frame")
#'
#' # Plot the result for all classes
#' plot(cw, class_id = 1:2)
#'
#' @references
#' * J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
#'  quantifying variable importance in artificial neural networks using
#'  simulated data.} Ecological Modelling 178, p. 389–397
#'
#' @export
ConnectionWeights <- R6::R6Class(
  classname = "ConnectionWeights",
  public = list(
    converter = NULL,
    channels_first = NULL,
    dtype = NULL,
    result = NULL,

    #' @param converter The converter of class [Converter] with the stored and
    #' torch-converted model.
    #' @param channels_first The data format of the result, i.e. channels on
    #' last dimension (`FALSE`) or on the first dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data and parameters
    #' (either `'float'` or `'double'`).
    #'
    initialize = function(converter,
                          channels_first = TRUE,
                          dtype = "float") {
      checkmate::assertClass(converter, "Converter")
      self$converter <- converter

      checkmate::assert_logical(channels_first)
      self$channels_first <- channels_first

      checkmate::assertChoice(dtype, c("float", "double"))
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
      checkmate::assertChoice(type, c("array", "data.frame", "torch.tensor"))

      result <- self$result
      if (type == "array") {
        result <- as.array(result)
      } else if (type == "data.frame") {
        result <- private$get_dataframe()
      } else if (type != "torch.tensor") {
        stop(sprintf(
          "Unknown data format '%s'! Use for argument 'type' one
                     of 'array', 'data.frame' or 'torch.tensor' instead.",
          type
        ))
      }

      result
    },

    #'
    #' @description
    #' This method visualizes the result of the ConnectionWeight method in a
    #' [ggplot2::ggplot]. You can use the argument `class_id` to select
    #' the classes for the plot. The different results for the selected classes
    #' are visualized using the method [ggplot2::facet_grid].
    #'
    #' @param class_id An integer vector containing the numbers of the classes
    #' whose result is to be plotted, e.g. `c(1,4)` for the first and fourth
    #' class. Default: `c(1)`.
    #' @param aggr_channels Pass a function to aggregate the channels. The
    #' default function is [base::sum], but you can pass an arbitrary function.
    #' For example, the maximum `max` or minimum `min` over the channels or
    #' only individual channels with `function(x) x[1]`.
    #'
    #' @return
    #' Returns a [ggplot2::ggplot] with the plotted results.
    #'
    plot = function(class_id = 1, aggr_channels = sum) {
      l <- length(dim(self$result))
      # 1D Input
      if (l == 2) {
        private$plot_1d_input(class_id)
      }
      # 2D Input
      else if (l == 3) {
        private$plot_2d_input(class_id, aggr_channels)
      }
      # 3D Input
      else if (l == 4) {
        private$plot_3d_input(class_id, aggr_channels)
      }
    }
  ),
  private = list(
    run = function() {
      if (self$dtype == "double") {
        grad <-
          torch::torch_tensor(diag(self$converter$model_dict$output_dim),
            dtype = torch::torch_double()
          )$unsqueeze(1)
      } else {
        grad <-
          torch::torch_tensor(diag(self$converter$model_dict$output_dim),
            dtype = torch::torch_float()
          )$unsqueeze(1)
      }

      for (layer in rev(self$converter$model$modules_list)) {
        if ("Flatten_Layer" %in% layer$".classes") {
          grad <- layer$reshape_to_input(grad)
        } else {
          grad <- layer$get_gradient(grad, layer$W)
        }
      }
      if (!self$channels_first) {
        grad <- torch::torch_movedim(grad, 2, length(dim(grad)) - 1)
      }

      grad$squeeze(1)
    },
    get_dataframe = function() {
      result <- as.array(self$result)
      input_names <- self$converter$model_dict$input_names
      class <- unlist(self$converter$model_dict$output_names)

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
    },
    plot_1d_input = function(class_id = 1, blank = TRUE) {
      checkmate::assertNumeric(class_id,
        lower = 1,
        upper = rev(dim(self$result))[1]
      )

      result <- private$get_dataframe()
      output_names <- unlist(self$converter$model_dict$output_names)
      result <- result[result$class %in% output_names[class_id], ]
      result$x <- as.numeric(result$feature)

      ggplot(data = result) +
        geom_rect(aes(
          xmin = x - 0.3,
          xmax = x + 0.3,
          ymin = 0,
          ymax = value,
          fill = value
        ),
        show.legend = FALSE
        ) +
        scale_fill_gradient2(low = "#377EB8", mid = "gray", high = "#E41A1C") +
        scale_x_discrete(limits = levels(result$feature)) +
        geom_hline(yintercept = 0) +
        facet_grid(cols = vars(class), scales = "free_y")
    },
    plot_2d_input = function(class_id = 1, aggr_channel = sum) {
      checkmate::assertNumeric(class_id,
        lower = 1,
        upper = rev(dim(self$result))[1]
      )

      result <- private$get_dataframe()
      output_names <- unlist(self$converter$model_dict$output_names)
      result <-
        result[result$class %in% output_names[class_id], ]
      result$x <- as.numeric(result$feature)
      result <-
        do.call(
          data.frame,
          aggregate(list(value = result$value),
            by = list(
              feature_l = result$feature_l,
              class = result$class,
              x = result$x
            ), FUN = aggr_channel
          )
        )

      result$value_scaled <- ave(
        x = result$value, FUN = function(x) x / max(abs(x))
      )

      ggplot(data = result) +
        geom_rect(aes(
          xmin = x - 0.5,
          xmax = x + 0.5,
          ymin = 0,
          ymax = value,
          fill = value_scaled
        ),
        show.legend = FALSE
        ) +
        scale_fill_gradient2(low = "#377EB8", mid = "gray", high = "#E41A1C") +
        scale_x_discrete(
          limits = levels(result$feature_l),
          labels = levels(result$feature_l)
        ) +
        geom_hline(yintercept = 0) +
        facet_grid(cols = vars(class), scale = "free_y")
    },
    plot_3d_input = function(class_id = 1, aggr_channel = sum) {
      checkmate::assertNumeric(class_id,
        lower = 1,
        upper = rev(dim(self$result))[1]
      )

      result <- private$get_dataframe()
      output_names <- unlist(self$converter$model_dict$output_names)
      result <- result[result$class %in% output_names[class_id], ]

      result <- do.call(
        data.frame,
        aggregate(list(value = result$value),
          by = list(
            feature_h = result$feature_h,
            feature_w = result$feature_w,
            class = result$class
          ),
          FUN = aggr_channel
        )
      )

      result$feature_h <- as.numeric(result$feature_h)
      result$feature_w <- as.numeric(result$feature_w)

      ggplot(data = result, aes(x = feature_w, y = feature_h, fill = value)) +
        geom_raster() +
        scale_fill_gradient2(low = "#377EB8", mid = "gray", high = "#E41A1C") +
        facet_grid(cols = vars(class), scale = "free_y") +
        xlab("") +
        ylab("")
    }
  )
)
