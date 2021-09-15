#' @title Superclass for Interpreting Methods
#' @description This is a superclass for all data-based interpreting methods.
#' Implemented are the following methods:
#'
#' - Deep Learning Important Features ([DeepLift])
#' - Layer-wise Relevance Propagation ([LRP])
#' - Gradient-based methods:
#'    - Normal gradients ([Gradient])
#'    - Smoothed Gradients ([SmoothGrad])
#'
#'
#' @field data The given data as a torch tensor to be interpreted with the
#' selected method.
#' @field converter The converter with the stored and torch-converted model.
#' @field dtype The type of the data (either `'float'` or `'double'`).
#' @field channels_first The format of the given date, i.e. channels on
#' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
#' data has no channels, use the default value `TRUE`.
#' @field ignore_last_act A boolean value to include the last
#' activation into all the calculations, or not. In some cases, the last
#' activation leads to a saturation problem.
#' @field result The methods result of the given data as a
#' torch tensor of size (batch_size, dim_in, dim_out).
#'
#' @import ggplot2
#'
InterpretingMethod <- R6::R6Class(
  classname = "InterpretingMethod",
  public = list(
    data = NULL,
    converter = NULL,
    channels_first = NULL,
    dtype = NULL,
    ignore_last_act = NULL,
    result = NULL,

    #' @description
    #' Create a new instance of this class.
    #'
    #' @param converter The converter with the stored and torch-converted model.
    #' @param data The given data in an array-like format to be interpreted
    #' with the selected method.
    #' @param channels_first The format of the given data, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data (either `'float'` or `'double'`).
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not. In some cases, the last
    #' activation leads to a saturation problem.

    initialize = function(converter, data,
                          channels_first = TRUE,
                          dtype = "float",
                          ignore_last_act = TRUE) {
      checkmate::assertClass(converter, "Converter")
      self$converter <- converter

      checkmate::assert_logical(channels_first)
      self$channels_first <- channels_first

      checkmate::assert_logical(ignore_last_act)
      self$ignore_last_act <- ignore_last_act

      checkmate::assertChoice(dtype, c("float", "double"))
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      self$data <- private$test_data(data)
    },

    #'
    #' @description
    #' This function returns the result of this method for the given data
    #' either as an array (`'array'`), a torch tensor (`'torch.tensor'`) of
    #' size (batch_size, dim_in, dim_out) or a data.frame (`'data.frame'`).
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
    #' This method visualizes the result of the selected method in a
    #' [ggplot2::ggplot]. You can use the argument `data_id` to select
    #' the data points in the given data for the plot. In addition, the
    #' individual classes for the plot can be selected with the argument
    #' `class_id`. The different results for the selected data points and
    #' classes are visualized using the method [ggplot2::facet_grid].
    #'
    #' @param data_id An integer vector containing the numbers of the data
    #' points whose result is to be plotted, e.g. `c(1,3)` for the first
    #' and third data point in the given data. Default: `c(1)`.
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
    plot = function(data_id = 1, class_id = 1, aggr_channels = sum) {
      l <- length(dim(self$result))
      # 1D Input
      if (l == 3) {
        private$plot_1d_input(data_id, class_id)
      }
      # 2D Input
      else if (l == 4) {
        private$plot_2d_input(data_id, class_id, aggr_channels)
      }
      # 3D Input
      else if (l == 5) {
        private$plot_3d_input(data_id, class_id, aggr_channels)
      }
    }
  ),
  private = list(
    test_data = function(data, name = "data") {
      if (missing(data)) {
        stop("Argument 'data' is missing!")
      }
      data <- tryCatch({
          if (is.data.frame(data)) {
            data <- as.matrix(data)
          }
          as.array(data)
        },
        error = function(e) {
          stop(sprintf("Failed to convert the argument '%s' to an array using
                       the function 'base::as.array'. The class of your '%s':
                       %s", name, name, class(data)))
        }
      )

      ordered_dim <- self$converter$model_dict$input_dim
      if (!self$channels_first) {
        channels <- ordered_dim[1]
        ordered_dim <- c(ordered_dim[-1], channels)
      }

      if (length(dim(data)[-1]) != length(ordered_dim) ||
        !all(dim(data)[-1] == ordered_dim)) {
        stop(sprintf(
          "Unmatch in model dimension (*,%s) and dimension of argument '%s'
          (%s). Try to change the argument 'channels_first', if only
          the channels are wrong.",
          paste0(ordered_dim, sep = "", collapse = ","),
          name,
          paste0(dim(data), sep = "", collapse = ",")
        ))
      }


      if (self$dtype == "float") {
        data <- torch::torch_tensor(data,
          dtype = torch::torch_float()
        )
      } else {
        data <- torch::torch_tensor(data,
          dtype = torch::torch_double()
        )
      }

      data
    },
    get_dataframe = function() {
      result <- as.array(self$result)
      input_names <- self$converter$model_dict$input_names
      num_data <- dim(result)[1]
      data_names <- paste0("data_", 1:num_data)

      if (length(input_names) == 1) {
        df <- expand.grid(
          data = data_names,
          feature = input_names[[1]],
          class = unlist(self$converter$model_dict$output_names)
        )
      }
      # input (channels, signal_length)
      else if (length(input_names) == 2) {
        if (self$channels_first) {
          df <- expand.grid(
            data = data_names,
            channel = input_names[[1]],
            feature_l = input_names[[2]],
            class = unlist(self$converter$model_dict$output_names)
          )
        } else {
          df <- expand.grid(
            data = data_names,
            feature_l = input_names[[2]],
            channel = input_names[[1]],
            class = unlist(self$converter$model_dict$output_names)
          )
        }
      } else if (length(input_names) == 3) {
        if (self$channels_first) {
          df <- expand.grid(
            data = data_names,
            channel = input_names[[1]],
            feature_h = input_names[[2]],
            feature_w = input_names[[3]],
            class = unlist(self$converter$model_dict$output_names)
          )
        } else {
          df <- expand.grid(
            data = data_names,
            feature_h = input_names[[2]],
            feature_w = input_names[[3]],
            channel = input_names[[1]],
            class = unlist(self$converter$model_dict$output_names)
          )
        }
      }
      df$value <- as.vector(result)
      df
    },
    plot_1d_input = function(data_id = 1, class_id = 1, blank = TRUE) {
      checkmate::assertNumeric(data_id,
        lower = 1,
        upper = dim(self$result)[1]
      )
      checkmate::assertNumeric(class_id,
        lower = 1,
        upper = rev(dim(self$result))[1]
      )

      result <- private$get_dataframe()
      output_names <- unlist(self$converter$model_dict$output_names)
      result <- result[result$data %in% paste0("data_", data_id) &
        result$class %in% output_names[class_id], ]
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
        facet_grid(rows = vars(data), cols = vars(class), scales = "free_y")
    },
    plot_2d_input = function(data_id = 1, class_id = 1, aggr_channel = sum) {
      checkmate::assertNumeric(data_id,
        lower = 1,
        upper = dim(self$result)[1]
      )
      checkmate::assertNumeric(class_id,
        lower = 1,
        upper = rev(dim(self$result))[1]
      )

      result <- private$get_dataframe()
      output_names <- unlist(self$converter$model_dict$output_names)
      result <-
        result[result$data %in% paste0("data_", data_id) &
          result$class %in% output_names[class_id], ]
      result$x <- as.numeric(result$feature)
      result <-
        do.call(
          data.frame,
          aggregate(list(value = result$value),
            by = list(
              data = result$data,
              feature_l = result$feature_l,
              class = result$class,
              x = result$x
            ), FUN = aggr_channel
          )
        )

      result$value_scaled <- ave(
        x = result$value, factor(result$data),
        FUN = function(x) x / max(abs(x))
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
        facet_grid(rows = vars(data), cols = vars(class), scale = "free_y")
    },
    plot_3d_input = function(data_id = 1, class_id = 1, aggr_channel = sum) {
      checkmate::assertNumeric(data_id,
        lower = 1,
        upper = dim(self$result)[1]
      )
      checkmate::assertNumeric(class_id,
        lower = 1,
        upper = rev(dim(self$result))[1]
      )

      result <- private$get_dataframe()
      output_names <- unlist(self$converter$model_dict$output_names)
      result <- result[result$data %in% paste0("data_", data_id) &
        result$class %in% output_names[class_id], ]

      result <- do.call(
        data.frame,
        aggregate(list(value = result$value),
          by = list(
            data = result$data,
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
        facet_grid(rows = vars(data), cols = vars(class), scale = "free_y") +
        xlab("") +
        ylab("")
    }
  )
)
