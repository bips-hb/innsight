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
InterpretingMethod <- R6Class(
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
      assertClass(converter, "Converter")
      self$converter <- converter

      assert_logical(channels_first)
      self$channels_first <- channels_first

      assert_logical(ignore_last_act)
      self$ignore_last_act <- ignore_last_act

      assertChoice(dtype, c("float", "double"))
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
      assertChoice(type, c("array", "data.frame", "torch.tensor"))

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
        data <- torch_tensor(data, dtype = torch_float())
      } else {
        data <- torch_tensor(data, dtype = torch_double())
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

    # ----------------------- Plot Function ----------------------------------

    plot = function(datapoint = 1,
                    classes = 1,
                    aggr_channels = sum,
                    as_plotly = FALSE,
                    value_name = "value") {

      assertNumeric(datapoint, lower = 1, upper = dim(self$result)[1])
      assertNumeric(classes, lower = 1, upper = rev(dim(self$result))[1])
      assertFunction(aggr_channels)
      assertLogical(as_plotly)

      num_dims <- length(dim(self$result))

      result <- private$get_dataframe()
      output_names <- unlist(self$converter$model_dict$output_names)
      result <- result[result$data %in% paste0("data_", datapoint) &
                         result$class %in% output_names[classes], ]
      # 1D Input
      if (num_dims == 3) {
        p <- plot_1d_input(result, value_name)
        dynamicTicks <- FALSE
      }
      # 2D Input
      else if (num_dims == 4) {
        p <- plot_2d_input(result, aggr_channels, value_name)
        dynamicTicks <- TRUE
      }
      # 3D Input
      else if (num_dims == 5) {
        p <- plot_3d_input(result, aggr_channels, value_name)
        dynamicTicks <- TRUE
      }

      p <- p +
        theme(
          strip.text.x = element_text(size = 10),
          strip.text.y = element_text(size = 10),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12))

      if (as_plotly) {
        if (!requireNamespace("plotly", quietly = FALSE)) {
          stop("Please install the 'plotly' package if you want to create an
         interactive plot.")
        }
        p <-
          plotly::ggplotly(p, tooltip = "text", dynamicTicks = dynamicTicks)
      }
      p
    },

    # -------------------- Summary Plots -------------------------------------

    boxplot = function(preprocess_FUN, boxplot_data, classes, ref_datapoint,
                       aggr_channels, individual_data,
                       individual_max, as_plotly, value_name) {

      assertFunction(preprocess_FUN)
      assertFunction(aggr_channels)
      assertLogical(as_plotly)
      assert(
        checkNumeric(boxplot_data, lower = 1, upper = dim(self$result)[1]),
        checkChoice(boxplot_data, c("all"))
      )
      assertNumeric(classes, lower = 1, upper = rev(dim(self$result))[1])
      assertInt(ref_datapoint,
                               lower = 1,
                               upper = dim(self$result)[1], null.ok = TRUE
                               )
      checkNumeric(individual_data,
                              lower = 1,
                              upper = dim(self$result)[1], null.ok = TRUE)

      if (is.character(boxplot_data) && boxplot_data == "all") {
        boxplot_data <- 1:dim(self$result)[1]
      }

      if (is.null(individual_data)) {
        individual_data <- boxplot_data
      }
      if (length(individual_data) > individual_max) {
        individual_data <- individual_data[1:individual_max]
      }

      l <- length(dim(self$result))

      result <- private$get_dataframe()
      all_data_ids <- c(boxplot_data, individual_data, ref_datapoint)
      output_names <- unlist(self$converter$model_dict$output_names)
      result <- result[result$data %in% paste0("data_", all_data_ids) &
                         result$class %in% output_names[classes], ]
      result$summary_data <- result$data %in% paste0("data_", boxplot_data)
      result$individual_data <-
        result$data %in% paste0("data_", c(individual_data, ref_datapoint))

      result$value <- preprocess_FUN(result$value)

      if (as_plotly) {
        p <- boxplot_plotly(result, aggr_channels, ref_datapoint, value_name)
      } else {
        p <- boxplot_ggplot(result, aggr_channels, ref_datapoint, value_name)
      }

      p
    }
  )
)
