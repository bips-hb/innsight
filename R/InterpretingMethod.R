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
#' @field output_idx This vector determines for which outputs the method
#' will be applied.
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
    output_idx = NULL,

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
    #' @param output_idx This vector determines for which output indices the
    #' method will be applied. By default (`NULL`), all outputs (but limited to
    #' the first 10) are considered.

    initialize = function(converter, data,
                          channels_first = TRUE,
                          dtype = "float",
                          ignore_last_act = TRUE,
                          output_idx = NULL) {
      assertClass(converter, "Converter")
      self$converter <- converter

      assert_logical(channels_first)
      self$channels_first <- channels_first

      assert_logical(ignore_last_act)
      self$ignore_last_act <- ignore_last_act

      assertChoice(dtype, c("float", "double"))
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      assertIntegerish(output_idx,
        null.ok = TRUE, lower = 1,
        upper = converter$model_dict$output_dim
      )


      if (is.null(output_idx)) {
        output_idx <-
          1:min(converter$model_dict$output_dim, 10)
      }
      self$output_idx <- output_idx

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
      class <- unlist(self$converter$model_dict$output_names)[self$output_idx]

      if (length(input_names) == 1) {
        df <- expand.grid(
          data = data_names,
          feature = input_names[[1]],
          class = class
        )
      }
      # input (channels, signal_length)
      else if (length(input_names) == 2) {
        if (self$channels_first) {
          df <- expand.grid(
            data = data_names,
            channel = input_names[[1]],
            feature_l = input_names[[2]],
            class = class
          )
        } else {
          df <- expand.grid(
            data = data_names,
            feature_l = input_names[[2]],
            channel = input_names[[1]],
            class = class
          )
        }
      } else if (length(input_names) == 3) {
        if (self$channels_first) {
          df <- expand.grid(
            data = data_names,
            channel = input_names[[1]],
            feature_h = input_names[[2]],
            feature_w = input_names[[3]],
            class = class
          )
        } else {
          df <- expand.grid(
            data = data_names,
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

    # ----------------------- Plot Function ----------------------------------

    plot = function(datapoint = 1,
                    output_idx = c(),
                    aggr_channels = 'sum',
                    as_plotly = FALSE,
                    value_name = "value") {

      # Check correctness of arguments
      assertNumeric(datapoint, lower = 1, upper = dim(self$result)[1])
      assertSubset(output_idx, self$output_idx)
      assert(
        checkFunction(aggr_channels),
        checkChoice(aggr_channels, c("norm", "sum", "mean"))
      )
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

      # The input and output names are used for every plot-function
      input_names <- self$converter$model_dict$input_names
      output_names <- unlist(self$converter$model_dict$output_names)[classes]

      # Depending on the number of dimensions, a different plot-function
      # is used
      num_dims <- length(dim(self$result))
      # 1D Input
      if (num_dims == 3) {
        # Filter all results by the given 'datapoint' and 'classes'
        res <- self$result[datapoint, , classes_idx, drop = FALSE]
        # Plot the result
        p <- plot_1d_input(
          res, value_name, paste0("data_", datapoint),
          input_names,
          output_names,
          self$channels_first, FALSE
        )
        dynamicTicks <- FALSE
      }
      # 2D Input
      else if (num_dims == 4) {
        # Filter all results by the given 'datapoint' and 'classes'
        res <- as_array(self$result[datapoint, , , classes_idx, drop = FALSE])

        # Depending on the dataformat the channels are on axis '2' or '3'
        if (self$channels_first) {
          dims <- c(1, 3, 4)
          d <- 2
        } else {
          dims <- c(1, 2, 4)
          d <- 3
        }
        # Summarize the channels by function 'aggr_channels'
        res <- torch_tensor(apply(res, dims, aggr_channels))$unsqueeze(d)
        # Modify input names because we changed the number of channels
        input_names[[1]] <- c("aggr")

        # Plot the result
        p <- plot_2d_input(
          res, value_name, paste0("data_", datapoint), input_names,
          output_names, self$channels_first, FALSE
        )
        dynamicTicks <- TRUE
      }
      # 3D Input
      else if (num_dims == 5) {
        # Filter all results by the given 'datapoint' and 'classes'
        res <- as_array(self$result[datapoint, , , , classes_idx, drop = FALSE])
        # Depending on the dataformat the channels are on axis '2' or '3'
        if (self$channels_first) {
          dims <- c(1, 3, 4, 5)
          d <- 2
        } else {
          dims <- c(1, 2, 3, 5)
          d <- 4
        }
        # Summarize the channels by function 'aggr_channels'
        res <- torch_tensor(apply(res, dims, aggr_channels))$unsqueeze(d)
        # Modify input names because we changed the number of channels
        input_names[[1]] <- c("aggr")
        # Plot the result
        p <- plot_3d_input(
          res, value_name, paste0("data_", datapoint), input_names,
          output_names, self$channels_first, FALSE
        )
        dynamicTicks <- TRUE
      }

      # Set the size of labels and titles
      p <- p +
        theme(
          strip.text.x = element_text(size = 10),
          strip.text.y = element_text(size = 10),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12)
        )

      # If 'as_plotly = TRUE', we transform the ggplot into a plotly plot by
      # using 'plotly::ggplotly'
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

    # ------------------------ Boxplots -------------------------------------

    boxplot = function(preprocess_FUN, boxplot_data, classes, ref_datapoint,
                       aggr_channels, individual_data,
                       individual_max, as_plotly, value_name) {
      # Check correctness of arguments
      assertFunction(preprocess_FUN)
      assertFunction(aggr_channels)
      assertLogical(as_plotly)
      assert(
        checkNumeric(boxplot_data, lower = 1, upper = dim(self$result)[1]),
        checkChoice(boxplot_data, c("all"))
      )
      assertNumeric(classes, lower = 1, upper = rev(dim(self$result))[1])
      assertInt(ref_datapoint,
        lower = 1, upper = dim(self$result)[1], null.ok = TRUE
      )
      checkNumeric(individual_data,
        lower = 1, upper = dim(self$result)[1], null.ok = TRUE
      )

      # Set default value for 'boxplot_data'
      if (is.character(boxplot_data) && boxplot_data == "all") {
        boxplot_data <- 1:dim(self$result)[1]
      }

      # Set default for 'individual_data' (only for plotly plots)
      if (is.null(individual_data)) {
        individual_data <- boxplot_data
      }
      if (length(individual_data) > individual_max) {
        individual_data <- individual_data[1:individual_max]
      }

      if (as_plotly) {
        result <- private$get_dataframe()
        all_data_ids <- c(boxplot_data, individual_data, ref_datapoint)
        output_names <- unlist(self$converter$model_dict$output_names)
        result <- result[result$data %in% paste0("data_", all_data_ids) &
          result$class %in% output_names[classes], ]
        result$summary_data <- result$data %in% paste0("data_", boxplot_data)
        result$individual_data <-
          result$data %in% paste0("data_", c(individual_data, ref_datapoint))

        result$value <- preprocess_FUN(result$value)
        p <- boxplot_plotly(result, aggr_channels, ref_datapoint, value_name)
      } else {
        output_names <- unlist(self$converter$model_dict$output_names)[classes]
        input_names <- self$converter$model_dict$input_names
        p <- boxplot_ggplot(
          self$result, aggr_channels, ref_datapoint, value_name,
          boxplot_data, classes, input_names, output_names,
          preprocess_FUN, self$channels_first
        )
      }

      p
    }
  )
)
