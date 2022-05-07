
###############################################################################
#                         Interpreting Method
###############################################################################

#' @title Super class for Interpreting Methods
#' @description This is a super class for all data-based interpreting methods.
#' Implemented are the following methods:
#'
#' - Deep Learning Important Features ([DeepLift])
#' - Layer-wise Relevance Propagation ([LRP])
#' - Gradient-based methods:
#'    - Vanilla gradients including 'Gradients x Input' ([Gradient])
#'    - Smoothed gradients including 'SmoothGrad x Input' ([SmoothGrad])
#'
#'
#' @field data The passed data as a torch tensor in the given data type
#' (`dtype`) to be interpreted with the selected method.
#' @field converter An instance of the R6 class \code{\link{Converter}}.
#' @field dtype The data type for the calculations. Either `'float'`
#' for [torch::torch_float] or `'double'` for [torch::torch_double].
#' @field channels_first The format of the given date, i.e. channels on
#' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
#' data has no channels, the default value `TRUE` is used.
#' @field ignore_last_act A boolean value to include the last
#' activation into all the calculations, or not (default: `TRUE`). In some
#' cases, the last activation leads to a saturation problem.
#' @field result The methods result of the given data as a
#' torch tensor of size *(batch_size, dim_in, dim_out)* in the given data type
#' (`dtype`).
#' @field output_idx This vector determines for which outputs the method
#' will be applied. By default (`NULL`), all outputs (but limited to the
#' first 10) are considered.
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
    #' Create a new instance of this super class.
    #'
    #' @param converter An instance of the R6 class \code{\link{Converter}}.
    #' @param data The data for which this method is to be applied. It has
    #' to be an array or array-like format of size *(batch_size, dim_in)*.
    #' @param channels_first The format of the given data, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype dtype The data type for the calculations. Use
    #' either `'float'` for [torch::torch_float] or `'double'` for
    #' [torch::torch_double].
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not (default: `TRUE`). In
    #' some cases, the last activation leads to a saturation problem.
    #' @param output_idx This vector determines for which output indices the
    #' method will be applied. By default (`NULL`), all outputs (but limited to
    #' the first 10) are considered.

    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          dtype = "float") {
      assertClass(converter, "Converter")
      self$converter <- converter

      assert_logical(channels_first)
      self$channels_first <- channels_first

      assert_logical(ignore_last_act)
      self$ignore_last_act <- ignore_last_act

      assertChoice(dtype, c("float", "double"))
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      # Check output indices
      self$output_idx <- check_output_idx(output_idx, converter$output_dim)

      self$data <- private$test_data(data)
    },

    #'
    #' @description
    #' This function returns the result of this method for the given data
    #' either as an array (`'array'`), a torch tensor (`'torch.tensor'`,
    #' or `'torch_tensor'`) of size *(batch_size, dim_in, dim_out)* or as a
    #' data.frame (`'data.frame'`).
    #'
    #' @param type The data type of the result. Use one of `'array'`,
    #' `'torch.tensor'`, `'torch_tensor'` or `'data.frame'`
    #' (default: `'array'`).
    #'
    #' @return The result of this method for the given data in the chosen
    #' type.
    #'

    get_result = function(type = "array") {
      assertChoice(type, c("array", "data.frame", "torch.tensor",
                           "torch_tensor"))

      # Get the result as an array
      if (type == "array") {
        # Get the input names and move the channel dimension (if necessary)
        input_names <- self$converter$input_names
        if (!self$channels_first) {
          input_names <- move_channels_last(input_names)
        }
        # Convert the torch_tensor result into a named array
        result <- tensor_list_to_named_array(
          self$result, input_names, self$converter$output_names, self$output_idx)
      }
      # Get the result as a data.frame
      else if (type == "data.frame") {
        # Convert the torch_tensor result into a data.frame
        result <- create_dataframe_from_result(
          seq_len(dim(self$data[[1]])[1]), self$result, self$converter$input_names,
          self$converter$output_names, self$output_idx)
        # Remove unnecessary columns
        if (all(result$input_dimension <= 2)) {
          result$feature_2 <- NULL
        }
        if (all(result$input_dimension <= 1)) {
          result$channel <- NULL
        }
      }
      # Get the result as a torch_tensor and remove unnecessary axis
      else {
        num_inputs <- length(self$converter$input_names)
        out_null_idx <- unlist(lapply(self$output_idx, is.null))
        out_nonnull_idx <- seq_along(self$converter$output_names)[!out_null_idx]
        result <- self$result
        # Name the inner list or remove the inner list
        for (out_idx in seq_along(result)) {
          if (num_inputs == 1) {
            result[[out_idx]] <- result[[out_idx]][[1]]
          } else {
            names(result[[out_idx]]) <- paste0("Input_", seq_len(num_inputs))
          }
        }
        # Name the outer list or remove it
        if (length(self$output_idx) == 1) {
          result <- result[[1]]
        } else {
          names(result) <- paste0("Output_", out_nonnull_idx)
        }
      }

      result
    }
  ),
  private = list(
    test_data = function(data, name = "data") {
      if (missing(data)) {
        stop("Argument 'data' is missing!")
      }
      if (!is.list(data) | is.data.frame(data)) {
        data <- list(data)
      }

      lapply(seq_along(data), function(i) {
        input_data <- data[[i]]
        input_data <- tryCatch({
          if (is.data.frame(input_data)) {
            input_data <- as.matrix(input_data)
          }
          as.array(input_data)
        },
        error = function(e) {
          stop("Failed to convert the argument '", name, "[[", i, "]]' to an array ",
               "using the function 'base::as.array'. The class of your ",
               "argument '", name, "[[", i, "]]': '",
               paste(class(input_data), collapse = "', '"), "'")
        })

        ordered_dim <- self$converter$input_dim[[i]]
        if (!self$channels_first) {
          channels <- ordered_dim[1]
          ordered_dim <- c(ordered_dim[-1], channels)
        }

        if (length(dim(input_data)[-1]) != length(ordered_dim) ||
            !all(dim(input_data)[-1] == ordered_dim)) {
          stop(
            "Unmatch in model dimension (*, ",
            paste0(ordered_dim, collapse = ", "), ") and dimension of ",
            "argument '", name, "[[", i, "]]' (",
            paste0(dim(input_data), collapse = ", "),
            "). Try to change the argument 'channels_first', if only ",
            "the channels are wrong."
          )
        }


        if (self$dtype == "float") {
          input_data <- torch_tensor(input_data, dtype = torch_float())
        } else {
          input_data <- torch_tensor(input_data, dtype = torch_double())
        }

        input_data
      })
    },

    # ----------------------- Plot Function ----------------------------------

    plot = function(data_idx = 1,
                    output_idx = c(),
                    aggr_channels = 'sum',
                    as_plotly = FALSE,
                    value_name = "value") {

      # Check correctness of arguments
      assertNumeric(data_idx, lower = 1, upper = dim(self$result)[1])
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
        # Filter all results by the given 'data_idx' and 'classes'
        res <- self$result[data_idx, , classes_idx, drop = FALSE]
        # Plot the result
        p <- plot_1d_input(
          res, value_name, paste0("data_", data_idx),
          input_names,
          output_names,
          self$channels_first, FALSE
        )
        dynamicTicks <- FALSE
      }
      # 2D Input
      else if (num_dims == 4) {
        # Filter all results by the given 'data_idx' and 'classes'
        res <- as_array(self$result[data_idx, , , classes_idx, drop = FALSE])

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
          res, value_name, paste0("data_", data_idx), input_names,
          output_names, self$channels_first, FALSE
        )
        dynamicTicks <- TRUE
      }
      # 3D Input
      else if (num_dims == 5) {
        # Filter all results by the given 'data_idx' and 'classes'
        res <- as_array(self$result[data_idx, , , , classes_idx, drop = FALSE])
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
          res, value_name, paste0("data_", data_idx), input_names,
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
          stop("Please install the 'plotly' package if you want to create",
               "an interactive plot.")
        }
        p <-
          plotly::ggplotly(p, tooltip = "text", dynamicTicks = dynamicTicks)
      }
      p
    },

    # ------------------------ Boxplots -------------------------------------

    boxplot = function(output_idx, data_idx, ref_data_idx, aggr_channels,
                       preprocess_FUN, as_plotly, individual_data_idx,
                       individual_max, value_name) {

      # Check correctness of arguments
      dim_result <- dim(self$result)
      assertFunction(preprocess_FUN)
      assert(
        checkFunction(aggr_channels),
        checkChoice(aggr_channels, c("norm", "sum", "mean"))
      )
      assertLogical(as_plotly)
      assert(
        checkNumeric(data_idx, lower = 1, upper = dim_result[1]),
        checkChoice(data_idx, c("all"))
      )
      assertSubset(output_idx, self$output_idx)
      assertInt(ref_data_idx,
        lower = 1, upper = dim_result[1], null.ok = TRUE
      )
      checkNumeric(individual_data_idx,
        lower = 1, upper = dim_result[1], null.ok = TRUE
      )

      # Set default value for 'data_idx'
      if (is.character(data_idx) && data_idx == "all") {
        data_idx <- 1:(dim_result[1])
      }

      # Set aggregation function for channels
      if (!is.function(aggr_channels)) {
        if (aggr_channels == "norm") {
          aggr_channels <- function(x) sum(x^2)^0.5
        } else if (aggr_channels == "sum") {
          aggr_channels <- sum
        } else if (aggr_channels == "mean") {
          aggr_channels <- mean
        }
      }

      # Set default value for 'output_idx'
      if (length(output_idx) == 0) {
        classes <- self$output_idx[1]
        classes_idx <- 1
      } else {
        classes <- output_idx
        classes_idx <- match(classes, self$output_idx)
      }

      # Set default for 'individual_data_idx' (only for plotly plots)
      if (is.null(individual_data_idx)) {
        individual_data_idx <- data_idx
      }
      if (length(individual_data_idx) > individual_max) {
        individual_data_idx <- individual_data_idx[1:individual_max]
      }

      if (as_plotly) {
        result <- private$get_dataframe()
        all_data_ids <- c(data_idx, individual_data_idx, ref_data_idx)
        output_names <- unlist(self$converter$model_dict$output_names)
        result <- result[result$data %in% paste0("data_", all_data_ids) &
          result$class %in% output_names[classes], ]
        result$summary_data <- result$data %in% paste0("data_", data_idx)
        result$individual_data <-
          result$data %in% paste0("data_", c(individual_data_idx, ref_data_idx))

        result$value <- preprocess_FUN(result$value)
        p <- boxplot_plotly(result, aggr_channels, ref_data_idx, value_name)
      } else {
        output_names <- unlist(self$converter$model_dict$output_names)[classes]
        input_names <- self$converter$model_dict$input_names
        p <- boxplot_ggplot(
          self$result, aggr_channels, ref_data_idx, value_name,
          data_idx, classes_idx, input_names, output_names,
          preprocess_FUN, self$channels_first
        )
      }

      p
    }
  )
)


###############################################################################
#                                 Utils
###############################################################################

check_output_idx <- function(output_idx, output_dim) {
  # for the default value, choose from the first output the first ten
  # (maybe less) output nodes
  if (is.null(output_idx)) {
    output_idx <- list(1:min(10, output_dim[[1]]))
  }
  # or only a number (assumes the first output)
  else if (testIntegerish(output_idx,
                          lower = 1,
                          upper = output_dim[[1]])) {
    output_idx <- list(output_idx)
  }
  # the argument output_idx is a list of output_nodes for each output
  else if (testList(output_idx, max.len = length(output_dim))) {
    n <- 1
    for (output in output_idx) {
      limit <- output_dim[[n]]
      assertInt(limit)
      if (!testIntegerish(output, lower = 1, upper = limit, null.ok = TRUE)) {
        stop("Assertion on 'output_idx[[", n, "]]' failed: Values ",
             paste(output, collapse = ",")," is not <= ", limit, ".")
      }
      n <- n + 1
    }
  } else {
    stop("The argument 'output_idx' has to be either a vector with maximum value of '",
         output_dim[[1]], "' or a list of length '",
         length(output_dim), "' with maximal values of '",
         paste(unlist(output_dim), collapse = ","), "'.")
  }

  # Fill up with NULLs
  if (length(output_idx) < length(output_dim)) {
    output_idx <-
      append(output_idx,
             rep(list(NULL), length(output_dim) - length(output_idx)))
  }

  output_idx
}


tensor_list_to_named_array <- function(torch_result, input_names, output_names,
                                       output_idx) {
  # get the indices of the output for which we have and haven't calculated
  # attribution values
  out_null_idx <- unlist(lapply(output_idx, is.null))
  out_nonnull_idx <- seq_along(output_names)[!out_null_idx]

  # select only relevant output indices and output names
  output_idx <- output_idx[out_nonnull_idx]
  output_names <- output_names[out_nonnull_idx]

  # 'torch_result' is a list (with output layer indices) of list (input layer
  # indices) and the inner list contains the corresponding result of the
  # respective output and input layer combination
  result <- lapply(
    # for each output layer
    seq_along(torch_result),
    function(out_idx) {
      result_i <- lapply(
        # and for each input layer
        seq_along(torch_result[[out_idx]]),
        function(in_idx) {
          # get the corresponding result
          result_ij <- torch_result[[out_idx]][[in_idx]]
          # if the output layer isn't connected to the input layer, we set
          # the value NaN
          if (is.null(result_ij)) {
            result_ij <- NaN
          }
          # otherwise convert the result to an array and set dimnames
          else {
            result_ij <- as_array(result_ij)
            in_name <- input_names[[in_idx]]
            out_name <-
              list(output_names[[out_idx]][[1]][output_idx[[out_idx]]])
            names <- append(list(NULL), in_name)
            names <- append(names, out_name)
            dimnames(result_ij) <- names
          }

          result_ij
        }
      )
      # Skip one list dimension if there is only one input layer, otherwise
      # set the names of the list entries
      if (length(input_names) == 1) {
        result_i <- result_i[[1]]
      } else {
        names(result_i) <- paste0("Input_", seq_along(input_names))
      }

      result_i
    }
  )
  # Skip one list dimension if there is only one output layer, otherwise set
  # the names of the list entries
  if (length(output_idx) == 1) {
    result <- result[[1]]
  } else {
    names(result) <- paste0("Output_", out_nonnull_idx)
  }

  result
}


create_dataframe_from_result <- function(data_idx, result, input_names,
                                         output_names, output_idx) {

  null_idx <- unlist(lapply(output_idx, is.null))
  nonnull_idx <- seq_along(output_names)[!null_idx]
  output_idx <- output_idx[nonnull_idx]
  output_names <- output_names[nonnull_idx]

  fun <- function(result, out_idx, in_idx, input_names, output_names,
                  output_idx, nonnull_idx) {
    res <- result[[out_idx]][[in_idx]]
    result_df <-
      create_grid(data_idx, input_names[[in_idx]],
                  output_names[[out_idx]][[1]][output_idx[[out_idx]]])
    if (is.null(res)) {
      result_df$value <- NaN
    } else {
      result_df$value <- as.vector(as.array(res))
    }
    result_df$model_input <- paste0("Input_", in_idx)
    result_df$model_output <- paste0("Output_", nonnull_idx[out_idx])

    result_df
  }

  result <- apply_results(result, fun, input_names, output_names,
                          output_idx, nonnull_idx)
  result_df <- do.call("rbind",
                       lapply(result, function(x) do.call("rbind", x)))

  result_df[, c(1, 8, 9, 3, 4, 2, 5, 7, 6)]
}

create_grid <- function(data_idx, input_names, output_names) {
  dimension <- length(input_names)

  if (dimension == 1) {
    feature = input_names[[1]]
    feature_2 <- NaN
    channel <- NaN
  } else if (dimension == 2) {
    feature = input_names[[2]]
    feature_2 <- NaN
    channel <- input_names[[1]]
  } else {
    feature = input_names[[2]]
    feature_2 <- input_names[[3]]
    channel <- input_names[[1]]
  }

  expand.grid(data = paste0("data_", data_idx),
              channel = channel,
              feature = feature,
              feature_2 = feature_2,
              output_node = output_names,
              input_dimension = dimension)
}

move_channels_last <- function(names) {
  for (idx in seq_along(names)) {
    if (length(names[[idx]]) == 2) { # 1d input
      names[[idx]] <- names[[idx]][c(2,1)]
    } else if (length(names[[idx]]) == 3) { # 2d input
      names[[idx]] <- names[[idx]][c(2,3,1)]
    }
  }

  names
}


apply_results <- function(result, FUN, ...) {
  # loop over all the output layers
  lapply(seq_along(result), function(out_idx) {
    # then loop over all input layers
    lapply(seq_along(result[[out_idx]]), function(in_idx) {
      # apply FUN to results
      FUN(result, out_idx, in_idx, ...)
    })
  })
}

