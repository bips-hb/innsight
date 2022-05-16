
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

    # ----------------------- backward Function -------------------------------
    run = function(method_name) { # only 'LRP' or 'DeepLift'

      # Declare vector for relevances for each output node
      rel_list <- vector(mode = "list",
                         length = length(self$converter$model$output_nodes))

      message(paste0("Backward pass '", method_name, "':"))
      # Define Progressbar
      pb <- txtProgressBar(min = 0,
                           max = length(self$converter$model$graph),
                           style = 3)
      n <- 0
      # We go through the graph in reversed order
      for (step in rev(self$converter$model$graph)) {
        # set the rule name
        rule_name <- self$rule_name

        # Get the current layer
        layer <- self$converter$model$modules_list[[step$used_node]]

        # Get the upper layer relevance ...
        # ... for an output layer
        if (step$used_node %in% self$converter$model$output_nodes) {
          # check if current node is required in 'self$output_idx'
          # get index of the current layer in 'self$output_idx'
          idx <- match(step$used_node,
                       self$converter$model$output_nodes)
          # The current node is not required, i.e. we do not need to calculate
          # relevances for this output
          if (is.null(self$output_idx[[idx]])) {
            rel <- NULL
          }
          # Otherwise ...
          else {
            # get the corresponding output depending on the argument 'ignore_last_act'
            if (self$ignore_last_act) {
              out <- layer$preactivation
            } else {
              out <- layer$output

              # For probabilistic output we need to subtract 0.5, such that
              # 0 means no relevance
              if (method_name == "LRP" & layer$activation_name %in%
                  c("softmax", "sigmoid", "logistic")) {
                out <- out - 0.5
              }
            }

            # For DeepLift, we only need ones
            if (method_name == "DeepLift") {
              rel <- torch_diag_embed(torch_ones_like(out))
              # Overwrite rule name
              if (self$ignore_last_act) {
                rule_name <- "ignore_last_act"
              }
            } else {
              rel <- torch_diag_embed(out)
            }

            # Get necessary output nodes and fill up with zeros
            #
            # We flatten the list of outputs and put the corresponding outputs
            # into the last axis of the relevance tensor, e.g. we have
            # output_idx = list(c(1), c(2,4,5)) and the current layer (of shape (10,4))
            # corresponds to the first entry (c(1)), then we concatenate the
            # output of this layer (shape (10,1)) and three times the same
            # tensor with zeros (shape (10,3) )
            tensor_list <- list()
            for (i in seq_along(self$output_idx)) {
              out_idx <- self$output_idx[[i]]
              # if current layer, use the true output/preactivation and only
              # relevant output nodes
              if (i == idx) {
                tensor_list <- append(tensor_list, list(rel[,,out_idx, drop = FALSE]))
              }
              # otherwise, create for each output node a tensor of zeros
              else if (!is.null(out_idx)) {
                dims <- c(rel$shape[-length(rel$shape)], length(out_idx))
                tensor_list <- append(tensor_list, list(torch_zeros(dims)))
              }
            }
            # concatenate all together
            rel <- torch_cat(tensor_list, dim = -1)
          }
        }
        # ... or a normal layer
        else {
          # Get relevant entries from 'rel_list' for the current layer
          rel <- rel_list[seq_len(step$times) + min(step$used_idx) - 1]
          if (step$times == 1) {
            rel <- rel[[1]]
          } else {
            # If more than one output for this layer was created, we sum up
            # all relevances from the corresponding upper nodes
            result <- 0
            for (res in rel) {
              if (!is.null(res)) {
                result <- result + res
              }
            }
            rel <- result
          }
        }

        # Remove the used relevances from 'rel_list'
        rel_list <- rel_list[-(seq_len(step$times) + min(step$used_idx) - 1)]

        # Apply the LRP method for the current layer and reset the layer
        # afterwards
        if (!is.null(rel)) {
          if (method_name == "LRP") {
            rel <- layer$get_input_relevances(rel, rule_name = self$rule_name,
                                              rule_param = self$rule_param)
          } else if (method_name == "DeepLift") {
            rel <- layer$get_input_multiplier(rel, rule_name = rule_name)
          }
        }
        layer$reset()

        # Transform it back to a list
        if (!is.list(rel)) {
          rel <- list(rel)
        }

        # Save the lower-layer relevances in the list 'rel_list' in the
        # required order
        order <- order(step$used_idx)
        ordered_idx <- step$used_idx[order]
        rel_ordered <- rel[order]
        for (i in seq_along(step$used_idx)) {
          rel_list <- append(rel_list, rel_ordered[i], after = ordered_idx[i] - 1)
        }

        # Update progress bar
        n <- n + 1
        setTxtProgressBar(pb, n)
      }
      close(pb)

      # If necessary, move channels last
      if (self$channels_first == FALSE) {
        rel_list <- lapply(rel_list, function(x) torch_movedim(x, source = 2, destination = -2))
      }

      # As mentioned above, the results of the individual output nodes are
      # stored in the last dimension of the results for each input. Hence,
      # we need to transform it back to the structure: outer list (model output)
      # and inner list (model input)
      result <- list()
      sum_nodes <- 0
      for (i in seq_along(self$output_idx)) {
        if (!is.null(self$output_idx[[i]])) {
          res_output_i <- lapply(rel_list, torch_index_select, dim = -1,
                                 index = as.integer(seq_len(length(self$output_idx[[i]])) + sum_nodes))
          result <- append(result, list(res_output_i))

          sum_nodes <- sum_nodes + length(self$output_idx[[i]])
        }
      }

      # For the DeepLift method, we only get the multiplier. Hence, we have
      # to multiply this by the differences of inputs
      if (method_name == "DeepLift") {
        fun <- function(result, out_idx, in_idx, x, x_ref) {
          res <- result[[out_idx]][[in_idx]]
          if (is.null(res)) {
            res <- NULL
          } else {
            res <- res * (x[[in_idx]] - x_ref[[in_idx]])$unsqueeze(-1)
          }
        }
        result <- apply_results(result, fun, x = self$data, x_ref = self$x_ref)
      }

      result
    },

    # ----------------------- Test data ----------------------------------

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
      assertIntegerish(data_idx, lower = 1, upper = dim(self$data[[1]])[1])
      output_idx <- check_output_idx_for_plot(output_idx, self$output_idx)
      assertLogical(as_plotly)

      # Set aggregation function for channels
      aggr_channels <- get_aggr_function(aggr_channels)

      # Get only relevant model outputs
      null_idx <- unlist(lapply(output_idx, is.null))
      result <- self$result[!null_idx]

      # Get the relevant output and class node indices
      # This is done by matching the given output indices ('output_idx') with
      # the calculated output indices ('self$output_idx'). Afterwards,
      # all non-relevant output indices are removed
      idx_matches <- lapply(seq_along(output_idx), function(i)
        match(output_idx[[i]], self$output_idx[[i]]))[!null_idx]

      result <- apply_results(result, aggregate_channels, idx_matches, data_idx,
                              self$channels_first, aggr_channels)

      # Get and modify input names
      input_names <- lapply(self$converter$input_names, function(in_name) {
        if (length(in_name) > 1) {
          in_name[[1]] <- "aggregated"
        }
        in_name
      })

      result_df <- create_dataframe_from_result(
        data_idx, result, input_names, self$converter$output_names, output_idx)

      # Get plot
      p <- plot_func(result_df, value_name, as_plotly)

      p
    },

    # ------------------------ Boxplots -------------------------------------

    boxplot = function(output_idx, data_idx, ref_data_idx, aggr_channels,
                       preprocess_FUN, as_plotly, individual_data_idx,
                       individual_max, value_name) {

      #
      # Do checks
      #

      # output_idx
      output_idx <- check_output_idx_for_plot(output_idx, self$output_idx)
      # data_idx
      num_data <- dim(self$data[[1]])[1]
      if (identical(data_idx, "all")) {
        data_idx <- seq_len(num_data)
      }
      assertIntegerish(data_idx, lower = 1,
                       upper = num_data,
                       any.missing = FALSE)
      # ref_data_idx
      assertInt(ref_data_idx, lower = 1, upper = num_data, null.ok = TRUE)
      # aggr_channels
      aggr_channels <- get_aggr_function(aggr_channels)
      # preprocess_FUN
      assertFunction(preprocess_FUN)
      # as_plotly
      assertLogical(as_plotly)
      # individual_data_idx
      assertIntegerish(individual_data_idx, lower = 1, upper = num_data, null.ok = TRUE,
                       any.missing = FALSE)
      # individual_max
      assertInt(individual_max, lower = 1)
      individual_max <- min(individual_max, num_data)

      # Set the individual instances for the plot
      if (!as_plotly) {
        individual_idx <- ref_data_idx
      } else {
        individual_idx <- unique(
          c(ref_data_idx, individual_data_idx[seq_len(individual_max)]))
        individual_idx <- individual_idx[!is.na(individual_idx)]
      }


      # Get only relevant model outputs
      null_idx <- unlist(lapply(output_idx, is.null))
      result <- self$result[!null_idx]

      # Get the relevant output and class node indices
      # This is done by matching the given output indices ('output_idx') with
      # the calculated output indices ('self$output_idx'). Afterwards,
      # all non-relevant output indices are removed
      idx_matches <- lapply(seq_along(output_idx), function(i)
        match(output_idx[[i]], self$output_idx[[i]]))[!null_idx]

      # apply preprocess function
      preprocess <- function(result, out_idx, in_idx, idx_matches) {
        res <- result[[out_idx]][[in_idx]]
        if (is.null(res)) {
          res <- NULL
        } else {
          res <- preprocess_FUN(res)
        }

        res
      }
      result <- apply_results(result, preprocess, idx_matches)

      # Get and modify input names
      input_names <- lapply(self$converter$input_names, function(in_name) {
        if (length(in_name) > 1) {
          in_name[[1]] <- "aggregated"
        }
        in_name
      })

      # Create boxplot data
      boxplot_data_aggr <-
        apply_results(result, aggregate_channels, idx_matches, data_idx,
                      self$channels_first, aggr_channels)
      df_boxplot_aggr <- create_dataframe_from_result(
        data_idx, boxplot_data_aggr, input_names, self$converter$output_names, output_idx)

      # Get data for individuals
      individual_data_aggr <-
        apply_results(result, aggregate_channels, idx_matches, individual_idx,
                      self$channels_first, aggr_channels)
      df_individual_aggr <- create_dataframe_from_result(
        individual_idx, individual_data_aggr, input_names, self$converter$output_names, output_idx)

      p <- boxplot_func(df_boxplot_aggr, df_individual_aggr, value_name, as_plotly)

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

  if (length(data_idx) == 0) {
    result_df <- NULL
  } else {
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
    result_df <- result_df[, c(1, 8, 9, 3, 4, 2, 5, 7, 6)]
  }

  result_df
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


check_output_idx_for_plot <- function(output_idx, true_output_idx) {
  if (is.null(output_idx)) {
    # Find first non-NULL value
    output_idx <- rep(list(NULL), length(true_output_idx))
    idx <- which(unlist(lapply(true_output_idx, is.null)) == FALSE)[1]
    output_idx[[idx]] <- true_output_idx[[idx]][1]
  } else if (testIntegerish(output_idx)) {
    assertSubset(output_idx, true_output_idx[[1]])
    output_idx <- list(output_idx)
  } else if (testList(output_idx, max.len = length(true_output_idx))) {
    for (out_idx in seq_along(true_output_idx)) {
      assertSubset(output_idx[[out_idx]], true_output_idx[[out_idx]],
                   .var.name = paste0("output_idx[[", out_idx, "]]"))
    }
  } else {
    values <- unlist(lapply(true_output_idx, paste, collapse = ","))
    values <- paste0("[[", seq_along(values), "]] ", values, " ")
    stop("The argument 'output_idx' has to be either a vector with value of '",
         paste(true_output_idx[[1]], collapse = ","),
         "' or a list of length '", length(true_output_idx),
         "' with values of '", values, "'. Only for these output nodes ",
         "the method has been applied!")
  }

  # Fill up with NULLs
  if (length(output_idx) < length(true_output_idx)) {
    output_idx <-
      append(output_idx,
             rep(list(NULL), length(true_output_idx) - length(output_idx)))
  }

  output_idx
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

get_aggr_function <- function(aggr_channels) {
  assert(
    checkFunction(aggr_channels),
    checkChoice(aggr_channels, c("norm", "sum", "mean"))
  )

  if (!is.function(aggr_channels)) {
    if (aggr_channels == "norm") {
      aggr_channels <- function(x) sum(x^2)^0.5
    } else if (aggr_channels == "sum") {
      aggr_channels <- sum
    } else if (aggr_channels == "mean") {
      aggr_channels <- mean
    }
  }

  aggr_channels
}

# Define function for aggregating the channels
aggregate_channels <- function(result, out_idx, in_idx, idx_matches, data_idx,
                               channels_first, aggr_channels) {
  res <- result[[out_idx]][[in_idx]]
  if (is.null(res)) {
    res <- NULL
  } else {
    d <- length(dim(res))
    idx <- idx_matches[[out_idx]]

    # Select only relevant data and output class
    res <- res$index_select(1, as.integer(data_idx))
    res <- res$index_select(-1, as.integer(idx))
    res <- as_array(res)

    # Only aggregate if the input is non-tabular
    if (d != 3) {
      # get arguments for aggregating
      num_axis <- length(dim(res))
      channel_axis <- ifelse(channels_first, 2, num_axis - 1)
      aggr_axis <- setdiff(seq_len(num_axis), channel_axis)

      # aggregate channels
      res <- apply(res, aggr_axis, aggr_channels)
      dim(res) <- append(dim(res), 1, channel_axis)
    }
  }

  res
}

