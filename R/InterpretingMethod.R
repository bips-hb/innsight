
###############################################################################
#                         Interpreting Method
###############################################################################

#' @title Super class for interpreting methods
#' @description This is a super class for all interpreting methods in the
#' `innsight` package. Implemented are the following methods:
#'
#' - *Deep Learning Important Features* ([`DeepLift`])
#' - *Layer-wise Relevance Propagation* ([`LRP`])
#' - Gradient-based methods:
#'    - *Vanilla gradients* including *Gradient\eqn{\times}Input* ([`Gradient`])
#'    - Smoothed gradients including *SmoothGrad\eqn{\times}Input* ([`SmoothGrad`])
#' - *Connection Weights* (global and local) ([`ConnectionWeights`])
#'
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-ignore_last_act
#' @template param-dtype
#' @template param-aggr_channels
#' @template param-as_plotly
#' @template param-verbose
#' @template param-ref_data_idx
#' @template param-preprocess_FUN
#' @template param-individual_data_idx
#' @template param-individual_max
#' @template param-winner_takes_all
#' @template field-data
#' @template field-converter
#' @template field-channels_first
#' @template field-dtype
#' @template field-ignore_last_act
#' @template field-result
#' @template field-output_idx
#' @template field-verbose
#' @template field-winner_takes_all
#'
InterpretingMethod <- R6Class(
  classname = "InterpretingMethod",
  public = list(
    data = NULL,
    converter = NULL,
    channels_first = NULL,
    dtype = NULL,
    winner_takes_all = TRUE,
    ignore_last_act = NULL,
    result = NULL,
    output_idx = NULL,
    verbose = NULL,

    #' @description
    #' Create a new instance of this super class.
    #'
    #' @param output_idx (`integer`, `list` or `NULL`)\cr
    #' These indices specify the output nodes for which the method is to be
    #' applied. In order to allow models with multiple output layers, there are
    #' the following possibilities to select the indices of the output
    #' nodes in the individual output layers:
    #' \itemize{
    #'   \item An `integer` vector of indices: If the model has only one output
    #'   layer, the values correspond to the indices of the output nodes, e.g.
    #'   `c(1,3,4)` for the first, third and fourth output node. If there are
    #'   multiple output layers, the indices of the output nodes from the first
    #'   output layer are considered.
    #'   \item A `list` of `integer` vectors of indices: If the method is to be
    #'   applied to output nodes from different layers, a list can be passed
    #'   that specifies the desired indices of the output nodes for each
    #'   output layer. Unwanted output layers have the entry `NULL` instead
    #'   of a vector of indices, e.g. `list(NULL, c(1,3))` for the first and
    #'   third output node in the second output layer.
    #'   \item `NULL` (default): The method is applied to all output nodes
    #'   in the first output layer but is limited to the first ten as the
    #'   calculations become more computationally expensive for more
    #'   output nodes.\cr
    #' }
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          winner_takes_all = TRUE,
                          verbose = interactive(),
                          dtype = "float") {
      cli_check(checkClass(converter, "Converter"), "converter")
      self$converter <- converter

      cli_check(checkLogical(channels_first), "channels_first")
      self$channels_first <- channels_first

      cli_check(checkLogical(ignore_last_act), "ignore_last_act")
      self$ignore_last_act <- ignore_last_act

      cli_check(checkLogical(winner_takes_all), "winner_takes_all")
      self$winner_takes_all <- winner_takes_all

      cli_check(checkLogical(verbose), "verbose")
      self$verbose <- verbose

      cli_check(checkChoice(dtype, c("float", "double")), "dtype")
      self$dtype <- dtype
      self$converter$model$set_dtype(dtype)

      # Check output indices
      self$output_idx <- check_output_idx(output_idx, converter$output_dim)

      self$data <- private$test_data(data)
    },

    #' @description
    #' This function returns the result of this method for the given data
    #' either as an array (`'array'`), a torch tensor (`'torch.tensor'`,
    #' or `'torch_tensor'`) of size *(batch_size, dim_in, dim_out)* or as a
    #' data.frame (`'data.frame'`). This method is also implemented as a
    #' generic S3 function [`get_result`]. For a detailed description, we refer
    #' to our in-depth vignette (`vignette("detailed_overview", package = "innsight")`)
    #' or our [website](https://bips-hb.github.io/innsight/articles/detailed_overview.html#get-results).
    #'
    #' @param type (`character(1)`)\cr
    #' The data type of the result. Use one of `'array'`,
    #' `'torch.tensor'`, `'torch_tensor'` or `'data.frame'`
    #' (default: `'array'`).\cr
    #'
    #' @return The result of this method for the given data in the chosen
    #' type.
    get_result = function(type = "array") {
      cli_check(checkChoice(type, c("array", "data.frame", "torch.tensor",
                           "torch_tensor")), "type")

      # Get the result as an array
      if (type == "array") {
        # Get the input names and move the channel dimension (if necessary)
        input_names <- self$converter$input_names
        if (!self$channels_first) {
          input_names <- move_channels_last(input_names)
        }
        # Convert the torch_tensor result into a named array
        result <- tensor_list_to_named_array(
          self$result, input_names, self$converter$output_names,
          self$output_idx)
      } else if (type == "data.frame") {
        # Get the result as a data.frame
        # The function 'create_dataframe_from_result' assumes the channels
        # first format
        result <- self$result
        if (self$channels_first == FALSE) {
          FUN <- function(result, out_idx, in_idx) {
            res <- result[[out_idx]][[in_idx]]
            if (res$dim() > 1) {
              res <- torch_movedim(res, source = -2, destination = 2)
            }

            res
          }
          result <- apply_results(result, FUN)
        }

        # Convert the torch_tensor result into a data.frame
        result <- create_dataframe_from_result(
          seq_len(dim(self$data[[1]])[1]), result,
          self$converter$input_names, self$converter$output_names,
          self$output_idx)
        # Remove unnecessary columns
        if (all(result$input_dimension <= 2)) {
          result$feature_2 <- NULL
        }
        if (all(result$input_dimension <= 1)) {
          result$channel <- NULL
        }
      } else {
        # Get the result as a torch_tensor and remove unnecessary axis
        num_inputs <- length(self$converter$input_names)
        out_null_idx <- unlist(lapply(self$output_idx, is.null))
        out_nonnull_idx <-
          seq_along(self$converter$output_names)[!out_null_idx]
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
    },

    #' @description
    #' This method visualizes the result of the selected
    #' method and enables a visual in-depth investigation with the help
    #' of the S4 classes [`innsight_ggplot2`] and [`innsight_plotly`].\cr
    #' You can use the argument `data_idx` to select the data points in the
    #' given data for the plot. In addition, the individual output nodes for
    #' the plot can be selected with the argument `output_idx`. The different
    #' results for the selected data points and outputs are visualized using
    #' the ggplot2-based S4 class `innsight_ggplot2`. You can also use the
    #' `as_plotly` argument to generate an interactive plot with
    #' `innsight_plotly` based on the plot function [plotly::plot_ly]. For
    #' more information and the whole bunch of possibilities,
    #' see [`innsight_ggplot2`] and [`innsight_plotly`].\cr
    #' \cr
    #' **Notes:**
    #' 1. For the interactive plotly-based plots, the suggested package
    #' `plotly` is required.
    #' 2. The ggplot2-based plots for models with multiple input layers are
    #' a bit more complex, therefore the suggested packages `'grid'`,
    #' `'gridExtra'` and `'gtable'` must be installed in your R session.
    #' 3. If the global *Connection Weights* method was applied, the
    #' unnecessary argument `data_idx` will be ignored.
    #'
    #' @param data_idx (`integer`)\cr
    #' An integer vector containing the numbers of the data
    #' points whose result is to be plotted, e.g., `c(1,3)` for the first
    #' and third data point in the given data. Default: `1`. This argument
    #' will be ignored for the global *Connection Weights* method.\cr
    #' @param output_idx (`integer`, `list` or `NULL`)\cr
    #' The indices of the output nodes for which the results
    #' is to be plotted. This can be either a `integer` vector of indices or a
    #' `list` of `integer` vectors of indices but must be a subset of the indices for
    #' which the results were calculated, i.e., a subset of `output_idx` from the
    #' initialization `new()` (see argument `output_idx` in method `new()` of
    #' this R6 class for details). By default (`NULL`), the smallest index
    #' of all calculated output nodes and output layers is used.\cr
    #' @param same_scale (`logical`)\cr
    #' A logical value that specifies whether the individual plots have the
    #' same fill scale across multiple input layers or whether each is
    #' scaled individually. This argument is only used if more than one input
    #' layer results are plotted.\cr
    #'
    #' @return
    #' Returns either an [`innsight_ggplot2`] (`as_plotly = FALSE`) or an
    #' [`innsight_plotly`] (`as_plotly = TRUE`) object with the plotted
    #' individual results.
    #'
    plot = function(data_idx = 1,
                    output_idx = NULL,
                    aggr_channels = "sum",
                    as_plotly = FALSE,
                    same_scale = FALSE) {

      if (inherits(self, "ConnectionWeights")) {
        if (!self$times_input) {
          if (!identical(data_idx, 1)) {
            messagef(
              "Without the 'times_input' argument, the method ",
              "'ConnectionWeights' is a global method, therefore no individual",
              " data instances can be plotted. But you passed the argument ",
              "'data_idx': 'c(", paste(data_idx, collapse = ", "), ")'!",
              "\nThe argument 'data_idx' will be ignored in the following!"
            )
          }
          data_idx <- 1
          self$data <- list(array(0, dim = c(1, 1)))
          include_data <- TRUE
        } else {
          include_data <- FALSE
        }
        value_name <- "Relative Importance"
      } else if (inherits(self, "LRP")) {
        value_name <- "Relevance"
        include_data <- TRUE
      } else if (inherits(self, "DeepLift")) {
        value_name <- "Contribution"
        include_data <- TRUE
      } else if (inherits(self, "GradientBased")) {
        value_name <- "Gradient"
        include_data <- TRUE
      }

      # Check correctness of arguments
      cli_check(
        checkIntegerish(data_idx, lower = 1, upper = dim(self$data[[1]])[1]),
        "data_idx")
      output_idx <- check_output_idx_for_plot(output_idx, self$output_idx)
      cli_check(checkLogical(as_plotly), "as_plotly")
      cli_check(checkLogical(same_scale), "same_scale")

      # Set aggregation function for channels
      aggr_channels <- get_aggr_function(aggr_channels)

      # Get only relevant model outputs
      null_idx <- unlist(lapply(output_idx, is.null))
      result <- self$result[!null_idx]

      # Get the relevant output and class node indices
      # This is done by matching the given output indices ('output_idx') with
      # the calculated output indices ('self$output_idx'). Afterwards,
      # all non-relevant output indices are removed
      idx_matches <- lapply(
        seq_along(output_idx),
        function(i) match(output_idx[[i]], self$output_idx[[i]]))[!null_idx]

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
      if (as_plotly) {
        p <- create_plotly(result_df, value_name, include_data, FALSE, NULL,
                           same_scale)
      } else {
        p <- create_ggplot(result_df, value_name, include_data, FALSE, NULL,
                           same_scale)
      }

      p
    },

    #' @description
    #' This method visualizes the results of the selected method summarized as
    #' boxplots and enables a visual in-depth investigation of the global
    #' behavior with the help of the S4 classes [`innsight_ggplot2`] and
    #' [`innsight_plotly`].\cr
    #' You can use the argument `output_idx` to select the individual output
    #' nodes for the plot. For tabular and 1D data, boxplots are created in
    #' which a reference value can be selected from the data using the
    #' `ref_data_idx` argument. For images, only the pixel-wise median is
    #' visualized due to the complexity. The plot is generated using the
    #' ggplot2-based S4 class `innsight_ggplot2`. You can also use the
    #' `as_plotly` argument to generate an interactive plot with
    #' `innsight_plotly` based on the plot function [plotly::plot_ly]. For
    #' more information and the whole bunch of possibilities, see
    #' [`innsight_ggplot2`] and [`innsight_plotly`].\cr \cr
    #' **Notes:**
    #' 1. This method can only be used for the local *Connection Weights*
    #' method, i.e., if `times_input` is `TRUE` and `data` is provided.
    #' 2. For the interactive plotly-based plots, the suggested package
    #' `plotly` is required.
    #' 3. The ggplot2-based plots for models with multiple input layers are
    #' a bit more complex, therefore the suggested packages `'grid'`,
    #' `'gridExtra'` and `'gtable'` must be installed in your R session.
    #'
    #' @param output_idx (`integer`, `list` or `NULL`)\cr
    #' The indices of the output nodes for which the
    #' results is to be plotted. This can be either a `vector` of indices or
    #' a `list` of vectors of indices but must be a subset of the indices for
    #' which the results were calculated, i.e., a subset of `output_idx` from
    #' the initialization `new()` (see argument `output_idx` in method `new()`
    #' of this R6 class for details). By default (`NULL`), the smallest index
    #' of all calculated output nodes and output layers is used.\cr
    #' @param data_idx (`integer`)\cr
    #' By default, all available data points are used
    #' to calculate the boxplot information. However, this parameter can be
    #' used to select a subset of them by passing the indices. For example, with
    #' `c(1:10, 25, 26)` only the first 10 data points and
    #' the 25th and 26th are used to calculate the boxplots.\cr
    #'
    #' @return
    #' Returns either an [`innsight_ggplot2`] (`as_plotly = FALSE`) or an
    #' [`innsight_plotly`] (`as_plotly = TRUE`) object with the plotted
    #' summarized results.
    boxplot = function(output_idx = NULL,
                       data_idx = "all",
                       ref_data_idx = NULL,
                       aggr_channels = "sum",
                       preprocess_FUN = abs,
                       as_plotly = FALSE,
                       individual_data_idx = NULL,
                       individual_max = 20) {

      if (inherits(self, "ConnectionWeights")) {
        if (!self$times_input) {
          stopf(
            "Only if the result of the {.emph ConnectionWeights} method is ",
            "multiplied by the data ({.arg times_input} = TRUE), it is a local ",
            "method and only then boxplots can be generated over multiple ",
            "instances. Thus, the argument {.arg data} must be specified and ",
            "{.arg times_input} = TRUE when applying the ",
            "{.code ConnectionWeights$new} method.")
        }
        value_name <- "Relative Importance"
      } else if (inherits(self, "LRP")) {
        value_name <- "Relevance"
      } else if (inherits(self, "DeepLift")) {
        value_name <- "Contribution"
      } else if (inherits(self, "GradientBased")) {
        value_name <- "Gradient"
      }

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
      cli_check(checkIntegerish(data_idx, lower = 1, upper = num_data,
                                any.missing = FALSE), "data_idx")
      # ref_data_idx
      cli_check(
        checkInt(ref_data_idx, lower = 1, upper = num_data, null.ok = TRUE),
        "ref_data_idx")
      # aggr_channels
      aggr_channels <- get_aggr_function(aggr_channels)
      # preprocess_FUN
      cli_check(checkFunction(preprocess_FUN), "preprocess_FUN")
      # as_plotly
      cli_check(checkLogical(as_plotly), "as_plotly")
      # individual_data_idx
      cli_check(
        checkIntegerish(individual_data_idx, lower = 1, upper = num_data,
                        null.ok = TRUE, any.missing = FALSE),
        "individual_data_idx")
      if (is.null(individual_data_idx)) individual_data_idx <- seq_len(num_data)
      # individual_max
      cli_check(checkInt(individual_max, lower = 1), "individual_max")
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
      idx_matches <- lapply(
        seq_along(output_idx),
        function(i) match(output_idx[[i]], self$output_idx[[i]]))[!null_idx]

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

      idx <- sort(unique(c(individual_idx, data_idx)))
      # Create boxplot data
      result <-
        apply_results(result, aggregate_channels, idx_matches, idx,
                      self$channels_first, aggr_channels)
      result_df <- create_dataframe_from_result(
        idx, result, input_names, self$converter$output_names, output_idx)

      idx <- as.numeric(gsub("data_", "", as.character(result_df$data)))
      result_df$boxplot_data <- ifelse(idx %in% data_idx, TRUE, FALSE)
      result_df$individual_data <- ifelse(idx %in% individual_idx, TRUE, FALSE)

      # Get plot
      if (as_plotly) {
        p <- create_plotly(result_df, value_name, FALSE, TRUE, ref_data_idx,
                           TRUE)
      } else {
        p <- create_ggplot(result_df, value_name, FALSE, TRUE, ref_data_idx,
                           TRUE)
      }

      p
    },

    #' @description
    #' Print a summary of the method object. This summary contains the
    #' individual fields and in particular the results of the applied method.
    #'
    #' @return Returns the method object invisibly via [`base::invisible`].
    #'
    print = function() {
      cli_h1(paste0("Method {.emph ", class(self)[1], "} ({.pkg innsight})"))
      cat("\n")

      cli_div(theme = list(ul = list(`margin-left` = 2, before = ""),
                           dl = list(`margin-left` = 2, before = "")))
      cli_text("{.strong Fields} (method-specific):")
      private$print_method_specific()
      cat("\n")

      cli_text("{.strong Fields} (other):")
      i <- cli_ul()
      print_output_idx(self$output_idx, self$converter$output_names)
      cli_li(paste0("{.field ignore_last_act}:  ", self$ignore_last_act))
      cli_li(paste0("{.field channels_first}:  ", self$channels_first))
      cli_li(paste0("{.field dtype}:  '", self$dtype, "'"))
      cli_end(id = i)

      cli_h2("{.strong Result} ({.field result})")
      print_result(self$result)

      cli_h1("")

      invisible(self)
    }
  ),
  private = list(

    # ----------------------- backward Function -------------------------------
    run = function(method_name) { # only 'LRP' or 'DeepLift'

      # Declare vector for relevances for each output node
      rel_list <- vector(mode = "list",
                         length = length(self$converter$model$output_nodes))

      if (self$verbose) {
        #messagef("Backward pass '", method_name, "':")
        # Define Progressbar
        cli_progress_bar(name = paste0("Backward pass '", method_name, "'"),
                         total = length(self$converter$model$graph),
                         type = "iterator", clear = FALSE)
      }

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
          } else {
            # Otherwise ...
            # get the corresponding output depending on the argument
            # 'ignore_last_act'
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
            } else if (method_name == "Connection-Weights") {
              if (self$dtype == "float") {
                rel <- torch_diag_embed(torch_ones(c(1, layer$output_dim)))
              } else {
                rel <- torch_diag_embed(
                  torch_ones(c(1, layer$output_dim), dtype = torch_double()))
              }

            } else {
              rel <- torch_diag_embed(out)
            }

            # Get necessary output nodes and fill up with zeros
            #
            # We flatten the list of outputs and put the corresponding outputs
            # into the last axis of the relevance tensor, e.g. we have
            # output_idx = list(c(1), c(2,4,5)) and the current layer
            # (of shape (10,4)) corresponds to the first entry (c(1)), then
            # we concatenate the output of this layer (shape (10,1)) and
            # three times the same tensor with zeros (shape (10,3) )
            tensor_list <- list()
            for (i in seq_along(self$output_idx)) {
              out_idx <- self$output_idx[[i]]
              # if current layer, use the true output/preactivation and only
              # relevant output nodes
              if (i == idx) {
                tensor_list <-
                  append(tensor_list, list(rel[, , out_idx, drop = FALSE]))
              } else if (!is.null(out_idx)) {
                # otherwise, create for each output node a tensor of zeros
                dims <- c(rel$shape[-length(rel$shape)], length(out_idx))
                tensor_list <- append(tensor_list, list(torch_zeros(dims)))
              }
            }
            # concatenate all together
            rel <- torch_cat(tensor_list, dim = -1)
          }
        } else {
          # ... or a normal layer
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
            lrp_rule <-
              get_lrp_rule(self$rule_name, self$rule_param, class(layer)[1])
            rel <- layer$get_input_relevances(rel, rule_name = lrp_rule$rule_name,
                                              rule_param = lrp_rule$rule_param,
                                              winner_takes_all = self$winner_takes_all)
          } else if (method_name == "DeepLift") {
            rel <- layer$get_input_multiplier(rel, rule_name = rule_name,
                                              winner_takes_all = self$winner_takes_all)
          } else if (method_name == "Connection-Weights") {
            rel <- layer$get_gradient(rel, weight = layer$W,
                                      use_avgpool = !self$winner_takes_all)
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
          rel_list <-
            append(rel_list, rel_ordered[i], after = ordered_idx[i] - 1)
        }

        if (self$verbose) {
          # Update progress bar
          cli_progress_update(force = TRUE)
        }
      }

      if (self$verbose) cli_progress_done()

      # If necessary, move channels last
      if (self$channels_first == FALSE) {
        rel_list <- lapply(
          rel_list,
          function(x) torch_movedim(x, source = 2, destination = -2))
      }

      # As mentioned above, the results of the individual output nodes are
      # stored in the last dimension of the results for each input. Hence,
      # we need to transform it back to the structure: outer list (model output)
      # and inner list (model input)
      result <- list()
      sum_nodes <- 0
      for (i in seq_along(self$output_idx)) {
        if (!is.null(self$output_idx[[i]])) {
          index <- seq_len(length(self$output_idx[[i]])) + sum_nodes
          res_output_i <- lapply(rel_list, torch_index_select, dim = -1,
                                 index = as.integer(index))
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
        stopf("Argument {.arg data} is missing!")
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
          stopf("Failed to convert the argument {.arg ", name,
               "[[", i, "]]} to an array ",
               "using the function {.fn base::as.array}. The class of your ",
               "argument {.arg ", name, "[[", i, "]]}: '",
               paste(class(input_data), collapse = "', '"), "'")
        })

        ordered_dim <- self$converter$input_dim[[i]]
        if (!self$channels_first) {
          channels <- ordered_dim[1]
          ordered_dim <- c(ordered_dim[-1], channels)
        }

        if (length(dim(input_data)[-1]) != length(ordered_dim) ||
            !all(dim(input_data)[-1] == ordered_dim)) {
          stopf(
            "Unmatch in model input dimension (*, ",
            paste0(ordered_dim, collapse = ", "), ") and dimension of ",
            "argument {.arg ", name, "[[", i, "]]} (",
            paste0(dim(input_data), collapse = ", "),
            "). Try to change the argument {.arg channels_first}, if only ",
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

    print_method_specific = function() {
      NULL
    }
  )
)


#' Get the result of an interpretation method
#'
#' This is a generic S3 method for the R6 method
#' `InterpretingMethod$get_result()`. See the respective method described in
#' [`InterpretingMethod`] for details.
#'
#' @param x An object of the class [`InterpretingMethod`] including the
#' subclasses [`Gradient`], [`SmoothGrad`], [`LRP`], [`DeepLift`] and
#' [`ConnectionWeights`].
#' @param ... Other arguments specified in the R6 method
#' `InterpretingMethod$get_result()`. See [`InterpretingMethod`] for details.
#'
#' @export
get_result <- function(x, ...) UseMethod("get_result", x)

#' @exportS3Method
get_result.InterpretingMethod <- function(x, ...) {
  x$get_result(...)
}

###############################################################################
#                           print utility functions
###############################################################################
print_result <- function(result) {
  num_outlayers <- length(result)
  num_inlayers <- length(result[[1]])

  for (i in seq_along(result)) {
    if (num_outlayers > 1) cli_text(paste0("{.strong Output layer ", i, ":}"))
    for (j in seq_along(result[[i]])) {
      if (num_inlayers > 1) {
        in_l <- cli_ul()
        cli_li(paste0("Input layer ", j, ":"))
      }
      if (is.null(result[[i]][[j]])) {
        items <- paste0(col_cyan(symbol$i), " {.emph (not connected to output layer ", i, ")}")
        cli_bullets(c(" " = items))
      } else {
        items <- list(
          paste0("(", paste0(result[[i]][[j]]$shape, collapse = ", "), ")"),
          paste0(paste0(c("min: ", "median: ", "max: "),
                        signif(as_array(result[[i]][[j]]$quantile(c(0,0.5,1))))),
                 collapse = ", "),
          as_array(result[[i]][[j]]$isnan()$sum())
        )

        names(items) <- paste0(symbol$line,
                               c(" Shape", " Range", " Number of NaN values"))
        cli_dl(items)
      }
      if (num_inlayers > 1) cli_end(in_l)
    }
  }
}

print_output_idx <- function(output_idx, out_names) {
  draw_layer <- if (length(output_idx) > 1) TRUE else FALSE

  if (draw_layer) {
    cli_li("{.field output_idx}:")
    layer_list <- cli_ul()
  }

  for (i in seq_along(output_idx)) {
    if (draw_layer) {
      prefix <- paste0("Output layer ", i, ": {.emph ")
    } else {
      prefix <- "{.field output_idx}: {.emph "
    }

    if (is.null(output_idx[[i]])) {
      output_idx[[i]] <- "not applied!"
      labels <- ""
    } else {
      labels <- paste0(
        " (", symbol$arrow_right, " corresponding labels: {.emph '",
        paste0(out_names[[i]][[1]][output_idx[[i]]], collapse = "'}, {.emph '"),
        "'})")
    }

    cli_li(paste0(
      prefix,
      paste0(output_idx[[i]], collapse = "}, {.emph "), "}", labels))
  }
}


###############################################################################
#                                 Utils
###############################################################################

check_output_idx <- function(output_idx, output_dim) {
  # for the default value, choose from the first output the first ten
  # (maybe less) output nodes
  if (is.null(output_idx)) {
    output_idx <- list(1:min(10, output_dim[[1]]))
  } else if (testIntegerish(output_idx,
                          lower = 1,
                          upper = output_dim[[1]])) {
    # or only a number (assumes the first output)
    output_idx <- list(output_idx)
  } else if (testList(output_idx, max.len = length(output_dim))) {
    # the argument output_idx is a list of output_nodes for each output
    n <- 1
    for (output in output_idx) {
      limit <- output_dim[[n]]
      cli_check(checkInt(limit), "limit")
      if (!testIntegerish(output, lower = 1, upper = limit, null.ok = TRUE)) {
        stopf("Assertion on {.arg output_idx[[", n, "]]} failed: Value(s) ",
             paste(output, collapse = ","), " not <= ", limit, ".")
      }
      n <- n + 1
    }
  } else {
    stopf("The argument {.arg output_idx} has to be either a vector with maximum ",
         "value of '", output_dim[[1]], "' or a list of length '",
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
          } else {
            # otherwise convert the result to an array and set dimnames
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
    output_levels <- paste0("Output_", seq_along(output_names))
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
      result_df$model_output <- factor(
        paste0("Output_", nonnull_idx[out_idx]),
        levels = output_levels)



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
    feature <- input_names[[1]]
    feature_2 <- NaN
    channel <- NaN
  } else if (dimension == 2) {
    feature <- input_names[[2]]
    feature_2 <- NaN
    channel <- input_names[[1]]
  } else {
    feature <- input_names[[2]]
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
    cli_check(checkSubset(output_idx, true_output_idx[[1]]), "output_idx")
    output_idx <- list(output_idx)
  } else if (testList(output_idx, max.len = length(true_output_idx))) {
    for (out_idx in seq_along(true_output_idx)) {
      cli_check(checkSubset(output_idx[[out_idx]], true_output_idx[[out_idx]]),
                paste0("output_idx[[", out_idx, "]]"))
    }
  } else {
    values <- unlist(lapply(true_output_idx, paste, collapse = ","))
    values <- paste0("[[", seq_along(values), "]] ", values, " ")
    stopf("The argument {.arg output_idx} has to be either a vector with value of '",
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
      names[[idx]] <- names[[idx]][c(2, 1)]
    } else if (length(names[[idx]]) == 3) { # 2d input
      names[[idx]] <- names[[idx]][c(2, 3, 1)]
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
  cli_check(c(
    checkFunction(aggr_channels),
    checkChoice(aggr_channels, c("norm", "sum", "mean"))
  ), "aggr_channels")

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

get_lrp_rule <- function(rule_name, rule_param, layer_class) {
  if (is.list(rule_name)) {
    if (layer_class %in% names(rule_name)) {
      rule_name <- rule_name[[layer_class]]
    } else {
      rule_name <- "simple"
    }
  }
  if (is.list(rule_param)) {
    if (layer_class %in% names(rule_param)) {
      rule_param <- rule_param[[layer_class]]
    } else {
      rule_param <- NULL
    }
  }

  list(rule_name = rule_name, rule_param = rule_param)
}
