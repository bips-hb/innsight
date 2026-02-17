###############################################################################
#                         TorchGradientResult Class
###############################################################################

#' @title Result wrapper for direct torch gradient methods
#'
#' @description
#' This class wraps the results from direct torch gradient methods
#' (`torch_grad`, `torch_intgrad`, etc.) and provides the same interface
#' as the converter-based methods, including `plot()`, `get_result()`, and
#' other methods.
#'
#' @details
#' This class is typically created automatically by setting `return_object = TRUE`
#' in the torch gradient functions, but can also be created manually using
#' `TorchGradientResult$new()`.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' model <- nn_sequential(nn_linear(10, 3))
#' data <- torch_randn(5, 10)
#'
#' # Get result as innsight object
#' result <- torch_grad(model, data, return_object = TRUE)
#'
#' # Use standard innsight methods
#' plot(result)
#' summary(result)
#' get_result(result)
#' }
#'
#' @export
TorchGradientResult <- R6Class(
  classname = "TorchGradientResult",
  public = list(

    #' @field result (`torch_tensor`)\cr
    #' The gradient-based attributions as a torch tensor with shape
    #' `(batch_size, features, outputs)`.
    result = NULL,

    #' @field data (`torch_tensor`)\cr
    #' The input data used for calculating gradients.
    data = NULL,

    #' @field model (`nn_module`)\cr
    #' The torch model.
    model = NULL,

    #' @field method_name (`character(1)`)\cr
    #' Name of the method used (e.g., "Gradient", "IntegratedGradient").
    method_name = NULL,

    #' @field output_idx (`integer`)\cr
    #' Indices of the output nodes for which gradients were calculated.
    output_idx = NULL,

    #' @field channels_first (`logical(1)`)\cr
    #' Whether the data uses channels-first format.
    channels_first = TRUE,

    #' @field preds (`torch_tensor`)\cr
    #' Model predictions for the input data.
    preds = NULL,

    #' @field times_input (`logical(1)`)\cr
    #' Whether gradients were multiplied by input (if applicable).
    times_input = NULL,

    #' @field n (`integer(1)`)\cr
    #' Number of interpolation steps or samples (if applicable).
    n = NULL,

    #' @field x_ref (`torch_tensor`)\cr
    #' Reference input for IntegratedGradient (if applicable).
    x_ref = NULL,

    #' @field noise_level (`numeric(1)`)\cr
    #' Noise level for SmoothGrad (if applicable).
    noise_level = NULL,

    #' @field data_ref (`torch_tensor`)\cr
    #' Reference data for ExpectedGradient (if applicable).
    data_ref = NULL,

    #' @description
    #' Create a new TorchGradientResult object.
    #'
    #' @param result (`torch_tensor`)\cr
    #'   The gradient attributions.
    #' @param data (`torch_tensor`)\cr
    #'   The input data.
    #' @param model (`nn_module`)\cr
    #'   The torch model.
    #' @param method_name (`character(1)`)\cr
    #'   Name of the method.
    #' @param output_idx (`integer`)\cr
    #'   Output indices.
    #' @param ... Additional method-specific parameters.
    #'
    #' @return A new TorchGradientResult object.
    initialize = function(result, data, model, method_name, output_idx = NULL, ...) {
      self$result <- result
      self$data <- data
      self$model <- model
      self$method_name <- method_name
      self$output_idx <- output_idx %||% seq_len(result$shape[3])

      # Store method-specific parameters
      dots <- list(...)
      for (name in names(dots)) {
        self[[name]] <- dots[[name]]
      }

      # Calculate predictions if not provided
      if (is.null(self$preds)) {
        self$preds <- model(data)
      }
    },

    #' @description
    #' Get the result in different formats.
    #'
    #' @param type (`character(1)`)\cr
    #'   Output format: "array", "torch_tensor", or "data.frame".
    #'
    #' @return The result in the specified format.
    get_result = function(type = "array") {
      checkmate::assert_choice(type, c("array", "torch_tensor", "torch.tensor", "data.frame"))

      if (type %in% c("torch_tensor", "torch.tensor")) {
        return(self$result)
      } else if (type == "array") {
        return(as.array(self$result))
      } else if (type == "data.frame") {
        # Convert to data.frame format
        arr <- as.array(self$result)
        batch_size <- dim(arr)[1]
        n_features <- dim(arr)[2]
        n_outputs <- dim(arr)[3]

        # Create data.frame
        df <- data.frame(
          data_idx = rep(1:batch_size, each = n_features * n_outputs),
          feature = rep(rep(1:n_features, each = n_outputs), times = batch_size),
          output = rep(1:n_outputs, times = batch_size * n_features),
          value = as.vector(arr)
        )

        return(df)
      }
    },

    #' @description
    #' Plot the results.
    #'
    #' @param data_idx (`integer`)\cr
    #'   Indices of data points to plot (default: 1).
    #' @param output_idx (`integer`)\cr
    #'   Indices of outputs to plot (default: first output).
    #' @param ... Additional arguments passed to plotting function.
    #'
    #' @return A ggplot2 object.
    plot = function(data_idx = 1, output_idx = NULL, ...) {
      if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("Package 'ggplot2' is required for plotting. Please install it.")
      }

      # Default to first output if not specified
      if (is.null(output_idx)) {
        output_idx <- 1
      }

      # Get data for plotting
      arr <- as.array(self$result)

      plots <- list()

      for (d_idx in data_idx) {
        for (o_idx in output_idx) {
          # Extract values for this data point and output
          values <- arr[d_idx, , o_idx]
          features <- seq_along(values)

          # Create data.frame
          plot_data <- data.frame(
            feature = features,
            importance = values
          )

          # Create plot
          p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = feature, y = importance)) +
            ggplot2::geom_col(fill = ifelse(values >= 0, "steelblue", "firebrick")) +
            ggplot2::labs(
              title = sprintf("Sample %d, Output %d", d_idx, o_idx),
              subtitle = sprintf("Method: %s", self$method_name),
              x = "Feature",
              y = "Attribution"
            ) +
            ggplot2::theme_minimal() +
            ggplot2::geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5)

          plots[[length(plots) + 1]] <- p
        }
      }

      # Return single plot or combined plots
      if (length(plots) == 1) {
        return(plots[[1]])
      } else {
        if (!requireNamespace("gridExtra", quietly = TRUE)) {
          warning("Package 'gridExtra' is required for multiple plots. Returning first plot only.")
          return(plots[[1]])
        }
        return(gridExtra::grid.arrange(grobs = plots))
      }
    },

    #' @description
    #' Print summary of the result.
    #'
    #' @return Invisibly returns self.
    print = function() {
      cat("TorchGradientResult\n")
      cat("===================\n\n")
      cat("Method:", self$method_name, "\n")
      cat("Data shape:", paste(self$data$shape, collapse = " x "), "\n")
      cat("Result shape:", paste(self$result$shape, collapse = " x "), "\n")
      cat("Output indices:", paste(self$output_idx, collapse = ", "), "\n")

      # Method-specific info
      if (!is.null(self$times_input)) {
        cat("Times input:", self$times_input, "\n")
      }
      if (!is.null(self$n)) {
        cat("Steps/samples (n):", self$n, "\n")
      }
      if (!is.null(self$noise_level)) {
        cat("Noise level:", self$noise_level, "\n")
      }

      # Summary statistics
      cat("\nSummary statistics:\n")
      result_array <- as.array(self$result)
      cat("  Min:", min(result_array), "\n")
      cat("  Max:", max(result_array), "\n")
      cat("  Mean:", mean(result_array), "\n")
      cat("  Std:", sd(result_array), "\n")

      invisible(self)
    }
  )
)

#' @rdname TorchGradientResult
#' @param x A \code{TorchGradientResult} object.
#' @param ... Additional arguments (currently unused for print, passed to plot method for plot).
#' @method print TorchGradientResult
#' @export
print.TorchGradientResult <- function(x, ...) {
  x$print()
}

#' @rdname TorchGradientResult
#' @method plot TorchGradientResult
#' @export
plot.TorchGradientResult <- function(x, ...) {
  x$plot(...)
}

#' Helper function for NULL default
#' @keywords internal
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
