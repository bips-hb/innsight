###############################################################################
#                    Direct Torch Gradient Methods
###############################################################################
#
# This file implements gradient-based interpretation methods that work
# directly with torch models without requiring conversion through the
# Converter class. This approach is more efficient for torch models as it:
# - Avoids model conversion overhead
# - Uses native torch autograd directly
# - Reduces memory footprint
# - Maintains compatibility with torch's computational graph
#

#' Wrap result in TorchGradientResult object if requested
#' @keywords internal
wrap_torch_result <- function(result, return_object, data, model, method_name,
                               output_idx, preds = NULL, ...) {
  if (!return_object) {
    return(result)
  }

  TorchGradientResult$new(
    result = result,
    data = data,
    model = model,
    method_name = method_name,
    output_idx = output_idx,
    preds = preds,
    ...
  )
}

#' @title Direct Gradient calculation for torch models
#'
#' @description
#' Calculate gradients of model outputs with respect to inputs using native
#' torch autograd. This function provides a lightweight alternative to
#' \code{\link{run_grad}} that works directly with \code{torch::nn_module}
#' objects without requiring model conversion.
#'
#' @param model (\code{\link[torch]{nn_module}})\cr
#'   A torch model. Must be an instance of \code{nn_module}.
#' @param data (\code{\link[torch]{torch_tensor}}, `array`, or `matrix`)\cr
#'   Input data for which to calculate gradients. If not already a torch
#'   tensor, it will be converted automatically. Expected shape:
#'   `(batch_size, ...)` where `...` represents the input dimensions.
#' @param output_idx (`integer`)\cr
#'   Index or indices of output nodes for which to calculate gradients.
#'   If the model outputs a tensor of shape `(batch_size, n_outputs)`,
#'   use indices 1 to n_outputs. Default: `NULL` (all outputs).
#' @param times_input (`logical(1)`)\cr
#'   If `TRUE`, multiplies the gradients by the input values
#'   (Gradient×Input method). Default: `FALSE` (Vanilla Gradient).
#' @param dtype (`character(1)`)\cr
#'   Data type for calculations. Either `"float"` for
#'   \code{\link[torch]{torch_float}} or `"double"` for
#'   \code{\link[torch]{torch_double}}. Default: `"float"`.
#' @param return_object (`logical(1)`)\cr
#'   If `TRUE`, returns a \code{\link{TorchGradientResult}} object with
#'   methods like `plot()` and `get_result()`. If `FALSE` (default), returns
#'   a raw \code{\link[torch]{torch_tensor}}.
#'
#' @return If `return_object = FALSE` (default): A \code{\link[torch]{torch_tensor}}
#'   containing the gradients with shape `(batch_size, ..., n_outputs)`.
#'   If `return_object = TRUE`: A \code{\link{TorchGradientResult}} object.
#'
#' @details
#' This function computes the gradients of the outputs with respect to the
#' input variables, i.e., for all input variable \eqn{i} and output class \eqn{j}:
#' \deqn{d f(x)_j / d x_i}
#'
#' If `times_input = TRUE`, the gradients are multiplied by the respective
#' input value (Gradient×Input):
#' \deqn{x_i * d f(x)_j / d x_i}
#'
#' While vanilla gradients emphasize prediction-sensitive features,
#' Gradient×Input provides a decomposition of the output into feature-wise
#' effects based on the first-order Taylor decomposition.
#'
#' @section Performance:
#' This function is typically faster and more memory-efficient than
#' \code{\link{run_grad}} because it:
#' \itemize{
#'   \item Avoids model conversion overhead
#'   \item Uses native torch autograd directly
#'   \item Does not store intermediate layer activations unless needed
#' }
#'
#' @section Comparison with run_grad:
#' \code{torch_grad} is recommended when:
#' \itemize{
#'   \item Working with torch models exclusively
#'   \item Performance is critical
#'   \item You don't need the full Converter functionality
#' }
#'
#' Use \code{\link{run_grad}} when:
#' \itemize{
#'   \item Working with keras or neuralnet models
#'   \item You need other interpretation methods (LRP, DeepLift, etc.)
#'   \item You want visualization capabilities
#' }
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Create a simple model
#' model <- nn_sequential(
#'   nn_linear(10, 50),
#'   nn_relu(),
#'   nn_linear(50, 3)
#' )
#'
#' # Generate some data
#' data <- torch_randn(5, 10)
#'
#' # Calculate vanilla gradients
#' grads <- torch_grad(model, data)
#'
#' # Calculate Gradient×Input
#' grads_times_input <- torch_grad(model, data, times_input = TRUE)
#'
#' # Calculate gradients for specific output
#' grads_class1 <- torch_grad(model, data, output_idx = 1)
#'
#' # Get result as innsight object with plot() support
#' result <- torch_grad(model, data, return_object = TRUE)
#' plot(result)
#' }
#'
#' @seealso
#' \code{\link{run_grad}}, \code{\link{Gradient}}, \code{\link{TorchGradientResult}}
#'
#' @export
torch_grad <- function(model,
                       data,
                       output_idx = NULL,
                       times_input = FALSE,
                       dtype = "float",
                       return_object = FALSE) {

  # Input validation
  checkmate::assert_class(model, "nn_module")
  checkmate::assert_choice(dtype, c("float", "double"))
  checkmate::assert_logical(times_input, len = 1)
  checkmate::assert_logical(return_object, len = 1)

  # Convert data to torch tensor if needed
  if (!inherits(data, "torch_tensor")) {
    data <- torch::torch_tensor(data)
  }

  # Set dtype
  torch_dtype <- if (dtype == "float") torch::torch_float() else torch::torch_double()
  data <- data$to(dtype = torch_dtype)

  # Convert model to the correct dtype
  model <- model$to(dtype = torch_dtype)

  # Enable gradient computation for input
  data$requires_grad_(TRUE)

  # Forward pass
  output <- model(data)

  # Handle output_idx
  if (is.null(output_idx)) {
    # Use all outputs
    if (output$dim() == 2) {
      output_idx <- seq_len(output$shape[2])
    } else if (output$dim() == 1) {
      output_idx <- 1
    } else {
      stop("Unexpected output dimension. Expected 1D or 2D output tensor.")
    }
  }

  # Validate output_idx
  if (output$dim() == 2) {
    max_idx <- output$shape[2]
  } else {
    max_idx <- 1
  }

  if (any(output_idx > max_idx) || any(output_idx < 1)) {
    stop(paste0("output_idx must be between 1 and ", max_idx))
  }

  # Calculate gradients for each selected output
  gradients_list <- list()

  for (i in seq_along(output_idx)) {
    idx <- output_idx[i]

    # Select output
    if (output$dim() == 2) {
      output_selected <- output[, idx]
    } else {
      output_selected <- output
    }

    # Sum over batch dimension for gradient calculation
    output_sum <- output_selected$sum()

    # Calculate gradient
    grad <- torch::autograd_grad(
      outputs = output_sum,
      inputs = data,
      retain_graph = (i < length(output_idx)),  # Keep graph for all but last
      create_graph = FALSE
    )[[1]]  # autograd_grad returns a list

    gradients_list[[i]] <- grad
  }

  # Stack gradients along last dimension
  if (length(gradients_list) == 1) {
    result <- gradients_list[[1]]$unsqueeze(-1)
  } else {
    result <- torch::torch_stack(gradients_list, dim = -1)
  }

  # Multiply by input if requested
  if (times_input) {
    # Expand data to match gradient dimensions
    data_expanded <- data$unsqueeze(-1)$expand_as(result)
    result <- result * data_expanded
  }

  # Detach from computation graph and disable gradient for input
  result <- result$detach()
  data$requires_grad_(FALSE)

  # Wrap result if requested
  wrap_torch_result(
    result = result,
    return_object = return_object,
    data = data,
    model = model,
    method_name = if (times_input) "Gradient x Input" else "Gradient",
    output_idx = output_idx,
    preds = output,
    times_input = times_input
  )
}


#' @title Direct Integrated Gradients for torch models
#'
#' @description
#' Calculate Integrated Gradients using native torch autograd. This function
#' provides a lightweight alternative to \code{\link{run_intgrad}} that works
#' directly with \code{torch::nn_module} objects.
#'
#' @param model (\code{\link[torch]{nn_module}})\cr
#'   A torch model. Must be an instance of \code{nn_module}.
#' @param data (\code{\link[torch]{torch_tensor}}, `array`, or `matrix`)\cr
#'   Input data for which to calculate gradients.
#' @param x_ref (\code{\link[torch]{torch_tensor}}, `array`, or `matrix`)\cr
#'   Reference input (baseline). If `NULL`, uses zeros. Must have shape
#'   `(1, ...)` where `...` matches input dimensions. Default: `NULL`.
#' @param output_idx (`integer`)\cr
#'   Index or indices of output nodes. Default: `NULL` (all outputs).
#' @param n (`integer(1)`)\cr
#'   Number of steps for approximating the integral. Default: `50`.
#' @param times_input (`logical(1)`)\cr
#'   If `TRUE`, multiplies integrated gradients by `(data - x_ref)`.
#'   This is the standard Integrated Gradients formulation. Default: `TRUE`.
#' @param dtype (`character(1)`)\cr
#'   Data type: `"float"` or `"double"`. Default: `"float"`.
#' @param return_object (`logical(1)`)\cr
#'   If `TRUE`, returns a \code{\link{TorchGradientResult}} object with
#'   methods like `plot()` and `get_result()`. If `FALSE` (default), returns
#'   a raw \code{\link[torch]{torch_tensor}}.
#'
#' @return If `return_object = FALSE` (default): A \code{\link[torch]{torch_tensor}}
#'   containing the integrated gradients with shape `(batch_size, ..., n_outputs)`.
#'   If `return_object = TRUE`: A \code{\link{TorchGradientResult}} object.
#'
#' @details
#' Integrated Gradients calculates feature importance by integrating gradients
#' along the path from a baseline \eqn{x'} to the input \eqn{x}:
#'
#' \deqn{(x - x') \times \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha (x - x'))}{\partial x} d\alpha}
#'
#' The integral is approximated using \code{n} interpolation steps.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' model <- nn_sequential(nn_linear(10, 3))
#' data <- torch_randn(5, 10)
#'
#' # Use zero baseline (default)
#' int_grads <- torch_intgrad(model, data)
#'
#' # Use custom baseline
#' baseline <- torch_zeros(1, 10)
#' int_grads <- torch_intgrad(model, data, x_ref = baseline)
#'
#' # More integration steps for higher accuracy
#' int_grads <- torch_intgrad(model, data, n = 100)
#' }
#'
#' @references
#' M. Sundararajan et al. (2017) \emph{Axiomatic attribution for deep networks.}
#' ICML 2017, PMLR 70, pp. 3319-3328.
#'
#' @seealso
#' \code{\link{run_intgrad}}, \code{\link{IntegratedGradient}}
#'
#' @export
torch_intgrad <- function(model,
                          data,
                          x_ref = NULL,
                          output_idx = NULL,
                          n = 50,
                          times_input = TRUE,
                          dtype = "float",
                          return_object = FALSE) {

  # Input validation
  checkmate::assert_class(model, "nn_module")
  checkmate::assert_choice(dtype, c("float", "double"))
  checkmate::assert_int(n, lower = 1)
  checkmate::assert_logical(times_input, len = 1)
  checkmate::assert_logical(return_object, len = 1)

  # Convert data to torch tensor if needed
  if (!inherits(data, "torch_tensor")) {
    data <- torch::torch_tensor(data)
  }

  # Set dtype
  torch_dtype <- if (dtype == "float") torch::torch_float() else torch::torch_double()
  data <- data$to(dtype = torch_dtype)

  # Convert model to the correct dtype
  model <- model$to(dtype = torch_dtype)

  batch_size <- data$shape[1]

  # Handle reference input
  if (is.null(x_ref)) {
    x_ref <- torch::torch_zeros_like(data[1, , drop = FALSE])
  } else if (!inherits(x_ref, "torch_tensor")) {
    x_ref <- torch::torch_tensor(x_ref)
  }

  x_ref <- x_ref$to(dtype = torch_dtype)

  # Validate x_ref shape
  if (x_ref$shape[1] != 1) {
    stop("x_ref must have batch size 1, got: ", x_ref$shape[1])
  }

  # Create interpolated inputs: x_ref + alpha * (data - x_ref)
  # Shape: (batch_size * n, ...)
  alphas <- torch::torch_linspace(0, 1, n, dtype = torch_dtype)

  # Repeat data n times
  data_repeated <- torch::torch_repeat_interleave(data, repeats = n, dim = 1)

  # Expand x_ref to match
  x_ref_expanded <- x_ref$expand(c(batch_size * n, -1))

  # Expand alphas for broadcasting
  alpha_shape <- c(batch_size * n, rep(1, data$dim() - 1))
  alphas_expanded <- alphas$`repeat`(batch_size)$view(alpha_shape)

  # Create interpolations
  interpolated <- x_ref_expanded + alphas_expanded * (data_repeated - x_ref_expanded)

  # Calculate gradients at interpolated points
  interpolated$requires_grad_(TRUE)

  # Forward pass
  output <- model(interpolated)

  # Handle output_idx
  if (is.null(output_idx)) {
    if (output$dim() == 2) {
      output_idx <- seq_len(output$shape[2])
    } else {
      output_idx <- 1
    }
  }

  # Calculate gradients for each output
  gradients_list <- list()

  for (i in seq_along(output_idx)) {
    idx <- output_idx[i]

    # Select output
    if (output$dim() == 2) {
      output_selected <- output[, idx]
    } else {
      output_selected <- output
    }

    # Sum for gradient calculation
    output_sum <- output_selected$sum()

    # Calculate gradient
    grad <- torch::autograd_grad(
      outputs = output_sum,
      inputs = interpolated,
      retain_graph = (i < length(output_idx)),
      create_graph = FALSE
    )[[1]]

    # Average gradients over integration steps
    # Reshape to (batch_size, n, ...)
    grad_shape <- c(batch_size, n, grad$shape[-1])
    grad_reshaped <- grad$view(grad_shape)

    # Average over n dimension
    grad_avg <- grad_reshaped$mean(dim = 2)

    gradients_list[[i]] <- grad_avg
  }

  # Stack gradients
  if (length(gradients_list) == 1) {
    result <- gradients_list[[1]]$unsqueeze(-1)
  } else {
    result <- torch::torch_stack(gradients_list, dim = -1)
  }

  # Multiply by (data - x_ref) if requested (standard IntGrad)
  if (times_input) {
    diff <- data - x_ref$expand_as(data)
    diff_expanded <- diff$unsqueeze(-1)$expand_as(result)
    result <- result * diff_expanded
  }

  # Clean up
  result <- result$detach()
  interpolated$requires_grad_(FALSE)

  # Calculate predictions for reference
  x_ref_output <- model(x_ref)

  # Wrap result if requested
  wrap_torch_result(
    result = result,
    return_object = return_object,
    data = data,
    model = model,
    method_name = "Integrated Gradient",
    output_idx = output_idx,
    preds = output,
    times_input = times_input,
    n = n,
    x_ref = x_ref
  )
}


#' @title Direct SmoothGrad for torch models
#'
#' @description
#' Calculate SmoothGrad by averaging gradients over noisy samples of the input.
#' This function provides a lightweight alternative to \code{\link{run_smoothgrad}}.
#'
#' @param model (\code{\link[torch]{nn_module}})\cr
#'   A torch model.
#' @param data (\code{\link[torch]{torch_tensor}}, `array`, or `matrix`)\cr
#'   Input data.
#' @param output_idx (`integer`)\cr
#'   Index or indices of output nodes. Default: `NULL` (all outputs).
#' @param n (`integer(1)`)\cr
#'   Number of noisy samples. Default: `50`.
#' @param noise_level (`numeric(1)`)\cr
#'   Standard deviation of noise relative to input range:
#'   \eqn{\sigma = (max(x) - min(x)) *} `noise_level`. Default: `0.1`.
#' @param times_input (`logical(1)`)\cr
#'   If `TRUE`, multiplies gradients by input (SmoothGrad×Input).
#'   Default: `FALSE`.
#' @param dtype (`character(1)`)\cr
#'   Data type: `"float"` or `"double"`. Default: `"float"`.
#' @param return_object (`logical(1)`)\cr
#'   If `TRUE`, returns a \code{\link{TorchGradientResult}} object with
#'   methods like `plot()` and `get_result()`. If `FALSE` (default), returns
#'   a raw \code{\link[torch]{torch_tensor}}.
#'
#' @return If `return_object = FALSE` (default): A \code{\link[torch]{torch_tensor}}
#'   containing the smoothed gradients with shape `(batch_size, ..., n_outputs)`.
#'   If `return_object = TRUE`: A \code{\link{TorchGradientResult}} object.
#'
#' @details
#' SmoothGrad computes gradients for \code{n} noisy versions of each input
#' and averages them. With \eqn{\epsilon \sim N(0,\sigma)}:
#'
#' \deqn{1/n \sum_{i=1}^n \frac{\partial f(x+ \epsilon_i)}{\partial x}}
#'
#' This reduces noise in gradient-based explanations.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' model <- nn_sequential(nn_linear(10, 3))
#' data <- torch_randn(5, 10)
#'
#' # Standard SmoothGrad
#' smooth_grads <- torch_smoothgrad(model, data)
#'
#' # SmoothGrad×Input
#' smooth_grads <- torch_smoothgrad(model, data, times_input = TRUE)
#'
#' # More samples for smoother result
#' smooth_grads <- torch_smoothgrad(model, data, n = 100)
#' }
#'
#' @references
#' D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' arXiv:1706.03825
#'
#' @seealso
#' \code{\link{run_smoothgrad}}, \code{\link{SmoothGrad}}
#'
#' @export
torch_smoothgrad <- function(model,
                              data,
                              output_idx = NULL,
                              n = 50,
                              noise_level = 0.1,
                              times_input = FALSE,
                              dtype = "float",
                              return_object = FALSE) {

  # Input validation
  checkmate::assert_class(model, "nn_module")
  checkmate::assert_choice(dtype, c("float", "double"))
  checkmate::assert_int(n, lower = 1)
  checkmate::assert_number(noise_level, lower = 0)
  checkmate::assert_logical(times_input, len = 1)
  checkmate::assert_logical(return_object, len = 1)

  # Convert data to torch tensor if needed
  if (!inherits(data, "torch_tensor")) {
    data <- torch::torch_tensor(data)
  }

  # Set dtype
  torch_dtype <- if (dtype == "float") torch::torch_float() else torch::torch_double()
  data <- data$to(dtype = torch_dtype)

  # Convert model to the correct dtype
  model <- model$to(dtype = torch_dtype)

  batch_size <- data$shape[1]

  # Calculate noise scale
  data_range <- (data$max() - data$min())$item()
  noise_scale <- if (data_range == 0) noise_level else data_range * noise_level

  # Create noisy samples
  # Shape: (batch_size * n, ...)
  data_repeated <- torch::torch_repeat_interleave(data, repeats = n, dim = 1)
  noise <- torch::torch_randn_like(data_repeated) * noise_scale
  noisy_data <- data_repeated + noise

  # Calculate gradients for noisy samples
  noisy_data$requires_grad_(TRUE)

  # Forward pass
  output <- model(noisy_data)

  # Handle output_idx
  if (is.null(output_idx)) {
    if (output$dim() == 2) {
      output_idx <- seq_len(output$shape[2])
    } else {
      output_idx <- 1
    }
  }

  # Calculate gradients for each output
  gradients_list <- list()

  for (i in seq_along(output_idx)) {
    idx <- output_idx[i]

    # Select output
    if (output$dim() == 2) {
      output_selected <- output[, idx]
    } else {
      output_selected <- output
    }

    # Sum for gradient calculation
    output_sum <- output_selected$sum()

    # Calculate gradient
    grad <- torch::autograd_grad(
      outputs = output_sum,
      inputs = noisy_data,
      retain_graph = (i < length(output_idx)),
      create_graph = FALSE
    )[[1]]

    # Average gradients over noise samples
    # Reshape to (batch_size, n, ...)
    grad_shape <- c(batch_size, n, grad$shape[-1])
    grad_reshaped <- grad$view(grad_shape)

    # Average over n dimension
    grad_avg <- grad_reshaped$mean(dim = 2)

    gradients_list[[i]] <- grad_avg
  }

  # Stack gradients
  if (length(gradients_list) == 1) {
    result <- gradients_list[[1]]$unsqueeze(-1)
  } else {
    result <- torch::torch_stack(gradients_list, dim = -1)
  }

  # Multiply by input if requested
  if (times_input) {
    data_expanded <- data$unsqueeze(-1)$expand_as(result)
    result <- result * data_expanded
  }

  # Clean up
  result <- result$detach()
  noisy_data$requires_grad_(FALSE)

  # Calculate clean predictions
  clean_output <- model(data)

  # Wrap result if requested
  wrap_torch_result(
    result = result,
    return_object = return_object,
    data = data,
    model = model,
    method_name = if (times_input) "SmoothGrad x Input" else "SmoothGrad",
    output_idx = output_idx,
    preds = clean_output,
    times_input = times_input,
    n = n,
    noise_level = noise_level
  )
}


#' @title Direct Expected Gradients for torch models
#'
#' @description
#' Calculate Expected Gradients (GradSHAP) using native torch autograd.
#' This function provides a lightweight alternative to \code{\link{run_expgrad}}.
#'
#' @param model (\code{\link[torch]{nn_module}})\cr
#'   A torch model.
#' @param data (\code{\link[torch]{torch_tensor}}, `array`, or `matrix`)\cr
#'   Input data.
#' @param data_ref (\code{\link[torch]{torch_tensor}}, `array`, or `matrix`)\cr
#'   Reference dataset for estimating conditional expectation.
#'   If `NULL`, uses zeros. Default: `NULL`.
#' @param output_idx (`integer`)\cr
#'   Index or indices of output nodes. Default: `NULL` (all outputs).
#' @param n (`integer(1)`)\cr
#'   Number of reference samples and integration steps. Default: `50`.
#' @param dtype (`character(1)`)\cr
#'   Data type: `"float"` or `"double"`. Default: `"float"`.
#' @param return_object (`logical(1)`)\cr
#'   If `TRUE`, returns a \code{\link{TorchGradientResult}} object with
#'   methods like `plot()` and `get_result()`. If `FALSE` (default), returns
#'   a raw \code{\link[torch]{torch_tensor}}.
#'
#' @return If `return_object = FALSE` (default): A \code{\link[torch]{torch_tensor}}
#'   containing the expected gradients with shape `(batch_size, ..., n_outputs)`.
#'   If `return_object = TRUE`: A \code{\link{TorchGradientResult}} object.
#'
#' @details
#' Expected Gradients extends Integrated Gradients by averaging over multiple
#' reference values from a distribution:
#'
#' \deqn{E_{x'\sim X', \alpha \sim U(0,1)}[(x - x') \times \frac{\partial f(x' + \alpha (x - x'))}{\partial x}]}
#'
#' This provides approximate Shapley values.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' model <- nn_sequential(nn_linear(10, 3))
#' data <- torch_randn(5, 10)
#' references <- torch_randn(100, 10)  # Reference distribution
#'
#' # Calculate Expected Gradients
#' exp_grads <- torch_expgrad(model, data, data_ref = references)
#' }
#'
#' @references
#' G. Erion et al. (2021) \emph{Improving performance of deep learning models
#' with axiomatic attribution priors and expected gradients.}
#' Nature Machine Intelligence 3, pp. 620-631.
#'
#' @seealso
#' \code{\link{run_expgrad}}, \code{\link{ExpectedGradient}}
#'
#' @export
torch_expgrad <- function(model,
                          data,
                          data_ref = NULL,
                          output_idx = NULL,
                          n = 50,
                          dtype = "float",
                          return_object = FALSE) {

  # Input validation
  checkmate::assert_class(model, "nn_module")
  checkmate::assert_choice(dtype, c("float", "double"))
  checkmate::assert_int(n, lower = 1)
  checkmate::assert_logical(return_object, len = 1)

  # Convert data to torch tensor if needed
  if (!inherits(data, "torch_tensor")) {
    data <- torch::torch_tensor(data)
  }

  # Set dtype
  torch_dtype <- if (dtype == "float") torch::torch_float() else torch::torch_double()
  data <- data$to(dtype = torch_dtype)

  # Convert model to the correct dtype
  model <- model$to(dtype = torch_dtype)

  batch_size <- data$shape[1]

  # Handle reference data
  if (is.null(data_ref)) {
    data_ref <- torch::torch_zeros_like(data[1:min(10, batch_size), , drop = FALSE])
  } else if (!inherits(data_ref, "torch_tensor")) {
    data_ref <- torch::torch_tensor(data_ref)
  }

  data_ref <- data_ref$to(dtype = torch_dtype)
  n_refs <- data_ref$shape[1]

  # Sample random references and alphas for each data point
  # Total samples: batch_size * n
  ref_idx <- sample.int(n_refs, size = batch_size * n, replace = TRUE)
  sampled_refs <- data_ref[ref_idx, , drop = FALSE]

  alphas <- torch::torch_rand(batch_size * n, dtype = torch_dtype)

  # Repeat data n times
  data_repeated <- torch::torch_repeat_interleave(data, repeats = n, dim = 1)

  # Expand alphas for broadcasting
  alpha_shape <- c(batch_size * n, rep(1, data$dim() - 1))
  alphas_expanded <- alphas$view(alpha_shape)

  # Create interpolations
  interpolated <- sampled_refs + alphas_expanded * (data_repeated - sampled_refs)

  # Calculate gradients at interpolated points
  interpolated$requires_grad_(TRUE)

  # Forward pass
  output <- model(interpolated)

  # Handle output_idx
  if (is.null(output_idx)) {
    if (output$dim() == 2) {
      output_idx <- seq_len(output$shape[2])
    } else {
      output_idx <- 1
    }
  }

  # Calculate gradients for each output
  gradients_list <- list()

  for (i in seq_along(output_idx)) {
    idx <- output_idx[i]

    # Select output
    if (output$dim() == 2) {
      output_selected <- output[, idx]
    } else {
      output_selected <- output
    }

    # Sum for gradient calculation
    output_sum <- output_selected$sum()

    # Calculate gradient
    grad <- torch::autograd_grad(
      outputs = output_sum,
      inputs = interpolated,
      retain_graph = (i < length(output_idx)),
      create_graph = FALSE
    )[[1]]

    # Multiply by (data - ref)
    diff <- data_repeated - sampled_refs
    grad_weighted <- grad * diff

    # Average over reference samples
    # Reshape to (batch_size, n, ...)
    grad_shape <- c(batch_size, n, grad$shape[-1])
    grad_reshaped <- grad_weighted$view(grad_shape)

    # Average over n dimension
    grad_avg <- grad_reshaped$mean(dim = 2)

    gradients_list[[i]] <- grad_avg
  }

  # Stack gradients
  if (length(gradients_list) == 1) {
    result <- gradients_list[[1]]$unsqueeze(-1)
  } else {
    result <- torch::torch_stack(gradients_list, dim = -1)
  }

  # Clean up
  result <- result$detach()
  interpolated$requires_grad_(FALSE)

  # Calculate predictions
  preds <- model(data)

  # Wrap result if requested
  wrap_torch_result(
    result = result,
    return_object = return_object,
    data = data,
    model = model,
    method_name = "Expected Gradient",
    output_idx = output_idx,
    preds = preds,
    n = n,
    data_ref = data_ref
  )
}
