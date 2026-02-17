###############################################################################
#                    Tests for Direct Torch Gradient Methods
###############################################################################

test_that("torch_grad works with simple model", {
  skip_if_not_installed("torch")

  # Create a simple model
  model <- nn_sequential(
    nn_linear(5, 10),
    nn_relu(),
    nn_linear(10, 3)
  )

  # Create test data
  set.seed(123)
  data <- torch_randn(2, 5)

  # Test vanilla gradient
  grads <- torch_grad(model, data, output_idx = 1)

  expect_true(inherits(grads, "torch_tensor"))
  expect_equal(grads$shape, c(2, 5, 1))  # (batch, features, outputs)

  # Test Gradient×Input
  grads_times_input <- torch_grad(model, data, output_idx = 1, times_input = TRUE)

  expect_true(inherits(grads_times_input, "torch_tensor"))
  expect_equal(grads_times_input$shape, c(2, 5, 1))
})


test_that("torch_grad equivalent to run_grad", {
  skip_if_not_installed("torch")

  # Create simple sequential model
  model <- nn_sequential(
    nn_linear(5, 3)
  )

  # Test data
  set.seed(123)
  data_tensor <- torch_randn(3, 5)
  data_array <- as.array(data_tensor)

  # Direct torch method
  grads_torch <- torch_grad(model, data_tensor, output_idx = 1, times_input = FALSE)

  # Converter method
  converter <- Converter$new(model, input_dim = c(5))
  grads_converter <- Gradient$new(
    converter,
    data_array,
    output_idx = 1,
    times_input = FALSE,
    verbose = FALSE
  )

  result_converter <- grads_converter$get_result("array")

  # Compare results (allowing small numerical differences)
  # Ignore dimnames as torch results don't have them
  expect_equal(
    as.numeric(grads_torch[,,1]),
    as.numeric(result_converter[,,1]),
    tolerance = 1e-5
  )
})


test_that("torch_intgrad works correctly", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  set.seed(123)
  data <- torch_randn(2, 5)
  x_ref <- torch_zeros(1, 5)

  # Test with explicit baseline
  int_grads <- torch_intgrad(model, data, x_ref = x_ref, n = 30)

  expect_true(inherits(int_grads, "torch_tensor"))
  expect_equal(int_grads$shape, c(2, 5, 3))

  # Test with default (zero) baseline
  int_grads_default <- torch_intgrad(model, data, n = 30)

  # Should be very similar to explicit zero baseline
  diff <- (int_grads - int_grads_default)$abs()$max()$item()
  expect_lt(diff, 1e-5)
})


test_that("torch_intgrad equivalent to run_intgrad", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  set.seed(123)
  data_tensor <- torch_randn(2, 5)
  data_array <- as.array(data_tensor)
  x_ref_tensor <- torch_zeros(1, 5)
  x_ref_array <- as.array(x_ref_tensor)

  # Direct torch method
  grads_torch <- torch_intgrad(
    model, data_tensor,
    x_ref = x_ref_tensor,
    output_idx = 1,
    n = 50,
    times_input = TRUE
  )

  # Converter method
  converter <- Converter$new(model, input_dim = c(5))
  grads_converter <- IntegratedGradient$new(
    converter,
    data_array,
    x_ref = x_ref_array,
    output_idx = 1,
    n = 50,
    times_input = TRUE,
    verbose = FALSE
  )

  result_converter <- grads_converter$get_result("array")

  # Compare results (ignore dimnames)
  expect_equal(
    as.numeric(grads_torch[,,1]),
    as.numeric(result_converter[,,1]),
    tolerance = 1e-4
  )
})


test_that("torch_smoothgrad works correctly", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  set.seed(123)
  data <- torch_randn(2, 5)

  # Test SmoothGrad
  smooth_grads <- torch_smoothgrad(model, data, n = 20, noise_level = 0.1)

  expect_true(inherits(smooth_grads, "torch_tensor"))
  expect_equal(smooth_grads$shape, c(2, 5, 3))

  # Test SmoothGrad×Input
  smooth_grads_input <- torch_smoothgrad(
    model, data,
    n = 20,
    times_input = TRUE
  )

  expect_true(inherits(smooth_grads_input, "torch_tensor"))
  expect_equal(smooth_grads_input$shape, c(2, 5, 3))
})


test_that("torch_smoothgrad equivalent to run_smoothgrad", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  # Use same seed for reproducibility
  set.seed(42)
  torch_manual_seed(42)
  data_tensor <- torch_randn(2, 5)
  data_array <- as.array(data_tensor)

  # Direct torch method
  set.seed(42)
  torch_manual_seed(42)
  grads_torch <- torch_smoothgrad(
    model, data_tensor,
    output_idx = 1,
    n = 50,
    noise_level = 0.1,
    times_input = FALSE
  )

  # Converter method
  converter <- Converter$new(model, input_dim = c(5))
  set.seed(42)
  torch_manual_seed(42)
  grads_converter <- SmoothGrad$new(
    converter,
    data_array,
    output_idx = 1,
    n = 50,
    noise_level = 0.1,
    times_input = FALSE,
    verbose = FALSE
  )

  result_converter <- grads_converter$get_result("array")

  # Compare results (allowing for some numerical variance due to randomness)
  # Ignore dimnames
  expect_equal(
    as.numeric(grads_torch[,,1]),
    as.numeric(result_converter[,,1]),
    tolerance = 1e-3
  )
})


test_that("torch_expgrad works correctly", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  set.seed(123)
  data <- torch_randn(2, 5)
  data_ref <- torch_randn(10, 5)

  # Test Expected Gradients
  exp_grads <- torch_expgrad(model, data, data_ref = data_ref, n = 20)

  expect_true(inherits(exp_grads, "torch_tensor"))
  expect_equal(exp_grads$shape, c(2, 5, 3))

  # Test with default (zero) baseline
  exp_grads_default <- torch_expgrad(model, data, n = 20)

  expect_true(inherits(exp_grads_default, "torch_tensor"))
  expect_equal(exp_grads_default$shape, c(2, 5, 3))
})


test_that("torch_expgrad equivalent to run_expgrad", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  set.seed(42)
  torch_manual_seed(42)
  data_tensor <- torch_randn(2, 5)
  data_array <- as.array(data_tensor)
  data_ref_tensor <- torch_randn(20, 5)
  data_ref_array <- as.array(data_ref_tensor)

  # Direct torch method
  set.seed(42)
  torch_manual_seed(42)
  grads_torch <- torch_expgrad(
    model, data_tensor,
    data_ref = data_ref_tensor,
    output_idx = 1,
    n = 30
  )

  # Converter method
  converter <- Converter$new(model, input_dim = c(5))
  set.seed(42)
  torch_manual_seed(42)
  grads_converter <- ExpectedGradient$new(
    converter,
    data_array,
    data_ref = data_ref_array,
    output_idx = 1,
    n = 30,
    verbose = FALSE
  )

  result_converter <- grads_converter$get_result("array")

  # Compare results (allowing for variance due to sampling)
  # Ignore dimnames
  expect_equal(
    as.numeric(grads_torch[,,1]),
    as.numeric(result_converter[,,1]),
    tolerance = 1e-3
  )
})


test_that("torch_grad handles multiple output indices", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 4)
  )

  data <- torch_randn(2, 5)

  # Test with multiple outputs
  grads <- torch_grad(model, data, output_idx = c(1, 3))

  expect_equal(grads$shape, c(2, 5, 2))  # 2 outputs selected
})


test_that("torch_grad handles array input", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  # Test with R array input
  data_array <- matrix(rnorm(10), nrow = 2, ncol = 5)

  grads <- torch_grad(model, data_array)

  expect_true(inherits(grads, "torch_tensor"))
  expect_equal(grads$shape, c(2, 5, 3))
})


test_that("torch methods work with double dtype", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  data <- torch_randn(2, 5)

  # Test with double precision
  grads_double <- torch_grad(model, data, dtype = "double")

  expect_true(grads_double$dtype == torch_double())
})


test_that("torch_grad input validation works", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  data <- torch_randn(2, 5)

  # Test invalid dtype
  expect_error(
    torch_grad(model, data, dtype = "invalid"),
    "dtype"
  )

  # Test invalid times_input
  expect_error(
    torch_grad(model, data, times_input = "yes"),
    "times_input"
  )

  # Test invalid model class
  expect_error(
    torch_grad("not a model", data),
    "nn_module"
  )
})


test_that("return_object = TRUE returns TorchGradientResult", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  data <- torch_randn(2, 5)

  # Test torch_grad
  result <- torch_grad(model, data, return_object = TRUE)
  expect_s3_class(result, "TorchGradientResult")
  expect_true(inherits(result, "R6"))

  # Test torch_intgrad
  result_int <- torch_intgrad(model, data, return_object = TRUE)
  expect_s3_class(result_int, "TorchGradientResult")

  # Test torch_smoothgrad
  result_smooth <- torch_smoothgrad(model, data, return_object = TRUE, n = 10)
  expect_s3_class(result_smooth, "TorchGradientResult")

  # Test torch_expgrad
  result_exp <- torch_expgrad(model, data, return_object = TRUE, n = 10)
  expect_s3_class(result_exp, "TorchGradientResult")
})


test_that("TorchGradientResult methods work correctly", {
  skip_if_not_installed("torch")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  data <- torch_randn(2, 5)

  result <- torch_grad(model, data, output_idx = 1, return_object = TRUE)

  # Test get_result methods
  result_array <- result$get_result("array")
  expect_true(is.array(result_array))
  expect_equal(dim(result_array), c(2, 5, 1))

  result_tensor <- result$get_result("torch_tensor")
  expect_true(inherits(result_tensor, "torch_tensor"))

  result_df <- result$get_result("data.frame")
  expect_true(is.data.frame(result_df))

  # Test print method
  expect_output(print(result), "TorchGradientResult")
  expect_output(print(result), "Method: Gradient")
})


test_that("TorchGradientResult plot method works", {
  skip_if_not_installed("torch")
  skip_if_not_installed("ggplot2")

  model <- nn_sequential(
    nn_linear(5, 3)
  )

  data <- torch_randn(2, 5)

  result <- torch_grad(model, data, output_idx = 1, return_object = TRUE)

  # Test plot method
  p <- plot(result, data_idx = 1, output_idx = 1)
  expect_s3_class(p, "ggplot")
})
