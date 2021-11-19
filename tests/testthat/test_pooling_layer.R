

test_that("Test 1D average pooling layer", {
  library(torch)

  x <- torch_randn(10, 4, 30)
  x_ref <- torch_randn(1, 4, 30)
  kernel_size <- c(2)

  avg_pool1d <- avg_pool1d_layer(kernel_size, c(4, 30), c(4, 15))

  # Works properly
  y_true <- nnf_avg_pool1d(x, kernel_size)
  y_ref_true <- nnf_avg_pool1d(x_ref, kernel_size)
  y <- avg_pool1d(x)
  y_ref <- avg_pool1d$update_ref(x_ref)

  expect_lt(as_array(mean(y-y_true)), 1e-12)
  expect_lt(as_array(mean(y_ref-y_ref_true)), 1e-12)

  # Test LRP: simple rule
  rel_output <- torch_randn(c(10, 4, 15, 3))
  rel <- avg_pool1d$get_input_relevances(rel_output)

  expect_equal(dim(rel), c(10, 4, 30, 3))
  expect_lt(as_array((sum(rel_output) - sum(rel))^2), 1e-3)

  # Test LRP: alpha-beta-rule
  rel_output <- torch_randn(c(10, 4, 15, 3))
  rel <- avg_pool1d$get_input_relevances(rel_output, rule_name = "alpha_beta")

  expect_equal(dim(rel), c(10, 4, 30, 3))

  # Test DeepLift
  multiplier <- torch_randn(c(10, 4, 15, 3))
  contrib_true <- (y_true - y_ref_true)$unsqueeze(-1) * multiplier
  mul <- avg_pool1d$get_input_multiplier(multiplier)
  contrib <- (x - x_ref)$unsqueeze(-1) * mul

  expect_equal(dim(mul), c(10, 4, 30, 3))
  expect_lt(as_array((sum(contrib) - sum(contrib_true))^2), 1e-8)
})


test_that("Test 2D average pooling layer", {
  library(torch)

  x <- torch_randn(10, 4, 20, 10)
  x_ref <- torch_randn(1, 4, 20, 10)
  kernel_size <- c(2,2)

  avg_pool2d <- avg_pool2d_layer(kernel_size, c(4, 20, 10), c(4, 10, 5))

  # Works properly
  y_true <- nnf_avg_pool2d(x, kernel_size)
  y_ref_true <- nnf_avg_pool2d(x_ref, kernel_size)
  y <- avg_pool2d(x)
  y_ref <- avg_pool2d$update_ref(x_ref)

  expect_lt(as_array(mean(y-y_true)), 1e-12)
  expect_lt(as_array(mean(y_ref-y_ref_true)), 1e-12)

  # Test LRP: simple rule
  rel_output <- torch_randn(c(10, 4, 10, 5, 3))
  rel <- avg_pool2d$get_input_relevances(rel_output)

  expect_equal(dim(rel), c(10, 4, 20, 10, 3))
  expect_lt(as_array((sum(rel_output) - sum(rel))^2), 1e-3)

  # Test LRP: alpha-beta-rule
  rel_output <- torch_randn(c(10, 4, 10, 5, 3))
  rel <- avg_pool2d$get_input_relevances(rel_output, rule_name = "alpha_beta")

  expect_equal(dim(rel), c(10, 4, 20, 10, 3))


  # Test DeepLift
  multiplier <- torch_randn(c(10, 4, 10, 5, 3))
  contrib_true <- (y_true - y_ref_true)$unsqueeze(-1) * multiplier
  mul <- avg_pool2d$get_input_multiplier(multiplier)
  contrib <- (x - x_ref)$unsqueeze(-1) * mul

  expect_equal(dim(mul), c(10, 4, 20, 10, 3))
  expect_lt(as_array((sum(contrib) - sum(contrib_true))^2), 1e-8)
})



test_that("Test 1D maximum pooling layer", {
  library(torch)

  x <- torch_randn(10, 4, 30)
  x_ref <- torch_randn(1, 4, 30)
  kernel_size <- c(2)

  max_pool1d <- max_pool1d_layer(kernel_size, c(4, 30), c(4, 15))

  # Works properly
  y_true <- nnf_max_pool1d(x, kernel_size)
  y_ref_true <- nnf_max_pool1d(x_ref, kernel_size)
  y <- max_pool1d(x)
  y_ref <- max_pool1d$update_ref(x_ref)

  expect_lt(as_array(mean(y-y_true)), 1e-12)
  expect_lt(as_array(mean(y_ref-y_ref_true)), 1e-12)

  # Test LRP: eps-rule
  rel_output <- torch_randn(c(10, 4, 15, 3))
  rel <- max_pool1d$get_input_relevances(rel_output)

  expect_equal(dim(rel), c(10, 4, 30, 3))
  expect_lt(as_array((sum(rel_output) - sum(rel))^2), 1e-3)

  # Test LRP: alpha-beta-rule
  rel_output <- torch_randn(c(10, 4, 15, 3))
  rel <- max_pool1d$get_input_relevances(rel_output, rule_name = "alpha_beta")

  expect_equal(dim(rel), c(10, 4, 30, 3))

  # Test DeepLift
  multiplier <- torch_randn(c(10, 4, 15, 3))
  contrib_true <- (y_true - y_ref_true)$unsqueeze(-1) * multiplier
  mul <- max_pool1d$get_input_multiplier(multiplier)
  contrib <- (x - x_ref)$unsqueeze(-1) * mul

  expect_equal(dim(mul), c(10, 4, 30, 3))
})


test_that("Test 2D maximum pooling layer", {
  library(torch)

  x <- torch_randn(10, 4, 20, 10)
  x_ref <- torch_randn(1, 4, 20, 10)
  kernel_size <- c(2,2)

  max_pool2d <- max_pool2d_layer(kernel_size, c(4, 20, 10), c(4, 10, 5))

  # Works properly
  y_true <- nnf_max_pool2d(x, kernel_size)
  y_ref_true <- nnf_max_pool2d(x_ref, kernel_size)
  y <- max_pool2d(x)
  y_ref <- max_pool2d$update_ref(x_ref)

  expect_lt(as_array(mean(y-y_true)), 1e-12)
  expect_lt(as_array(mean(y_ref-y_ref_true)), 1e-12)

  # Test LRP: simple rule
  rel_output <- torch_randn(c(10, 4, 10, 5, 3))
  rel <- max_pool2d$get_input_relevances(rel_output)

  expect_equal(dim(rel), c(10, 4, 20, 10, 3))
  expect_lt(as_array((sum(rel_output) - sum(rel))^2), 1e-3)

  # Test LRP: alpha-beta-rule
  rel_output <- torch_randn(c(10, 4, 10, 5, 3))
  rel <- max_pool2d$get_input_relevances(rel_output, rule_name = "alpha_beta")

  expect_equal(dim(rel), c(10, 4, 20, 10, 3))

  # Test DeepLift
  multiplier <- torch_randn(c(10, 4, 10, 5, 3))
  contrib_true <- (y_true - y_ref_true)$unsqueeze(-1) * multiplier
  mul <- max_pool2d$get_input_multiplier(multiplier)
  contrib <- (x - x_ref)$unsqueeze(-1) * mul

  expect_equal(dim(mul), c(10, 4, 20, 10, 3))
})

