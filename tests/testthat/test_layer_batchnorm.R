
test_that("Test initialization and forward of batchnorm layer", {
  library(torch)

  batch_size <- 10
  dim_in <- c(3, 10)
  dim_out <- c(3, 10)
  num_features <- 3
  eps <- abs(rnorm(1))
  gamma <- rnorm(3)
  beta <- rnorm(3) *0
  run_mean <- rnorm(3)*0
  run_var <- abs(rnorm(3))

  bn <- batchnorm_layer(num_features, eps, gamma, beta, run_mean, run_var,
                        dim_in, dim_out)

  input <- torch_randn(c(batch_size, dim_in))
  input_ref <- torch_randn(c(1, dim_in))


  # Test forward
  y <- bn(input)
  expect_equal(y$shape, c(batch_size,  dim_out))

  # Test update_ref
  y_ref <- bn$update_ref(input_ref)
  expect_equal(y_ref$shape, c(1,  dim_out))
})



test_that("Test get_gradient for batchnorm_layer", {
  library(torch)

  batch_size <- 10
  dim_in <- c(3, 5, 6)
  dim_out <- c(3, 5, 6)
  num_features <- 3
  eps <- abs(rnorm(1))
  gamma <- rnorm(3)
  beta <- rnorm(3)
  run_mean <- rnorm(3)
  run_var <- abs(rnorm(3))

  bn <- batchnorm_layer(num_features, eps, gamma, beta, run_mean, run_var,
                        dim_in, dim_out)
  input <- torch_randn(c(batch_size, dim_in), requires_grad = TRUE)

  # Test get_gradient
  y <- bn(input)
  sum(y)$backward()
  input_grad_true <- input$grad$unsqueeze(-1)
  input_grad <- bn$get_gradient(torch_ones_like(y$unsqueeze(-1)))

  expect_equal(dim(input_grad), dim(input_grad_true))
  expect_lt(as_array(mean((input_grad - input_grad_true)^2)), 1e-12)
})


test_that("Test get_input_relevances for batchnorm_layer", {
  library(torch)

  batch_size <- 5
  dim_in <- c(3, 10)
  dim_out <- c(3, 10)
  num_features <- 3
  eps <- abs(rnorm(1))
  gamma <- rnorm(3)
  beta <- rnorm(3) *0
  run_mean <- rnorm(3)*0
  run_var <- abs(rnorm(3))

  bn <- batchnorm_layer(num_features, eps, gamma, beta, run_mean, run_var,
                        dim_in, dim_out)

  input <- torch_randn(c(batch_size, dim_in))
  input_ref <- torch_randn(c(1, dim_in))

  out <- bn(input)

  rel <- torch_randn(c(batch_size, dim_out, 1))

  # Simple rule
  rel_simple <- bn$get_input_relevances(rel)
  expect_equal(dim(rel_simple), c(batch_size, dim_in, 1))
  expect_lt(as_array(mean((rel_simple - rel)^2)), 1e-12)

  # Epsilon rule
  rel_epsilon <- bn$get_input_relevances(rel, rule_name = "epsilon")
  expect_equal(dim(rel_epsilon), c(batch_size, dim_in, 1))

  # Alpha_Beta rule
  rel_alpha_beta <-
    bn$get_input_relevances(rel, rule_name = "alpha_beta")
  expect_equal(dim(rel_alpha_beta), c(batch_size, dim_in, 1))
  expect_lt(as_array(mean((rel_alpha_beta - rel)^2)), 1e-12)
})



test_that("Test get_input_multiplier for batchnorm_layer", {
  library(torch)

  batch_size <- 10
  dim_in <- c(3, 5, 6)
  dim_out <- c(3, 5, 6)
  num_features <- 3
  eps <- abs(rnorm(1))
  gamma <- rnorm(3)
  beta <- rnorm(3)
  run_mean <- rnorm(3)
  run_var <- abs(rnorm(3))

  bn <- batchnorm_layer(num_features, eps, gamma, beta, run_mean, run_var,
                        dim_in, dim_out)
  input <- torch_randn(c(batch_size, dim_in), requires_grad = TRUE)
  input_ref <- torch_randn(c(1, dim_in))

  y <- bn(input)
  y_ref <- bn$update_ref(input_ref)

  mult <- torch_randn(c(batch_size, dim_out, 1))
  diff_input <- (input - input_ref)$unsqueeze(-1)
  diff_output <- (y - y_ref)$unsqueeze(-1)

  contrib_true <- sum(mult * diff_output, dim = 2:4)

  mult_in <- bn$get_input_multiplier(mult)
  expect_equal(dim(mult_in), c(batch_size, dim_in, 1))
  contrib <- sum(mult_in * diff_input, dim = 2:4)
  expect_lt(as_array(mean((contrib - contrib_true)^2)), 1e-10)
})
