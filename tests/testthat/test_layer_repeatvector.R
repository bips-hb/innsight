
test_that("Test initialization and forward of repeatvector layer", {
  library(torch)

  batch_size <- 10
  n <- 3
  dim_in <- c(5)
  dim_out <- c(n, 5)

  repVec <- repeatvector_layer(n = n, dim_in = dim_in, dim_out = dim_out)

  input <- torch_randn(batch_size, dim_in)
  input_ref <- torch_randn(1, dim_in)

  # Test forward
  y <- repVec(input)
  expect_equal(y$shape, c(10,  dim_out))

  # Test update_ref
  y_ref <- repVec$update_ref(input_ref)
  expect_equal(y_ref$shape, c(1,  dim_out))
})

test_that("Test reshape_to_input and get_input_multipier for repeatvector layer", {
  library(torch)

  batch_size <- 10
  n <- 3
  dim_in <- c(5)
  dim_out <- c(n, 5)

  repVec <- repeatvector_layer(n = n, dim_in = dim_in, dim_out = dim_out)

  input <- torch_randn(batch_size, dim_in)
  input_ref <- torch_randn(1, dim_in)
  out <- repVec(input)
  out_ref <- repVec$update_ref(input_ref)

  # Reshape to input
  out_rel <- torch_randn(c(batch_size, dim_out, 4))
  in_rel <- repVec$get_input_relevances(out_rel)
  expect_equal(dim(in_rel), c(batch_size, dim_in, 4))
  expect_lt(as_array(sum(in_rel) - sum(out_rel))**2, 1e-8)

  # Get input multiplier
  mult <- torch_randn(c(batch_size, dim_out, 4))
  diff_input <- (input - input_ref)$unsqueeze(3)
  diff_output <- (out - out_ref)$unsqueeze(4)

  contrib_true <- sum(mult * diff_output, dim = 2:3)
  mult_in <- repVec$get_input_multiplier(mult)
  expect_equal(dim(mult_in), c(batch_size, dim_in, 4))
  contrib <- sum(mult_in * diff_input, dim = 2)
  expect_lt(as_array(mean((contrib - contrib_true)^2)), 1e-10)
})
