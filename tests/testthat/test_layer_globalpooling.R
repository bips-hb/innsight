
test_that("Test initialization and forward of GlobalAveragePooling layer", {
  library(torch)

  batch_size <- 5
  dim_in <- c(3, 10, 10)
  dim_out <- c(1, 10, 10)

  global_avgpool <- global_avgpool_layer(dim_in = dim_in, dim_out = dim_out)

  input <- torch_randn(c(batch_size, dim_in))
  input_ref <- torch_randn(c(1, dim_in))

  # Test forward
  y <- global_avgpool(input)
  expect_equal(y$shape, c(batch_size,  dim_out))

  # Test update_ref
  y_ref <- global_avgpool$update_ref(input_ref)
  expect_equal(y_ref$shape, c(1,  dim_out))
})


test_that("Test function reshape_to_input for GlobalAveragePooling layer", {
  library(torch)

  batch_size <- 5
  dim_in <- c(3, 10, 10)
  dim_out <- c(1, 10, 10)

  global_avgpool <- global_avgpool_layer(dim_in = dim_in, dim_out = dim_out)

  input <- torch_randn(c(batch_size, dim_in))

  # Forward pass
  out <- global_avgpool(input)
  # Define output relevance
  rel_out <- out$unsqueeze(-1)
  rel_in <- global_avgpool$reshape_to_input(rel_out)

  expect_equal(dim(rel_in), c(batch_size, dim_in, 1))
  expect_lt(mean(as.array((rel_in$sum(2, keepdim = TRUE) - rel_out)^2)), 1e-10)
})


test_that("Test initialization and forward of GlobalMaxPooling layer", {
  library(torch)

  batch_size <- 5
  dim_in <- c(3, 10, 10)
  dim_out <- c(1, 10, 10)

  global_avgpool <- global_maxpool_layer(dim_in = dim_in, dim_out = dim_out)

  input <- torch_randn(c(batch_size, dim_in))
  input_ref <- torch_randn(c(1, dim_in))

  # Test forward
  y <- global_avgpool(input)
  expect_equal(y$shape, c(batch_size,  dim_out))

  # Test update_ref
  y_ref <- global_avgpool$update_ref(input_ref)
  expect_equal(y_ref$shape, c(1,  dim_out))
})


test_that("Test function reshape_to_input for GlobalMaxPooling layer", {
  library(torch)

  batch_size <- 5
  dim_in <- c(3, 10, 10)
  dim_out <- c(1, 10, 10)

  global_avgpool <- global_maxpool_layer(dim_in = dim_in, dim_out = dim_out)

  input <- torch_randn(c(batch_size, dim_in))

  # Forward pass
  out <- global_avgpool(input)
  # Define output relevance
  rel_out <- out$unsqueeze(-1)
  rel_in <- global_avgpool$reshape_to_input(rel_out)

  expect_equal(dim(rel_in), c(batch_size, dim_in, 1))
  expect_lt(mean(as.array((rel_in$sum(2, keepdim = TRUE) - rel_out)^2)), 1e-10)
})
