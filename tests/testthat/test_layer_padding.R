
test_that("Test padding layer for dense input", {
  library(torch)

  batch_size <- 10
  pad <- c(1,3)
  dim_in <- c(8)
  dim_out <- dim_in + c(4)

  padding <- padding_layer(pad, dim_in, dim_out)
  input <- torch_randn(c(batch_size, dim_in))
  input_ref <- torch_randn(c(1, dim_in))

  # Test forward (channels first)
  y <- padding(input)
  expect_equal(dim(y), c(batch_size, dim_out))

  # Test update_ref
  y_ref <- padding$update_ref(input_ref)
  expect_equal(dim(y_ref), c(1, dim_out))

  # Test reshape_to_input
  y <- padding(input)$unsqueeze(-1)
  x_new <- padding$reshape_to_input(y)$squeeze()
  expect_equal(dim(input), dim(x_new))
  expect_true(torch_equal(input, x_new))
})

test_that("Test padding layer for 1D input", {
  library(torch)

  batch_size <- 10
  channels <- 3
  pad <- c(5,2)
  dim_in <- c(channels, 20)
  dim_out <- dim_in + c(0, 7)

  padding <- padding_layer(pad, dim_in, dim_out)
  input <- torch_randn(c(batch_size, dim_in))
  input_ref <- torch_randn(c(1, dim_in))

  # Test forward (channels first)
  y <- padding(input)
  expect_equal(dim(y), c(batch_size, dim_out))

  # Test update_ref
  y_ref <- padding$update_ref(input_ref)
  expect_equal(dim(y_ref), c(1, dim_out))

  # Test reshape_to_input
  y <- padding(input)$unsqueeze(-1)
  x_new <- padding$reshape_to_input(y)$squeeze()
  expect_equal(dim(input), dim(x_new))
  expect_true(torch_equal(input, x_new))
})


test_that("Test padding layer for 2D input", {
  library(torch)

  batch_size <- 10
  channels <- 3
  pad <- c(5, 2, 0, 6)
  dim_in <- c(channels, 20, 10)
  dim_out <- dim_in + c(0, 6, 7)

  padding <- padding_layer(pad, dim_in, dim_out)
  input <- torch_randn(c(batch_size, dim_in))
  input_ref <- torch_randn(c(1, dim_in))

  # Test forward (channels first)
  y <- padding(input)
  expect_equal(dim(y), c(batch_size, dim_out))

  # Test update_ref
  y_ref <- padding$update_ref(input_ref)
  expect_equal(dim(y_ref), c(1, dim_out))

  # Test reshape_to_input
  y <- padding(input)$unsqueeze(-1)
  x_new <- padding$reshape_to_input(y)$squeeze()
  expect_equal(dim(input), dim(x_new))
  expect_true(torch_equal(input, x_new))
})
