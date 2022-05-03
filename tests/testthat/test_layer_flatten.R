
test_that("Test initialization and forward of flatten layer", {
  library(torch)

  batch_size <- 10
  channels <- 3
  axis <- c(10, 5)
  dim_in <- c(channels, axis)
  dim_out <- prod(axis) * channels

  #
  # Test for dtype = "float"
  #
  flatten_float <- flatten_layer(dim_in = dim_in, dim_out = dim_out)

  input <- torch_randn(c(batch_size, dim_in), dtype = torch_float())
  input_ref <- torch_randn(c(1, dim_in), dtype = torch_float())

  # Test forward (channels first)
  y_float <- flatten_float(input)
  expect_true(y_float$dtype == torch_float())
  expect_equal(dim(y_float), c(batch_size, dim_out))

  # Test forward (channels last)
  y_float_last <- flatten_float(input, channels_first = FALSE)
  expect_true(y_float_last$dtype == torch_float())
  expect_equal(dim(y_float_last), c(batch_size, dim_out))

  # Test update_ref (channels first)
  y_ref_float <- flatten_float$update_ref(input_ref)
  expect_equal(dim(y_ref_float), c(1, dim_out))

  # Test update_ref (channels last)
  y_ref_float_last <- flatten_float$update_ref(input_ref, channels_first = FALSE)
  expect_equal(dim(y_ref_float_last), c(1, dim_out))

  #
  # Test for dtype = "double"
  #

  flatten_double <- flatten_layer(dim_in = dim_in, dim_out = dim_out)

  input <- torch_randn(c(batch_size, dim_in), dtype = torch_double())
  input_ref <- torch_randn(c(1, dim_in), dtype = torch_double())

  # Test forward (channels first)
  y_float <- flatten_double(input)
  expect_true(y_float$dtype == torch_double())
  expect_equal(dim(y_float), c(batch_size, dim_out))

  # Test forward (channels last)
  y_float_last <- flatten_double(input, channels_first = FALSE)
  expect_true(y_float_last$dtype == torch_double())
  expect_equal(dim(y_float_last), c(batch_size, dim_out))

  # Test update_ref (channels first)
  y_ref_float <- flatten_double$update_ref(input_ref)
  expect_equal(dim(y_ref_float), c(1, dim_out))

  # Test update_ref (channels last)
  y_ref_float_last <- flatten_double$update_ref(input_ref, channels_first = FALSE)
  expect_equal(dim(y_ref_float_last), c(1, dim_out))
})


test_that("Test function reshape_to_input for 1D-CNN", {
  batch_size <- 10
  channels <- 3
  axis <- 20
  dim_in <- c(channels, axis)
  dim_out <- prod(axis) * channels

  # Channels first

  # Sample input data
  input <- torch_randn(c(batch_size, dim_in))
  # Create flatten layer
  flatten <- flatten_layer(dim_in = dim_in, dim_out = dim_out)
  # Calculate output
  out <- flatten(input)
  # Define upper-layer relevance
  rel_out <- torch_stack(list(out, 2 * out, 3 * out), dim = -1)

  # True input relevance
  rel_true <- torch_stack(list(input, 2 * input, 3 * input), dim = -1)
  rel_in <- flatten$reshape_to_input(rel_out)

  expect_equal(rel_in$shape, rel_true$shape)
  expect_lt(as_array(mean((rel_in - rel_true)^2)), 1e-10)

  # Channels last

  # Sample input data
  input <- torch_randn(c(batch_size, dim_in))
  # Create flatten layer
  flatten <- flatten_layer(dim_in = dim_in, dim_out = dim_out)
  # Calculate output
  out <- flatten(input, channels_first = FALSE)
  # Define upper-layer relevance
  rel_out <- torch_stack(list(out, 2 * out, 3 * out), dim = -1)

  # True input relevance
  rel_true <- torch_stack(list(input, 2 * input, 3 * input), dim = -1)
  rel_in <- flatten$reshape_to_input(rel_out)

  expect_equal(rel_in$shape, rel_true$shape)
  expect_lt(as_array(mean((rel_in - rel_true)^2)), 1e-10)
})

test_that("Test function reshape_to_input for 2D-CNN", {
  batch_size <- 10
  channels <- 4
  axis <- c(8, 9)
  dim_in <- c(channels, axis)
  dim_out <- prod(axis) * channels

  # Channels first

  # Sample input data
  input <- torch_randn(c(batch_size, dim_in))
  # Create flatten layer
  flatten <- flatten_layer(dim_in = dim_in, dim_out = dim_out)
  # Calculate output
  out <- flatten(input)
  # Define upper-layer relevance
  rel_out <- torch_stack(list(out, 2 * out, 3 * out), dim = -1)

  # True input relevance
  rel_true <- torch_stack(list(input, 2 * input, 3 * input), dim = -1)
  rel_in <- flatten$reshape_to_input(rel_out)

  expect_equal(rel_in$shape, rel_true$shape)
  expect_lt(as_array(mean((rel_in - rel_true)^2)), 1e-10)

  # Channels last

  # Sample input data
  input <- torch_randn(c(batch_size, dim_in))
  # Create flatten layer
  flatten <- flatten_layer(dim_in = dim_in, dim_out = dim_out)
  # Calculate output
  out <- flatten(input, channels_first = FALSE)
  # Define upper-layer relevance
  rel_out <- torch_stack(list(out, 2 * out, 3 * out), dim = -1)

  # True input relevance
  rel_true <- torch_stack(list(input, 2 * input, 3 * input), dim = -1)
  rel_in <- flatten$reshape_to_input(rel_out)

  expect_equal(rel_in$shape, rel_true$shape)
  expect_lt(as_array(mean((rel_in - rel_true)^2)), 1e-10)
})
