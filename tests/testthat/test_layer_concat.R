
test_that("Test initialization and forward of concatenate layer", {
  library(torch)

  batch_size <- 10
  axis <- c(10, 5)
  dim_in <- list(c(axis, 1), c(axis, 3), c(axis, 6))
  dim_out <- c(axis, 10)
  concat_dim <- 4

  concat <- concatenate_layer(dim = concat_dim,
                              dim_in = dim_in,
                              dim_out = dim_out)

  input <- lapply(dim_in, function(x) torch_randn(c(batch_size, x)))

  input_ref <- lapply(dim_in, function(x) torch_randn(c(1, x)))

  # Test forward
  y <- concat(input)
  expect_equal(y$shape, c(10,  dim_out))

  # Test update_ref
  y_ref <- concat$update_ref(input_ref)
  expect_equal(y_ref$shape, c(1,  dim_out))
})


test_that("Test function reshape_to_input for dense layer", {
  library(torch)

  batch_size <- 10
  dim_in <- list(c(1), c(3), c(6))
  dim_out <- c(10)
  concat_dim <- 2

  concat <- concatenate_layer(dim = concat_dim,
                              dim_in = dim_in,
                              dim_out = dim_out)

  input <- lapply(dim_in, function(x) torch_randn(c(batch_size, x)))
  rel_in_true <- lapply(input, function(x) torch_stack(list(x, 2 * x, 5 * x), dim = -1))

  # Forward pass
  out <- concat(input)
  # Define output relevance
  rel_out <- torch_stack(list(out, 2 * out, 5 * out), dim = -1)

  rel_in <- concat$reshape_to_input(rel_out)

  expect_equal(lapply(rel_in, dim), lapply(rel_in_true, dim))
  expect_lt(mean(as.array((rel_in[[1]] - rel_in_true[[1]])^2)), 1e-10)
  expect_lt(mean(as.array((rel_in[[2]] - rel_in_true[[2]])^2)), 1e-10)
  expect_lt(mean(as.array((rel_in[[3]] - rel_in_true[[3]])^2)), 1e-10)

  rel_total_true <- rel_out$sum(c(2,3))
  rel_total <- torch_stack(lapply(rel_in,
                                  function(x) x$sum(c(2,3))), dim = -1)$sum(2)
  expect_lt(mean(as.array((rel_total - rel_total_true)^2)), 1e-8)
})

test_that("Test function reshape_to_input for 1D-CNN", {
  library(torch)

  batch_size <- 10
  axis <- 5
  dim_in <- list(c(axis, 1), c(axis, 3), c(axis, 6))
  dim_out <- c(axis, 10)
  concat_dim <- 3

  concat <- concatenate_layer(dim = concat_dim,
                              dim_in = dim_in,
                              dim_out = dim_out)

  input <- lapply(dim_in, function(x) torch_randn(c(batch_size, x)))
  rel_in_true <- lapply(input, function(x) torch_stack(list(x, 2 * x, 5 * x), dim = -1))

  # Forward pass
  out <- concat(input)
  # Define output relevance
  rel_out <- torch_stack(list(out, 2 * out, 5 * out), dim = -1)

  rel_in <- concat$reshape_to_input(rel_out)

  expect_equal(lapply(rel_in, dim), lapply(rel_in_true, dim))
  expect_lt(mean(as.array((rel_in[[1]] - rel_in_true[[1]])^2)), 1e-10)
  expect_lt(mean(as.array((rel_in[[2]] - rel_in_true[[2]])^2)), 1e-10)
  expect_lt(mean(as.array((rel_in[[3]] - rel_in_true[[3]])^2)), 1e-10)

  rel_total_true <- rel_out$sum(c(2,3,4))
  rel_total <- torch_stack(lapply(rel_in,
                                  function(x) x$sum(c(2,3,4))), dim = -1)$sum(2)
  expect_lt(mean(as.array((rel_total - rel_total_true)^2)), 1e-8)
})

test_that("Test function reshape_to_input for 2D-CNN", {
  library(torch)

  batch_size <- 10
  axis <- c(10, 5)
  dim_in <- list(c(axis, 1), c(axis, 6))
  dim_out <- c(axis, 10)
  concat_dim <- 4

  concat <- concatenate_layer(dim = concat_dim,
                              dim_in = dim_in,
                              dim_out = dim_out)

  input <- lapply(dim_in, function(x) torch_randn(c(batch_size, x)))
  rel_in_true <- lapply(input, function(x) torch_stack(list(x, 2 * x), dim = -1))

  # Forward pass
  out <- concat(input)
  # Define output relevance
  rel_out <- torch_stack(list(out, 2 * out), dim = -1)

  rel_in <- concat$reshape_to_input(rel_out)

  expect_equal(lapply(rel_in, dim), lapply(rel_in_true, dim))
  expect_lt(mean(as.array((rel_in[[1]] - rel_in_true[[1]])^2)), 1e-10)
  expect_lt(mean(as.array((rel_in[[2]] - rel_in_true[[2]])^2)), 1e-10)

  rel_total_true <- rel_out$sum(c(2,3,4,5))
  rel_total <- torch_stack(lapply(rel_in,
                                  function(x) x$sum(c(2,3,4,5))), dim = -1)$sum(2)
  expect_lt(mean(as.array((rel_total - rel_total_true)^2)), 1e-8)
})
