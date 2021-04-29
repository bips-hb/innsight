context("Test R6 class Dense_Layer")

test_that("Initialization", {

  #-------------positive tests--------------------
  num_rep = 100
  dim_in = sample(1:200, num_rep, replace = TRUE)
  dim_out = sample(1:200, num_rep, replace = TRUE)

  for (i in 1:num_rep) {
    d_in <- dim_in[i]
    d_out <- dim_out[i]

    W <- matrix(rnorm(d_in * d_out), nrow = d_in, ncol = d_out)
    b <- rnorm(d_out)
    relu <- function(x) ifelse(x >= 0, x, 0)

    expect_error(Dense_Layer$new(W, b, relu), NA)
  }

  #-------------negative tests----------------------
  W <- matrix(rnorm(5 * 10), nrow = 5, ncol = 10)
  b <- rnorm(10)

  # for argument bias
  expect_error(Dense_Layer$new(W, W, relu, "relu"))
  expect_error(Dense_Layer$new(W, NA, relu, "relu"))
  expect_error(Dense_Layer$new(W, NULL, relu, "relu"))
  # for argument weights
  expect_error(Dense_Layer$new(b, b, relu, "relu"))
  expect_error(Dense_Layer$new(NA, b, relu, "relu"))
  expect_error(Dense_Layer$new(NULL, b, relu, "relu"))
})


test_that("Forward method",{
  #---------------positive tests-----------------------
  num_rep = 100
  dim_in = sample(1:200, num_rep, replace = TRUE)
  dim_out = sample(1:200, num_rep, replace = TRUE)
  batch_size =  sample(1:50, num_rep, replace = TRUE)

  for (i in 1:num_rep) {
    d_in <- dim_in[i]
    d_out <- dim_out[i]
    b_size <- batch_size[i]+1
    W <- matrix(rnorm(d_in * d_out), nrow = d_in, ncol = d_out)
    b <- rnorm(d_out)
    relu <- function(x) ifelse(x >= 0, x, 0)

    inputs <- matrix(rnorm(b_size * d_in), ncol = d_in, nrow = b_size)
    inputs_ref <- rnorm(d_in)
    layer <- Dense_Layer$new(W, b, relu, "relu")

    # forward works properly
    expect_error(layer$forward(inputs), NA)
    out <-layer$forward(inputs)

    # output is list
    expect_equal(is.list(out), TRUE)
    # output is list of length 2
    expect_equal(length(out), 2)
    # reference output is NULL
    expect_equal(out[[2]], NULL)
    # output is matrix
    expect_equal(is.matrix(out[[1]]), TRUE)
    # inputs are stored correctly
    expect_equal(layer$inputs, inputs, ignore_attr = TRUE)
    # preactivations are stored correctly
    expect_equal(layer$preactivation, t(t(inputs %*% W) + b), ignore_attr = TRUE)
    # outputs are stored correctly
    expect_equal(layer$outputs, relu(t(t(inputs %*% W) + b)), ignore_attr = TRUE)

    # forward works properly for an additional reference value
    expect_error(layer$forward(inputs, inputs_ref), NA)
    out <-layer$forward(inputs, inputs_ref)

    # output is list
    expect_equal(is.list(out), TRUE)
    # output is list of length 2
    expect_equal(length(out), 2)
    # output is a matrix
    expect_equal(is.matrix(out[[1]]), TRUE)
    # reference output is a vector
    expect_equal(is.vector(out[[2]]), TRUE)
    # inputs are stored correctly
    expect_equal(layer$inputs, inputs, ignore_attr = TRUE)
    # reference inputs are stored correctly
    expect_equal(layer$inputs_ref, inputs_ref)
    # preactivations are stored correctly
    expect_equal(layer$preactivation, t(t(inputs %*% W) + b), ignore_attr = TRUE)
    # reference preactivations are stored correctly
    expect_equal(layer$preactivation_ref, as.vector(t(W) %*% inputs_ref + b))
    # outputs are stored correctly
    expect_equal(layer$outputs, relu(t(t(inputs %*% W) + b)), ignore_attr = TRUE)
    # reference outputs are stored correctly
    expect_equal(layer$outputs_ref, relu(as.vector(t(W) %*% inputs_ref + b)))

  }

  #--------------------negative tests--------------------------------
  W <- matrix(rnorm(5 * 10), nrow = 5, ncol = 10)
  b <- rnorm(10)
  inputs <- matrix(rnorm(3*5), nrow = 3, ncol = 5)
  inputs_ref <- rnorm(5)
  layer <- Dense_Layer$new(W, b, relu, "relu")

  expect_error(layer$forward(NULL, inputs_ref))
  expect_error(layer$forward(NA, inputs_ref))
  expect_error(layer$forward(matrix(rnorm(8), 2, 4), inputs_ref))
  expect_error(layer$forward(c(1,2,3), inputs_ref))

  expect_error(layer$forward(inputs, NA))
  expect_error(layer$forward(inputs, matrix(rnorm(10), 2, 5), inputs_ref))
  expect_error(layer$forward(inputs, c(1,2,3)))
})
