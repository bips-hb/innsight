
################################################################################
#
# Tests for keras3 model conversion
#
# These tests mirror the keras2 tests in test_Converter.R but use the keras3
# R package. All backends supported by keras3 (TF, JAX, PyTorch) are accepted.
#
################################################################################

skip_if_not_installed("keras3")

library(keras3)
library(torch)

################################################################################
#
# Sequential Models
#
################################################################################

test_that("Test keras3 sequential: Dense", {
  data <- matrix(rnorm(4 * 10), nrow = 10)

  model <- keras_model_sequential(input_shape = c(4))
  model |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dropout(0.1) |>
    layer_dense(units = 8, activation = "relu") |>
    layer_dropout(0.1) |>
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)
  # input dim as vector
  converter <- Converter$new(model, input_dim = c(4))
  # input dim as list
  converter <- Converter$new(model, input_dim = list(4))

  # forward pass
  y_true <- as.array(model(data))
  y <- as_array(converter$model(list(torch_tensor(data)))[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref <- matrix(rnorm(4), nrow = 1, ncol = 4)
  y_ref      <- as_array(converter$model$update_ref(list(torch_tensor(x_ref)))[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

  # input / output dimensions
  expect_equal(converter$input_dim[[1]], 4)
  expect_equal(converter$output_dim[[1]], 3)
})


test_that("Test keras3 sequential: Conv1D with 'valid' padding", {
  data <- array(rnorm(10 * 128 * 4), dim = c(10, 128, 4))

  model <- keras_model_sequential(input_shape = c(128, 4))
  model |>
    layer_conv_1d(
      kernel_size = 16, filters = 8,
      activation = "softplus"
    ) |>
    layer_max_pooling_1d() |>
    layer_conv_1d(kernel_size = 16, filters = 4, activation = "tanh") |>
    layer_zero_padding_1d(padding = c(1, 2)) |>
    layer_average_pooling_1d(pool_size = 2) |>
    layer_conv_1d(kernel_size = 16, filters = 2, activation = "relu") |>
    layer_flatten() |>
    layer_dense(units = 64, activation = "relu") |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")

  converter <- Converter$new(model)
  # input dim as vector (channels first)
  converter <- Converter$new(model, input_dim = c(4, 128))

  # forward pass
  y_true <- as.array(model(data))
  y <- as_array(converter$model(list(torch_tensor(data)),
                                channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref      <- array(rnorm(128 * 4), dim = c(1, 128, 4))
  y_ref      <- as_array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                                    channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref - y_ref_true)^2), 1e-12)

  # input / output dimensions
  expect_equal(converter$input_dim[[1]], c(4, 128))
  expect_equal(converter$output_dim[[1]], 1)
})


test_that("Test keras3 sequential: Conv1D with 'same' padding and BatchNorm", {
  data <- array(rnorm(10 * 128 * 4), dim = c(10, 128, 4))

  model <- keras_model_sequential(input_shape = c(128, 4))
  model |>
    layer_conv_1d(
      kernel_size = 16, filters = 8,
      activation = "softplus", padding = "same"
    ) |>
    layer_batch_normalization() |>
    layer_conv_1d(
      kernel_size = 16, filters = 4, activation = "tanh", padding = "same"
    ) |>
    layer_batch_normalization() |>
    layer_conv_1d(
      kernel_size = 16, filters = 2, activation = "relu", padding = "same"
    ) |>
    layer_flatten() |>
    layer_dense(units = 64, activation = "relu") |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")

  converter <- Converter$new(model)

  # forward pass
  y_true <- as.array(model(data))
  y <- as.array(converter$model(list(torch_tensor(data)),
                                channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref      <- array(rnorm(128 * 4), dim = c(1, 128, 4))
  y_ref      <- as.array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                                    channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

  expect_equal(converter$input_dim[[1]], c(4, 128))
  expect_equal(converter$output_dim[[1]], 1)
})


test_that("Test keras3 sequential: Conv2D with 'valid' padding and BatchNorm", {
  data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))

  model <- keras_model_sequential(input_shape = c(32, 32, 3))
  model |>
    layer_conv_2d(
      kernel_size = 8, filters = 8,
      activation = "softplus", padding = "valid"
    ) |>
    layer_batch_normalization() |>
    layer_max_pooling_2d() |>
    layer_zero_padding_2d(padding = list(c(2, 2), c(5, 3))) |>
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh", padding = "valid"
    ) |>
    layer_average_pooling_2d(pool_size = c(1, 1)) |>
    layer_batch_normalization() |>
    layer_conv_2d(
      kernel_size = 4, filters = 2, activation = "relu", padding = "valid"
    ) |>
    layer_flatten() |>
    layer_dense(units = 64, activation = "relu") |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")

  converter <- Converter$new(model)

  # forward pass
  y_true <- as.array(model(data))
  y <- as.array(converter$model(list(torch_tensor(data)),
                                channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref      <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))
  y_ref      <- as.array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                                    channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  expect_equal(converter$input_dim[[1]], c(3, 32, 32))
  expect_equal(converter$output_dim[[1]], 1)
})


test_that("Test keras3 sequential: Conv2D with 'same' padding", {
  data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))

  model <- keras_model_sequential(input_shape = c(32, 32, 3))
  model |>
    layer_conv_2d(
      kernel_size = 8, filters = 8,
      activation = "softplus", padding = "same"
    ) |>
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh", padding = "same"
    ) |>
    layer_conv_2d(
      kernel_size = 4, filters = 2, activation = "relu", padding = "same"
    ) |>
    layer_flatten() |>
    layer_dense(units = 64, activation = "relu") |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")

  converter <- Converter$new(model)

  # forward pass
  y_true <- as.array(model(data))
  y <- as.array(converter$model(list(torch_tensor(data)),
                                channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update_ref
  x_ref      <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))
  y_ref      <- as.array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                                    channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  expect_equal(converter$input_dim[[1]], c(3, 32, 32))
  expect_equal(converter$output_dim[[1]], 1)
})


################################################################################
#
# Functional Models
#
################################################################################

test_that("Test keras3 functional: single input + single output", {
  main_input <- layer_input(shape = c(10, 10, 2), name = "main_input")
  out <- main_input |>
    layer_conv_2d(2, c(2, 2)) |>
    layer_flatten() |>
    layer_dense(units = 4) |>
    layer_dense(units = 5, activation = "tanh") |>
    layer_dense(units = 4, activation = "tanh") |>
    layer_dense(units = 2, activation = "tanh") |>
    layer_dense(units = 3, activation = "softmax", name = "main_output")
  model <- keras_model(inputs = main_input, outputs = out)

  conv <- Converter$new(model)
  data       <- lapply(list(c(10, 10, 2)),
                       function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_torch <- lapply(data, torch_tensor)

  # forward pass
  y_true <- as.array(model(data[[1]]))
  y <- as_array(conv$model(data_torch, channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update_ref
  x_ref       <- lapply(list(c(10, 10, 2)),
                        function(x) array(rnorm(prod(x)), dim = c(1, x)))
  x_ref_torch <- lapply(x_ref, torch_tensor)
  y_ref       <- as_array(conv$model$update_ref(x_ref_torch,
                                                channels_first = FALSE)[[1]])
  y_ref_true  <- as.array(model(x_ref[[1]]))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)
})


test_that("Test keras3 functional: two inputs + one output", {
  main_input      <- layer_input(shape = c(10, 10, 2), name = "main_input")
  auxiliary_input <- layer_input(shape = c(5), name = "aux_input")

  cnn_out <- main_input |>
    layer_conv_2d(2, c(2, 2)) |>
    layer_flatten() |>
    layer_dense(units = 4)

  main_output <- layer_concatenate(list(cnn_out, auxiliary_input)) |>
    layer_dense(units = 5, activation = "tanh") |>
    layer_dense(units = 4, activation = "tanh") |>
    layer_dense(units = 2, activation = "tanh") |>
    layer_dense(units = 3, activation = "softmax", name = "main_output")

  model <- keras_model(
    inputs  = list(main_input, auxiliary_input),
    outputs = main_output
  )

  conv       <- Converter$new(model)
  data       <- lapply(list(c(10, 10, 2), c(5)),
                       function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_torch <- lapply(data, torch_tensor)

  # forward pass
  y_true <- as.array(model(data))
  y <- as_array(conv$model(data_torch, channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update_ref
  x_ref       <- lapply(list(c(10, 10, 2), c(5)),
                        function(x) array(rnorm(prod(x)), dim = c(1, x)))
  x_ref_torch <- lapply(x_ref, torch_tensor)
  y_ref       <- as_array(conv$model$update_ref(x_ref_torch,
                                                channels_first = FALSE)[[1]])
  y_ref_true  <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)
})


test_that("Test keras3 functional: two inputs + two outputs", {
  main_input      <- layer_input(shape = c(10, 10, 2), name = "main_input")
  auxiliary_input <- layer_input(shape = c(5), name = "aux_input")

  cnn_out <- main_input |>
    layer_conv_2d(2, c(2, 2)) |>
    layer_flatten() |>
    layer_dense(units = 4)

  auxiliary_output <- layer_concatenate(list(cnn_out, auxiliary_input)) |>
    layer_dense(units = 2, activation = "softmax", name = "aux_output")

  main_output <- layer_concatenate(list(cnn_out, auxiliary_input)) |>
    layer_dense(units = 5, activation = "tanh") |>
    layer_dense(units = 4, activation = "tanh") |>
    layer_dense(units = 2, activation = "tanh") |>
    layer_dense(units = 3, activation = "softmax", name = "main_output")

  model <- keras_model(
    inputs  = list(auxiliary_input, main_input),
    outputs = list(auxiliary_output, main_output)
  )

  conv       <- Converter$new(model)
  data       <- lapply(list(c(5), c(10, 10, 2)),
                       function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_torch <- lapply(data, torch_tensor)

  # forward pass
  y_true <- lapply(model(data), as.array)
  y      <- lapply(conv$model(data_torch, channels_first = FALSE), as_array)
  expect_equal(lapply(y, dim), lapply(y_true, dim))
  expect_lt(
    mean(unlist(lapply(seq_along(y),
                       function(i) mean((y_true[[i]] - y[[i]])^2)))),
    1e-12
  )
})
