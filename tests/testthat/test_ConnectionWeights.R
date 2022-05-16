
test_that("ConnectionWeights: General errors", {
  library(keras)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  expect_error(ConnectionWeights$new(model))
  expect_error(ConnectionWeights$new(converter, channels_first = NULL))
  expect_error(ConnectionWeights$new(converter, dtype = "asdf"))

  cw <- ConnectionWeights$new(converter, output_idx = c(1))

  # Test method 'get_results'
  res_array <- cw$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- cw$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- cw$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
  expect_error(cw$get_result(type = "adsf"))

  # Test plot function
  expect_error(plot(cw, output_idx = c(1,5)))
  expect_error(plot(cw, aggr_channels = "x^3"))
  expect_error(plot(cw, preprocess_FUN = "sum"))
  expect_error(plot(cw, as_plotly = NULL))
})


test_that("ConnectionWeights: Dense-Net", {
  library(keras)
  library(torch)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  cw <- ConnectionWeights$new(converter)
  expect_equal(dim(cw$get_result()), c(4, 3))
  expect_true(cw$get_result(type = "torch.tensor")$dtype == torch_float())

  cw_channels_last <- ConnectionWeights$new(converter, channels_first = FALSE)
  expect_equal(dim(cw_channels_last$get_result()), c(4, 3))
  expect_true(
    cw_channels_last$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  cw_double <- ConnectionWeights$new(converter, dtype = "double")
  expect_equal(dim(cw$get_result()), c(4, 3))
  expect_true(
    cw_double$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  cw_channels_last_double <-
    ConnectionWeights$new(converter, channels_first = FALSE, dtype = "double")
  expect_equal(dim(cw_channels_last_double$get_result()), c(4, 3))
  expect_true(
    cw_channels_last_double$get_result(type = "torch.tensor")$dtype ==
      torch_double()
  )

  # Test plot function
  p <- plot(cw)
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, output_idx = c(1,2))
  expect_true(inherits(p, "ggplot"))

  skip_if_not_installed("plotly")
  p <- plot(cw, as_plotly = TRUE)
  expect_true(inherits(p, "plotly"))
})

test_that("ConnectionWeights: Conv1D-Net", {
  library(keras)
  library(torch)

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(64, 3), kernel_size = 16, filters = 8,
      activation = "softplus"
    ) %>%
    layer_conv_1d(kernel_size = 16, filters = 4, activation = "tanh") %>%
    layer_conv_1d(kernel_size = 16, filters = 2, activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  converter <- Converter$new(model)

  cw <- ConnectionWeights$new(converter)
  expect_equal(dim(cw$get_result()), c(3, 64, 1))
  expect_true(cw$get_result(type = "torch.tensor")$dtype == torch_float())

  cw_channels_last <- ConnectionWeights$new(converter, channels_first = FALSE)
  expect_equal(dim(cw_channels_last$get_result()), c(64, 3, 1))
  expect_true(
    cw_channels_last$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  cw_double <- ConnectionWeights$new(converter, dtype = "double")
  expect_equal(dim(cw$get_result()), c(3, 64, 1))
  expect_true(
    cw_double$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  cw_channels_last_double <-
    ConnectionWeights$new(converter, channels_first = FALSE, dtype = "double")
  expect_equal(dim(cw_channels_last_double$get_result()), c(64, 3, 1))
  expect_true(
    cw_channels_last_double$get_result(type = "torch.tensor")$dtype ==
      torch_double()
  )

  # Test get_dataframe
  res_dataframe <- cw$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_dataframe <- cw_channels_last$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))

  # Test plot function
  p <- plot(cw)
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw_channels_last)
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, output_idx = c(1))
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, aggr_channels = "sum")
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, aggr_channels = "mean")
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, aggr_channels = "norm")
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, aggr_channels = mean)
  expect_true(inherits(p, "ggplot"))

  skip_if_not_installed("plotly")
  p <- plot(cw, as_plotly = TRUE)
  expect_true(inherits(p, "plotly"))
})

test_that("ConnectionWeights: Conv2D-Net", {
  library(keras)
  library(torch)

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
      activation = "softplus", padding = "same"
    ) %>%
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh",
      padding = "same"
    ) %>%
    layer_conv_2d(
      kernel_size = 4, filters = 2, activation = "relu",
      padding = "same"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 2, activation = "sigmoid")

  converter <- Converter$new(model)

  cw <- ConnectionWeights$new(converter)
  expect_equal(dim(cw$get_result()), c(3, 32, 32, 2))
  expect_true(cw$get_result(type = "torch.tensor")$dtype == torch_float())

  cw_channels_last <- ConnectionWeights$new(converter, channels_first = FALSE)
  expect_equal(dim(cw_channels_last$get_result()), c(32, 32, 3, 2))
  expect_true(
    cw_channels_last$get_result(type = "torch.tensor")$dtype == torch_float()
  )

  cw_double <- ConnectionWeights$new(converter, dtype = "double")
  expect_equal(dim(cw$get_result()), c(3, 32, 32, 2))
  expect_true(
    cw_double$get_result(type = "torch.tensor")$dtype == torch_double()
  )

  cw_channels_last_double <-
    ConnectionWeights$new(converter, channels_first = FALSE, dtype = "double")
  expect_equal(dim(cw_channels_last_double$get_result()), c(32, 32, 3, 2))
  expect_true(
    cw_channels_last_double$get_result(type = "torch.tensor")$dtype ==
      torch_double()
  )

  # Test get_dataframe
  res_dataframe <- cw$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_dataframe <- cw_channels_last$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))

  # Test plot function
  p <- plot(cw)
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw_channels_last)
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, output_idx = c(1))
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, aggr_channels = "sum")
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, aggr_channels = "mean")
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, aggr_channels = "norm")
  expect_true(inherits(p, "ggplot"))
  p <- plot(cw, aggr_channels = mean)
  expect_true(inherits(p, "ggplot"))

  skip_if_not_installed("plotly")
  p <- plot(cw, as_plotly = TRUE)
  expect_true(inherits(p, "plotly"))
})
