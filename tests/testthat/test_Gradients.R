
test_that("Gradient: Plot and Boxplot", {
  library(neuralnet)
  library(torch)

  data(iris)
  data <- iris[sample.int(150, size = 10), -5]
  nn <- neuralnet(Species ~ .,
                  iris,
                  linear.output = FALSE,
                  hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
  )
  # create an converter for this model
  converter <- Converter$new(nn)

  # Rescale Rule
  grad <- Gradient$new(converter, data,
                    dtype = "double",
  )

  # ggplot2

  # Non-existing data points
  expect_error(plot(grad, datapoint = c(1,11)))
  expect_error(boxplot(grad, boxplot_data = 1:11))
  # Non-existing class
  expect_error(plot(grad, output_idx = c(5)))
  expect_error(boxplot(grad, classes = c(5)))

  p <- plot(grad)
  boxp <- boxplot(grad)
  expect_true("ggplot" %in% class(p))
  expect_true("ggplot" %in% class(boxp))
  p <- plot(grad, datapoint = 1:3)
  boxp <- boxplot(grad, boxplot_data = 1:4)
  expect_true("ggplot" %in% class(p))
  expect_true("ggplot" %in% class(boxp))
  p <- plot(grad, datapoint = 1:3, output_idx = 1:3)
  boxp <- boxplot(grad, boxplot_data = 1:5, classes = 1:3)
  expect_true("ggplot" %in% class(p))
  expect_true("ggplot" %in% class(boxp))

  # plotly
  library(plotly)

  p <- plot(grad, as_plotly = TRUE)
  boxp <- boxplot(grad, as_plotly = TRUE)
  expect_true("plotly" %in% class(p))
  expect_true("plotly" %in% class(boxp))
  p <- plot(grad, datapoint = 1:3, as_plotly = TRUE)
  boxp <- boxplot(grad, boxplot_data = 1:4, as_plotly = TRUE)
  expect_true("plotly" %in% class(p))
  expect_true("plotly" %in% class(boxp))
  p <- plot(grad, datapoint = 1:3, output_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(grad, boxplot_data = 1:5, classes = 1:3, as_plotly = TRUE)
  expect_true("plotly" %in% class(p))
  expect_true("plotly" %in% class(boxp))

})


test_that("Gradient: Dense-Net (Neuralnet)", {
  library(neuralnet)
  library(torch)

  data(iris)
  data <- iris[sample.int(150, size = 10), -5]
  nn <- neuralnet(Species ~ .,
                  iris,
                  linear.output = FALSE,
                  hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
  )
  # create an converter for this model
  converter <- Converter$new(nn)

  grad <- Gradient$new(converter, data)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- Gradient$new(converter, data, dtype = "double")
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- Gradient$new(converter, data, times_input = FALSE)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- Gradient$new(converter, data, ignore_last_act = FALSE)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))
})


test_that("Gradient: Dense-Net (keras)", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  grad <- Gradient$new(converter, data)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- Gradient$new(converter, data, dtype = "double")
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- Gradient$new(converter, data, times_input = FALSE)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- Gradient$new(converter, data, ignore_last_act = FALSE)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))
})

test_that("SmoothGrad: Dense-Net", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  grad <- SmoothGrad$new(converter, data)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- SmoothGrad$new(converter, data, dtype = "double")
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- SmoothGrad$new(converter, data, times_input = FALSE)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- SmoothGrad$new(converter, data, ignore_last_act = FALSE)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- SmoothGrad$new(converter, data, n = 5)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))

  grad <- SmoothGrad$new(converter, data, noise_level = 1.5)
  expect_equal(dim(grad$get_result()), c(10, 4, 3))
})

test_that("Gradient: Conv1D-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(4 * 64 * 3), dim = c(4, 64, 3))

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

  grad <- Gradient$new(converter, data, channels_first = FALSE)
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))

  grad <- Gradient$new(converter, data,
    dtype = "double",
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))

  grad <- Gradient$new(converter, data,
    times_input = FALSE,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))

  grad <- Gradient$new(converter, data,
    ignore_last_act = FALSE,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))
})


test_that("SmoothGrad: Conv1D-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(4 * 64 * 3), dim = c(4, 64, 3))

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

  grad <- SmoothGrad$new(converter, data, channels_first = FALSE)
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))

  grad <- SmoothGrad$new(converter, data,
    dtype = "double",
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))

  grad <- SmoothGrad$new(converter, data,
    times_input = FALSE,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))

  grad <- SmoothGrad$new(converter, data,
    ignore_last_act = FALSE,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))

  grad <- SmoothGrad$new(converter, data, n = 5, channels_first = FALSE)
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))

  grad <- SmoothGrad$new(converter, data,
    noise_level = 1.5,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 64, 3, 1))
})


test_that("Gradient: Conv2D-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(4 * 32 * 32 * 3), dim = c(4, 32, 32, 3))

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

  grad <- Gradient$new(converter, data, channels_first = FALSE)
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))

  grad <- Gradient$new(converter, data,
    dtype = "double",
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))

  grad <- Gradient$new(converter, data,
    times_input = FALSE,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))

  grad <- Gradient$new(converter, data,
    ignore_last_act = FALSE,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))
})

test_that("SmoothGrad: Conv2D-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(4 * 32 * 32 * 3), dim = c(4, 32, 32, 3))

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

  grad <- SmoothGrad$new(converter, data, channels_first = FALSE)
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))

  grad <- SmoothGrad$new(converter, data,
    dtype = "double",
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))

  grad <- SmoothGrad$new(converter, data,
    times_input = FALSE,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))

  grad <- SmoothGrad$new(converter, data,
    ignore_last_act = FALSE,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))

  grad <- SmoothGrad$new(converter, data, n = 5, channels_first = FALSE)
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))

  grad <- SmoothGrad$new(converter, data,
    noise_level = 1.5,
    channels_first = FALSE
  )
  expect_equal(dim(grad$get_result()), c(4, 32, 32, 3, 2))
})


test_that("LRP: Correctness", {
  library(keras)
  library(torch)

  data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
      activation = "softplus", padding = "valid"
    ) %>%
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh",
      padding = "same"
    ) %>%
    layer_conv_2d(
      kernel_size = 4, filters = 2, activation = "relu",
      padding = "valid"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)

  grad <- Gradient$new(converter, data,
    channels_first = FALSE,
    times_input = FALSE
  )

  smooth_grad <- SmoothGrad$new(converter, data,
    channels_first = FALSE,
    times_input = FALSE,
    noise_level = 1e-8
  )

  result_grad <- grad$get_result(type = "torch.tensor")
  result_smoothgrad <- smooth_grad$get_result(type = "torch.tensor")

  expect_lt(as.array(mean(abs(sum(result_grad - result_smoothgrad,
    dim = c(2, 3, 4, 5)
  ))^2)), 1e-8)
})
