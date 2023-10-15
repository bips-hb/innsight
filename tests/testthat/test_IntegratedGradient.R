
test_that("IntegratedGradient: General errors", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  expect_error(IntegratedGradient$new(model, data))
  expect_error(IntegratedGradient$new(converter, model))
  expect_error(IntegratedGradient$new(converter, data, channels_first = NULL))
  expect_error(IntegratedGradient$new(converter, data, times_input = "asdf"))
  expect_error(IntegratedGradient$new(converter, data, x_ref = "asdf"))
  expect_error(IntegratedGradient$new(converter, data, n = "asdf"))
  expect_error(IntegratedGradient$new(converter, data, dtype = NULL))
  expect_error(IntegratedGradient$new(converter, data, ignore_last_act = c(1)))
})

test_that("IntegratedGradient: Plot and Boxplot", {
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

  ig <- IntegratedGradient$new(converter, data,
                               dtype = "double",
                               ignore_last_act = FALSE
  )

  # ggplot2

  # Non-existing data points
  expect_error(plot(ig, data_idx = c(1,11)))
  expect_error(boxplot(ig, data_idx = 1:11))
  # Non-existing class
  expect_error(plot(ig, output_idx = c(5)))
  expect_error(boxplot(ig, output_idx = c(5)))

  p <- plot(ig)
  boxp <- boxplot(ig)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(ig, data_idx = 1:3)
  boxp <- boxplot(ig, data_idx = 1:4)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(ig, data_idx = 1:3, output_idx = 1:3)
  boxp <- boxplot(ig, data_idx = 1:5, output_idx = 1:3)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # plotly
  library(plotly)

  p <- plot(ig, as_plotly = TRUE)
  boxp <- boxplot(ig, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(ig, data_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(ig, data_idx = 1:4, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(ig, data_idx = 1:3, output_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(ig, data_idx = 1:5, output_idx = 1:3, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
})

test_that("IntegratedGradient: Dense-Net (Neuralnet)", {
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
  x_ref <- matrix(rnorm(4), nrow = 1)

  # ignore last activation
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               ignore_last_act = FALSE)

  int_grad <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad), c(10, 4, 3))

  # include last activation
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               ignore_last_act = TRUE)
  int_grad_no_last_act <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad_no_last_act), c(10, 4, 3))

  # not times input
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               times_input = FALSE,
                               ignore_last_act = TRUE)
  int_grad_no_times_input <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad_no_times_input), c(10, 4, 3))
})

test_that("IntegratedGradient: Dense-Net (keras)", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)
  x_ref <- matrix(rnorm(4), nrow = 1)

  # ignore last activation
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               ignore_last_act = FALSE)

  int_grad <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad), c(10, 4, 3))

  # not times input
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               times_input = FALSE,
                               ignore_last_act = TRUE)
  int_grad_no_times_input <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad_no_times_input), c(10, 4, 3))
})

test_that("IntegratedGradient: Conv1D-Net", {
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

  # test non-fitted model
  converter <- Converter$new(model)
  x_ref <- array(rnorm(64 * 3), dim = c(1, 64, 3))

  # ignore last activation
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               channels_first = FALSE,
                               ignore_last_act = FALSE)

  int_grad <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad), c(4, 64, 3, 1))

  # not times input
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               times_input = FALSE,
                               channels_first = FALSE,
                               ignore_last_act = TRUE)
  int_grad_no_times_input <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad_no_times_input), c(4, 64, 3, 1))
})

test_that("IntegratedGradient: Conv2D-Net", {
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
    layer_dense(units = 2, activation = "softmax")

  # test non-fitted model
  converter <- Converter$new(model)
  x_ref <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))

  # ignore last activation
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               channels_first = FALSE,
                               ignore_last_act = FALSE)

  int_grad <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad), c(4, 32, 32, 3, 2))

  # not times input
  ig <- IntegratedGradient$new(converter, data,
                               x_ref = x_ref,
                               times_input = FALSE,
                               channels_first = FALSE,
                               ignore_last_act = TRUE)
  int_grad_no_times_input <- ig$get_result(type = "torch.tensor")
  expect_equal(dim(int_grad_no_times_input), c(4, 32, 32, 3, 2))
})



test_that("IntegratedGradient: Keras model with two inputs + two outputs (concat)", {
  library(keras)

  main_input <- layer_input(shape = c(10,10,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2), activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 4)
  auxiliary_input <- layer_input(shape = c(5), name = 'aux_input')
  auxiliary_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 2, activation = 'relu', name = 'aux_output')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'relu') %>%
    layer_dense(units = 3, activation = 'tanh', name = 'main_output')
  model <- keras_model(
    inputs = c(auxiliary_input, main_input),
    outputs = c(auxiliary_output, main_output)
  )

  converter <- Converter$new(model)

  # Check IntegratedGradient with ignoring last activation
  data <- lapply(list(c(5), c(10,10,2)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  x_ref <- lapply(list(c(5), c(10,10,2)),
                  function(x) array(rnorm(10 * prod(x)), dim = c(1, x)))

  int_grad <- IntegratedGradient$new(converter, data, x_ref = x_ref,
                                     channels_first = FALSE, output_idx = list(c(2), c(1,3)))
  result <- int_grad$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,10,10,2,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,10,10,2,2))

  # Check IntegratedGradient without times_input and ignoring last activation
  data <- lapply(list(c(5), c(10,10,2)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  x_ref <- lapply(list(c(5), c(10,10,2)),
                  function(x) array(rnorm(10 * prod(x)), dim = c(1, x)))
  int_grad <- IntegratedGradient$new(converter, data, x_ref = x_ref,
                                     channels_first = FALSE,
                                     times_input = FALSE,
                                     output_idx = list(c(1), c(1,2)))
  result <- int_grad$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,10,10,2,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,10,10,2,2))
})


test_that("IntegratedGradient: Keras model with three inputs + one output (add)", {
  library(keras)

  input_1 <- layer_input(shape = c(12,15,3))
  part_1 <- input_1 %>%
    layer_conv_2d(3, c(4,4), activation = "relu", use_bias = FALSE) %>%
    layer_conv_2d(2, c(3,3), activation = "relu", use_bias = FALSE) %>%
    layer_flatten() %>%
    layer_dense(20, activation = "relu", use_bias = FALSE)
  input_2 <- layer_input(shape = c(10))
  part_2 <- input_2 %>%
    layer_dense(50, activation = "tanh", use_bias = FALSE)
  input_3 <- layer_input(shape = c(20))
  part_3 <- input_3 %>%
    layer_dense(40, activation = "relu", use_bias = FALSE)

  output <- layer_concatenate(c(part_1, part_3, part_2)) %>%
    layer_dense(100, activation = "relu", use_bias = FALSE) %>%
    layer_dense(1, activation = "linear", use_bias = FALSE)

  model <- keras_model(
    inputs = c(input_1, input_3, input_2),
    outputs = output
  )

  converter <- Converter$new(model)

  # Check IntegratedGradient with ignoring last activation
  data <- lapply(list(c(12,15,3), c(20), c(10)),
                 function(x) torch_randn(c(10,x)))
  x_ref <- lapply(list(c(12,15,3), c(20), c(10)),
                  function(x) torch_randn(c(1,x)))

  int_grad <- IntegratedGradient$new(converter, data, x_ref = x_ref,
                                     channels_first = FALSE)
  result <- int_grad$get_result()
  expect_equal(length(result), 3)
  expect_equal(dim(result[[1]]), c(10,12,15,3,1))
  expect_equal(dim(result[[2]]), c(10,20,1))
  expect_equal(dim(result[[3]]), c(10,10,1))

  # Check IntegratedGradient without times_input and ignoring last activation
  data <- lapply(list(c(12,15,3), c(20), c(10)),
                 function(x) torch_randn(c(10,x)))
  x_ref <- lapply(list(c(12,15,3), c(20), c(10)),
                  function(x) torch_randn(c(1,x)))

  int_grad <- IntegratedGradient$new(converter, data, x_ref = x_ref,
                                     channels_first = FALSE,
                                     times_input = FALSE)
  result <- int_grad$get_result()
  expect_equal(length(result), 3)
  expect_equal(dim(result[[1]]), c(10,12,15,3,1))
  expect_equal(dim(result[[2]]), c(10,20,1))
  expect_equal(dim(result[[3]]), c(10,10,1))
})

