

test_that("ConnectionWeights: General errors", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  expect_message(ConnectionWeights$new(converter, data = data))
  expect_error(ConnectionWeights$new(model))
  expect_error(ConnectionWeights$new(converter, channels_first = NULL))
  expect_error(ConnectionWeights$new(converter, times_input = NULL))
  expect_error(ConnectionWeights$new(converter, dtype = "asdf"))
  expect_error(ConnectionWeights$new(converter, times_input = TRUE))
  expect_error(ConnectionWeights$new(converter, times_input = TRUE, data = "asdf"))
})

###############################################################################
#                       ConnectionWeights (global)
###############################################################################

test_that("ConnectionWeights (global): Dense-Net", {
  library(keras)
  library(torch)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  # Channels first and float
  cw_first <- ConnectionWeights$new(converter)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(1,4,3))
  expect_true(result$dtype == torch_float())

  # Channels last and float
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(1,4,3))
  expect_true(result$dtype == torch_float())

  # Channels first and double
  cw_first <- ConnectionWeights$new(converter, dtype = "double")
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(1,4,3))
  expect_true(result$dtype == torch_double())

  # Channels last and double
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE, dtype = "double")
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(1,4,3))
  expect_true(result$dtype == torch_double())

  # get_result method
  result <- cw_first$get_result()
  expect_equal(dim(result), c(1,4,3))
  expect_equal(dimnames(result),
               list(NULL, c("X1", "X2", "X3", "X4"), c("Y1", "Y2", "Y3")))
  result <- cw_first$get_result("torch.tensor")
  expect_equal(dim(result), c(1,4,3))
  result <- cw_first$get_result("data.frame")
  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 12)

  # Test plot function with channels first
  p <- plot(cw_first)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, output_idx = c(1,2))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_first, as_plotly = TRUE, output_idx = c(1,2))
  expect_s4_class(p, "innsight_plotly")
  expect_error(boxplot(cw_first))
  expect_message(plot(cw_first, data_idx = c(1,3)))

  # Test plot function with channels last
  p <- plot(cw_last)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, output_idx = c(1,2))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_last, as_plotly = TRUE, output_idx = c(1,2))
  expect_s4_class(p, "innsight_plotly")
  expect_error(boxplot(cw_last))
  expect_message(plot(cw_last, data_idx = c(1,3)))
})

test_that("ConnectionWeights (global): Conv1D-Net", {
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
    layer_dense(units = 4, activation = "sigmoid")

  converter <- Converter$new(model)

  # Channels first and float
  cw_first <- ConnectionWeights$new(converter)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(1,3,64,4))
  expect_true(result$dtype == torch_float())

  # Channels last and float
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(1,64,3,4))
  expect_true(result$dtype == torch_float())

  # Channels first and double
  cw_first <- ConnectionWeights$new(converter, dtype = "double")
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(1,3,64,4))
  expect_true(result$dtype == torch_double())

  # Channels last and double
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE, dtype = "double")
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(1,64,3,4))
  expect_true(result$dtype == torch_double())

  # get_result method
  result <- cw_first$get_result()
  expect_equal(dim(result), c(1,3,64,4))
  result <- cw_first$get_result("torch.tensor")
  expect_equal(dim(result), c(1,3,64,4))
  result <- cw_first$get_result("data.frame")
  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 3*64*4)

  # Test plot function with channels first
  p <- plot(cw_first)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, output_idx = c(1,3))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_first, as_plotly = TRUE, output_idx = c(1,2))
  expect_s4_class(p, "innsight_plotly")
  expect_error(boxplot(cw_first))
  expect_message(plot(cw_first, data_idx = c(1,3)))

  # Test plot function with channels last
  p <- plot(cw_last)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, output_idx = c(1,2))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_last, as_plotly = TRUE, output_idx = c(1,2))
  expect_s4_class(p, "innsight_plotly")
  expect_error(boxplot(cw_last))
  expect_message(plot(cw_last, data_idx = c(1,3)))
})

test_that("ConnectionWeights (global): Conv2D-Net", {
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
    layer_dense(units = 5, activation = "sigmoid")

  converter <- Converter$new(model)

  # Channels first and float
  cw_first <- ConnectionWeights$new(converter)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(1,3,32,32,5))
  expect_true(result$dtype == torch_float())

  # Channels last and float
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(1,32,32,3,5))
  expect_true(result$dtype == torch_float())

  # Channels first and double
  cw_first <- ConnectionWeights$new(converter, dtype = "double")
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(1,3,32,32,5))
  expect_true(result$dtype == torch_double())

  # Channels last and double
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE, dtype = "double")
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(1,32,32,3,5))
  expect_true(result$dtype == torch_double())

  # get_result method
  result <- cw_first$get_result()
  expect_equal(dim(result), c(1,3,32,32,5))
  result <- cw_first$get_result("torch.tensor")
  expect_equal(dim(result), c(1,3,32,32,5))
  result <- cw_first$get_result("data.frame")
  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 3*32*32*5)

  # Test plot function with channels first
  p <- plot(cw_first)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, output_idx = c(1,3))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_first, as_plotly = TRUE, output_idx = c(1,2))
  expect_s4_class(p, "innsight_plotly")
  expect_error(plot_global(cw_first))
  expect_message(plot(cw_first, data_idx = c(1,3)))

  # Test plot function with channels last
  p <- plot(cw_last)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, output_idx = c(1,2))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_last, as_plotly = TRUE, output_idx = c(1,2))
  expect_s4_class(p, "innsight_plotly")
  expect_error(plot_global(cw_last))
  expect_message(plot(cw_last, data_idx = c(1,3)))
})



test_that("ConnectionWeights (global): Keras model with two inputs + two outputs", {
  library(keras)

  main_input <- layer_input(shape = c(10,10,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2)) %>%
    layer_flatten() %>%
    layer_dense(units = 4)
  auxiliary_input <- layer_input(shape = c(5), name = 'aux_input')
  auxiliary_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 2, activation = 'softmax', name = 'aux_output')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 4, activation = 'tanh') %>%
    layer_dense(units = 2, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'sigmoid', name = 'main_output')
  model <- keras_model(
    inputs = c(auxiliary_input, main_input),
    outputs = c(auxiliary_output, main_output)
  )

  converter <- Converter$new(model)

  # Channels first
  cw_first <- ConnectionWeights$new(converter, output_idx = list(c(2), c(1,3)))
  result <- cw_first$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(1,5,1))
  expect_equal(dim(result[[1]][[2]]), c(1,2,10,10,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(1,5,2))
  expect_equal(dim(result[[2]][[2]]), c(1,2,10,10,2))

  # Channels last
  cw_last <- ConnectionWeights$new(converter, output_idx = list(c(2), c(1,3)),
                                    channels_first = FALSE)
  result <- cw_last$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(1,5,1))
  expect_equal(dim(result[[1]][[2]]), c(1,10,10,2,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(1,5,2))
  expect_equal(dim(result[[2]][[2]]), c(1,10,10,2,2))

  # get_result method
  result <- cw_first$get_result("torch.tensor")
  expect_equal(dim(result[[1]][[1]]), c(1,5,1))
  expect_equal(dim(result[[1]][[2]]), c(1,2,10,10,1))
  expect_equal(dim(result[[2]][[1]]), c(1,5,2))
  expect_equal(dim(result[[2]][[2]]), c(1,2,10,10,2))
  result <- cw_first$get_result("data.frame")
  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 5 + 2*10*10 + 2*5 + 2*10*10*2)

  # Test plot function with channels first
  p <- plot(cw_first)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, output_idx = list(c(2), c(1)))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_first, as_plotly = TRUE, output_idx = list(c(2), c(1)))
  expect_s4_class(p, "innsight_plotly")
  expect_error(plot_global(cw_first))
  expect_message(plot(cw_first, data_idx = c(1,3)))

  # Test plot function with channels last
  p <- plot(cw_last)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, output_idx = list(c(2), c(1)))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_last, as_plotly = TRUE, output_idx = list(c(2), c(1)))
  expect_s4_class(p, "innsight_plotly")
  expect_error(plot_global(cw_last))
  expect_message(plot(cw_last, data_idx = c(1,3)))

})


###############################################################################
#                       ConnectionWeights (local)
###############################################################################

test_that("ConnectionWeights (local): Dense-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(10*4), dim = c(10,4))
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  # Channels first and float
  cw_first <- ConnectionWeights$new(converter, data = data, times_input = TRUE)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(10,4,3))
  expect_true(result$dtype == torch_float())

  # Channels last and float
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE,
                                   data = data, times_input = TRUE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(10,4,3))
  expect_true(result$dtype == torch_float())

  # Channels first and double
  cw_first <- ConnectionWeights$new(converter, dtype = "double",
                                    data = data, times_input = TRUE)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(10,4,3))
  expect_true(result$dtype == torch_double())

  # Channels last and double
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE,
                                   dtype = "double", data = data,
                                   times_input = TRUE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(10,4,3))
  expect_true(result$dtype == torch_double())

  # get_result method
  result <- cw_first$get_result()
  expect_equal(dim(result), c(10,4,3))
  result <- cw_first$get_result("torch.tensor")
  expect_equal(dim(result), c(10,4,3))
  result <- cw_first$get_result("data.frame")
  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 10*4*3)

  # Test plot function with channels first
  p <- plot(cw_first)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_first, as_plotly = TRUE, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_plotly")

  # Test plot function with channels last
  p <- plot(cw_last)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_last, as_plotly = TRUE, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_plotly")

  # Test boxplot
  box <- boxplot(cw_first)
  expect_s4_class(box, "innsight_ggplot2")
  box <- boxplot(cw_first, as_plotly = TRUE)
  expect_s4_class(box, "innsight_plotly")
})

test_that("ConnectionWeights (local): Conv1D-Net", {
  library(keras)
  library(torch)

  data_first <- array(rnorm(10*64*3), dim = c(10,3,64))
  data_last <- array(rnorm(10*64*3), dim = c(10,64,3))
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
    layer_dense(units = 4, activation = "sigmoid")

  converter <- Converter$new(model)

  # Channels first and float
  cw_first <- ConnectionWeights$new(converter, data = data_first, times_input = TRUE)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(10,3, 64,4))
  expect_true(result$dtype == torch_float())

  # Channels last and float
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE,
                                   data = data_last, times_input = TRUE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(10,64,3,4))
  expect_true(result$dtype == torch_float())

  # Channels first and double
  cw_first <- ConnectionWeights$new(converter, dtype = "double",
                                    data = data_first, times_input = TRUE)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(10,3, 64,4))
  expect_true(result$dtype == torch_double())

  # Channels last and double
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE,
                                   dtype = "double", data = data_last,
                                   times_input = TRUE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(10,64,3,4))
  expect_true(result$dtype == torch_double())

  # get_result method
  result <- cw_first$get_result()
  expect_equal(dim(result), c(10,3,64,4))
  result <- cw_first$get_result("torch.tensor")
  expect_equal(dim(result), c(10,3,64,4))
  result <- cw_first$get_result("data.frame")
  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 10*3*64*4)

  # Test plot function with channels first
  p <- plot(cw_first)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_first, as_plotly = TRUE, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_plotly")

  # Test plot function with channels last
  p <- plot(cw_last)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_last, as_plotly = TRUE, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_plotly")

  # Test boxplot
  box <- boxplot(cw_first)
  expect_s4_class(box, "innsight_ggplot2")
  box <- boxplot(cw_first, as_plotly = TRUE)
  expect_s4_class(box, "innsight_plotly")
})

test_that("ConnectionWeights (local): Conv2D-Net", {
  library(keras)
  library(torch)

  data_first <- array(rnorm(10*32*32*3), dim = c(10,3,32,32))
  data_last <- array(rnorm(10*32*32*3), dim = c(10,32,32,3))
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
    layer_dense(units = 5, activation = "sigmoid")

  converter <- Converter$new(model)

  # Channels first and float
  cw_first <- ConnectionWeights$new(converter, data = data_first, times_input = TRUE)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(10,3, 32,32,5))
  expect_true(result$dtype == torch_float())

  # Channels last and float
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE,
                                   data = data_last, times_input = TRUE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(10,32,32,3,5))
  expect_true(result$dtype == torch_float())

  # Channels first and double
  cw_first <- ConnectionWeights$new(converter, dtype = "double",
                                    data = data_first, times_input = TRUE)
  result <- cw_first$result[[1]][[1]]
  expect_equal(result$shape, c(10,3, 32,32,5))
  expect_true(result$dtype == torch_double())

  # Channels last and double
  cw_last <- ConnectionWeights$new(converter, channels_first = FALSE,
                                   dtype = "double", data = data_last,
                                   times_input = TRUE)
  result <- cw_last$result[[1]][[1]]
  expect_equal(result$shape, c(10,32,32,3,5))
  expect_true(result$dtype == torch_double())

  # get_result method
  result <- cw_first$get_result()
  expect_equal(dim(result), c(10,3,32,32,5))
  result <- cw_first$get_result("torch.tensor")
  expect_equal(dim(result), c(10,3,32,32,5))
  result <- cw_first$get_result("data.frame")
  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 10*3*32*32*5)

  # Test plot function with channels first
  p <- plot(cw_first)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_first, as_plotly = TRUE, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_plotly")

  # Test plot function with channels last
  p <- plot(cw_last)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_last, as_plotly = TRUE, output_idx = c(1,2), data_idx = c(1,4))
  expect_s4_class(p, "innsight_plotly")

  # Test plot_global
  box <- plot_global(cw_first)
  expect_s4_class(box, "innsight_ggplot2")
  box <- plot_global(cw_first, as_plotly = TRUE)
  expect_s4_class(box, "innsight_plotly")
})



test_that("ConnectionWeights (global): Keras model with two inputs + two outputs", {
  library(keras)

  data_first <- lapply(list(c(10,5), c(10,2,10,10)),
                       function(x) array(rnorm(prod(x)), dim = x))
  data_last <- lapply(list(c(10,5), c(10,10,10,2)),
                      function(x) array(rnorm(prod(x)), dim = x))
  main_input <- layer_input(shape = c(10,10,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2)) %>%
    layer_flatten() %>%
    layer_dense(units = 4)
  auxiliary_input <- layer_input(shape = c(5), name = 'aux_input')
  auxiliary_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 2, activation = 'softmax', name = 'aux_output')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 4, activation = 'tanh') %>%
    layer_dense(units = 2, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'sigmoid', name = 'main_output')
  model <- keras_model(
    inputs = c(auxiliary_input, main_input),
    outputs = c(auxiliary_output, main_output)
  )

  converter <- Converter$new(model)

  # Channels first
  cw_first <- ConnectionWeights$new(converter, output_idx = list(c(2), c(1,3)),
                                    data = data_first, times_input = TRUE)
  result <- cw_first$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,2,10,10,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,2,10,10,2))

  # Channels last
  cw_last <- ConnectionWeights$new(converter, output_idx = list(c(2), c(1,3)),
                                   channels_first = FALSE, data = data_last,
                                   times_input = TRUE)
  result <- cw_last$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,10,10,2,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,10,10,2,2))

  # get_result method
  result <- cw_first$get_result("torch.tensor")
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,2,10,10,1))
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,2,10,10,2))
  result <- cw_first$get_result("data.frame")
  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 10*5 + 10*2*10*10 + 10*2*5 + 10*2*10*10*2)

  # Test plot function with channels first
  p <- plot(cw_first)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, output_idx = list(c(2), c(1)))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_first, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_first, as_plotly = TRUE, output_idx = list(c(2), c(1)))
  expect_s4_class(p, "innsight_plotly")


  # Test plot function with channels last
  p <- plot(cw_last)
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, output_idx = list(c(2), c(1)))
  expect_s4_class(p, "innsight_ggplot2")
  p <- plot(cw_last, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  p <- plot(cw_last, as_plotly = TRUE, output_idx = list(c(2), c(1)))
  expect_s4_class(p, "innsight_plotly")

  # Test plot_global
  box <- plot_global(cw_first)
  expect_s4_class(box, "innsight_ggplot2")
  box <- plot_global(cw_first, as_plotly = TRUE)
  expect_s4_class(box, "innsight_plotly")
})
