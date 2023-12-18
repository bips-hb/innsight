
test_that("SHAP: General errors", {
  library(neuralnet)

  # Fit model
  model <- neuralnet(Species ~ Petal.Length + Petal.Width, iris,
                     linear.output = FALSE)
  data <- iris[, c(3,4)]

  expect_error(SHAP$new()) # missing converter
  expect_error(SHAP$new(model)) # missing data
  expect_error(SHAP$new(NULL, data[1:2, ], data)) # no output_type
  expect_error(SHAP$new(NULL, data[1:2, ], data, output_type = "regression")) # no pred_fun
  expect_error(SHAP$new(NULL, data[1:2, ], data,
                        output_type = "regression",
                        perd_fun = function(newdata, ...) newdata))

  SHAP$new(model, data[1:2, ], data) # successful run
  expect_error(SHAP$new(model, data[1:2, ], data, output_type = "ds")) # wrong output_type
  expect_error(SHAP$new(model, data[1:2, ], data, pred_fun = identity)) # wrong pred_fun
  expect_error(SHAP$new(model, data[1:2, ], data, output_idx = c(1,4))) # wrong output_idx
  SHAP$new(model, data[1:2, ], data, output_idx = c(2))
  expect_error(SHAP$new(model, data[1:2, ], data, input_dim = c(1))) # wrong input_dim
  expect_error(SHAP$new(model, data[1:2, ], data, input_names = c("a", "b", "d"))) # wrong input_names
  SHAP$new(model, data[1:2, ], data, input_names = factor(c("a", "b")))
  expect_error(SHAP$new(model, data[1:2, ], data, output_names = c("a", "d"))) # wrong output_names
  SHAP$new(model, data[1:2, ], data, output_names = factor(c("a", "d", "c")))

  # Forwarding arguments to fastshap::explain
  shap <- SHAP$new(model, data[1:10, ], data, nsim = 4)

  # get_result()
  res <- get_result(shap)
  expect_array(res)
  res <- get_result(shap, "data.frame")
  expect_data_frame(res)
  res <- get_result(shap, "torch_tensor")
  expect_class(res, "torch_tensor")

  # Plots

  # Non-existing data points
  expect_error(plot(shap, data_idx = c(1,11)))
  expect_error(boxplot(shap, data_idx = 1:11))
  # Non-existing class
  expect_error(plot(shap, output_idx = c(5)))
  expect_error(boxplot(shap, output_idx = c(5)))

  p <- plot(shap)
  boxp <- boxplot(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(shap, data_idx = 1:3)
  boxp <- boxplot(shap, data_idx = 1:4)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(shap, data_idx = 1:3, output_idx = 1:3)
  boxp <- boxplot(shap, data_idx = 1:3, output_idx = 1:3)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  boxp <- boxplot(shap, ref_data_idx = c(4))

  # plotly
  library(plotly)

  p <- plot(shap, as_plotly = TRUE)
  boxp <- boxplot(shap, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(shap, data_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(shap, data_idx = 1:4, as_plotly = TRUE, individual_max = 2,
                  individual_data_idx = c(1,2,5,6))
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(shap, data_idx = 1:3, output_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(shap, data_idx = 1:5, output_idx = 1:3, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")


})

test_that("SHAP: Dense-Net (Neuralnet)", {
  library(neuralnet)
  library(torch)

  data(iris)
  data <- iris[sample.int(150, size = 10), -5]
  nn <- neuralnet(Species ~ .,
                  iris,
                  linear.output = FALSE,
                  hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
  )

  # Normal model
  shap <- SHAP$new(nn, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 3))
  p <- plot(shap, output_idx = c(2,3))
  boxp <- boxplot(shap, output_idx = c(2,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(nn)
  shap <- SHAP$new(conv, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 3))
  p <- plot(shap, output_idx = c(2,3))
  boxp <- boxplot(shap, output_idx = c(2,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
})


test_that("SHAP: Dense-Net (keras)", {
  library(keras)
  library(torch)

  # Classification -------------------------------------------------------------
  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  # Normal model
  shap <- SHAP$new(model, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 3))
  p <- plot(shap, output_idx = c(1,3))
  boxp <- boxplot(shap, output_idx = c(1,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  shap <- SHAP$new(conv, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 3))
  p <- plot(shap, output_idx = c(1,3))
  boxp <- boxplot(shap, output_idx = c(1,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 2, activation = "linear")

  # Normal model
  shap <- SHAP$new(model, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- boxplot(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  shap <- SHAP$new(conv, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- boxplot(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result -------------------------------------------------------------
  res_array <- shap$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- shap$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- shap$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
  expect_error(shap$get_result(type = "adsf"))
})


test_that("SHAP: Conv1D-Net (keras)", {
  library(keras)
  library(torch)

  # Classification -------------------------------------------------------------
  data <- array(rnorm(4 * 14 * 3), dim = c(4, 14, 3))
  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(14, 3), kernel_size = 8, filters = 2,
      activation = "softplus"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 1, activation = "sigmoid")

  # Normal model
  shap <- SHAP$new(model, data[1:2,, ], data, channels_first = FALSE)
  expect_equal(dim(shap$get_result()), c(2, 14, 3, 1))
  p <- plot(shap)
  boxp <- boxplot(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  shap <- SHAP$new(conv, data[1:2,, ], data, channels_first = FALSE)
  expect_equal(dim(shap$get_result()), c(2, 14, 3, 1))
  p <- plot(shap)
  boxp <- boxplot(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- array(rnorm(4 * 14 * 3), dim = c(4, 14, 3))
  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(14, 3), kernel_size = 8, filters = 4,
      activation = "softplus"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 2, activation = "linear")

  # Normal model
  shap <- SHAP$new(model, data[1:2,, ], data, channels_first = FALSE)
  expect_equal(dim(shap$get_result()), c(2, 14, 3, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- boxplot(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  shap <- SHAP$new(conv, data[1:2,, ], data, channels_first = FALSE)
  expect_equal(dim(shap$get_result()), c(2, 14, 3, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- boxplot(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result ------------------------------------------------------------
  res_array <- shap$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- shap$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- shap$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})

test_that("SHAP: Conv2D-Net (keras)", {
  library(keras)
  library(torch)

  # Classification -------------------------------------------------------------
  data <- array(rnorm(4 * 4 * 4 * 3), dim = c(4, 4, 4, 3))
  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(4, 4, 3), kernel_size = 2, filters = 4,
      activation = "softplus"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 1, activation = "sigmoid")

  # Normal model
  shap <- SHAP$new(model, data[1:2,,, ], data, channels_first = FALSE)
  expect_equal(dim(shap$get_result()), c(2, 4, 4, 3, 1))
  p <- plot(shap)
  boxp <- plot_global(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  shap <- SHAP$new(conv, data[1:2,,, ], data, channels_first = FALSE)
  expect_equal(dim(shap$get_result()), c(2, 4, 4, 3, 1))
  p <- plot(shap)
  boxp <- plot_global(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- array(rnorm(4 * 4 * 4 * 3), dim = c(4, 4, 4, 3))
  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(4, 4, 3), kernel_size = 2, filters = 4,
      activation = "softplus"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 2, activation = "linear")

  # Normal model
  shap <- SHAP$new(model, data[1:2,,, ], data, channels_first = FALSE)
  expect_equal(dim(shap$get_result()), c(2, 4, 4, 3, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- plot_global(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  shap <- SHAP$new(conv, data[1:2,,, ], data, channels_first = FALSE)
  expect_equal(dim(shap$get_result()), c(2, 4, 4, 3, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- plot_global(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result ------------------------------------------------------------
  res_array <- shap$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- shap$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- shap$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})


test_that("SHAP: Dense-Net (torch)", {
  library(torch)

  # Classification -------------------------------------------------------------
  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- nn_sequential(
    nn_linear(4, 16),
    nn_relu(),
    nn_linear(16, 3),
    nn_softmax(dim = -1)
  )

  # Normal model
  shap <- SHAP$new(model, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 3))
  p <- plot(shap, output_idx = c(1,3))
  boxp <- boxplot(shap, output_idx = c(1,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(4))
  shap <- SHAP$new(conv, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 3))
  p <- plot(shap, output_idx = c(1,3))
  boxp <- boxplot(shap, output_idx = c(1,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- nn_sequential(
    nn_linear(4, 16),
    nn_relu(),
    nn_linear(16, 2)
  )

  # Normal model
  shap <- SHAP$new(model, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- boxplot(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(4))
  shap <- SHAP$new(conv, data[1:2, ], data)
  expect_equal(dim(shap$get_result()), c(2, 4, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- boxplot(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result -------------------------------------------------------------
  res_array <- shap$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- shap$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- shap$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})


test_that("SHAP: Conv1D-Net (torch)", {
  library(torch)

  # Classification -------------------------------------------------------------
  data <- array(rnorm(4 * 14 * 3), dim = c(4, 3, 14))
  model <- nn_sequential(
    nn_conv1d(3, 8, 8),
    nn_relu(),
    nn_flatten(),
    nn_linear(56, 16),
    nn_relu(),
    nn_linear(16, 1),
    nn_sigmoid()
  )

  # Normal model
  shap <- SHAP$new(model, data[1:2,, ], data, channels_first = TRUE)
  expect_equal(dim(shap$get_result()), c(2, 3, 14, 1))
  p <- plot(shap)
  boxp <- boxplot(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(3, 14))
  shap <- SHAP$new(conv, data[1:2,, ], data, channels_first = TRUE)
  expect_equal(dim(shap$get_result()), c(2, 3, 14, 1))
  p <- plot(shap)
  boxp <- boxplot(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- array(rnorm(4 * 14 * 3), dim = c(4, 3, 14))
  model <- nn_sequential(
    nn_conv1d(3, 8, 8),
    nn_relu(),
    nn_flatten(),
    nn_linear(56, 16),
    nn_relu(),
    nn_linear(16, 2)
  )

  # Normal model
  shap <- SHAP$new(model, data[1:2,, ], data, channels_first = TRUE)
  expect_equal(dim(shap$get_result()), c(2, 3, 14, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- boxplot(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(3, 14))
  shap <- SHAP$new(conv, data[1:2,, ], data, channels_first = TRUE)
  expect_equal(dim(shap$get_result()), c(2, 3, 14, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- boxplot(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result ------------------------------------------------------------
  res_array <- shap$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- shap$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- shap$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})

test_that("SHAP: Conv2D-Net (torch)", {
  library(keras)
  library(torch)

  # Classification -------------------------------------------------------------
  data <- array(rnorm(4 * 4 * 4 * 3), dim = c(4, 3, 4, 4))
  model <- nn_sequential(
    nn_conv2d(3, 8, c(2, 2)),
    nn_relu(),
    nn_flatten(),
    nn_linear(72, 16),
    nn_relu(),
    nn_linear(16, 1),
    nn_sigmoid()
  )

  # Normal model
  shap <- SHAP$new(model, data[1:2,,, ], data, channels_first = TRUE)
  expect_equal(dim(shap$get_result()), c(2, 3, 4, 4, 1))
  p <- plot(shap)
  boxp <- plot_global(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(3, 4, 4))
  shap <- SHAP$new(conv, data[1:2,,, ], data, channels_first = TRUE)
  expect_equal(dim(shap$get_result()), c(2, 3, 4, 4, 1))
  p <- plot(shap)
  boxp <- plot_global(shap)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- array(rnorm(4 * 4 * 4 * 3), dim = c(4, 3, 4, 4))
  model <- nn_sequential(
    nn_conv2d(3, 8, c(2, 2)),
    nn_relu(),
    nn_flatten(),
    nn_linear(72, 16),
    nn_relu(),
    nn_linear(16, 2)
  )

  # Normal model
  shap <- SHAP$new(model, data[1:2,,, ], data, channels_first = TRUE)
  expect_equal(dim(shap$get_result()), c(2, 3, 4, 4, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- plot_global(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(3, 4, 4))
  shap <- SHAP$new(conv, data[1:2,,, ], data, channels_first = TRUE)
  expect_equal(dim(shap$get_result()), c(2, 3, 4, 4, 2))
  p <- plot(shap, output_idx = c(2))
  boxp <- plot_global(shap, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result ------------------------------------------------------------
  res_array <- shap$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- shap$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- shap$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})


test_that("SHAP: Keras multiple input or output layers", {
  library(keras)

  # Multiple input layers
  main_input <- layer_input(shape = c(10,10,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2), activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 4)
  auxiliary_input <- layer_input(shape = c(5), name = 'aux_input')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  model <- keras_model(
    inputs = c(auxiliary_input, main_input),
    outputs = c(main_output)
  )
  data <- lapply(list(c(5), c(10,10,2)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))

  expect_error(SHAP$new(model, data))

  # Multiple output layers
  main_input <- layer_input(shape = c(10,10,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2), activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 4)
  auxiliary_output <- lstm_out %>%
    layer_dense(units = 2, activation = 'softmax', name = 'aux_output')
  main_output <- lstm_out %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  model <- keras_model(
    inputs = c(main_input),
    outputs = c(auxiliary_output, main_output)
  )
  data <- lapply(list(c(10,10,2)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))


  expect_error(SHAP$new(model, data))
})

test_that("Custom model", {

  # Ranger model and iris dataset
  library(ranger)

  model <- ranger(Species ~ ., data = iris, probability = TRUE)

  pred_fun <- function(newdata, ...) {
    predict(model, newdata, ...)$predictions
  }

  shap <- SHAP$new(model, iris[c(1,70, 111), -5], iris[, -5],
                   pred_fun = pred_fun,
                   output_names = levels(iris$Species))

  res <- get_result(shap)
  expect_equal(dim(res), c(3, 4, 3))

  p <- plot(shap, output_idx = c(1, 3))
  boxp <- boxplot(shap, output_idx = c(1, 3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
})
