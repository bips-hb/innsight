
test_that("LIME: General errors", {
  library(neuralnet)

  # Fit model
  model <- neuralnet(Species ~ Petal.Length + Petal.Width, iris,
                     linear.output = FALSE)
  data <- iris[, c(3,4)]

  expect_error(LIME$new()) # missing converter
  expect_error(LIME$new(model)) # missing data
  expect_error(LIME$new(NULL, data, data[1:2, ])) # no output_type
  expect_error(LIME$new(NULL, data, data[1:2, ], output_type = "regression")) # no pred_fun
  expect_error(LIME$new(NULL, data, data[1:2, ],
                        output_type = "regression",
                        perd_fun = function(newdata, ...) newdata))

  LIME$new(model, data, data[1:2, ]) # successful run
  expect_error(LIME$new(model, data, data[1:2, ], output_type = "ds")) # wrong output_type
  expect_error(LIME$new(model, data, data[1:2, ], pred_fun = identity)) # wrong pred_fun
  expect_error(LIME$new(model, data, data[1:2, ], output_idx = c(1,4))) # wrong output_idx
  LIME$new(model, data, data[1:2, ], output_idx = c(2))
  expect_error(LIME$new(model, data, data[1:2, ], input_dim = c(1))) # wrong input_dim
  expect_error(LIME$new(model, data, data[1:2, ], input_names = c("a", "b", "d"))) # wrong input_names
  LIME$new(model, data, data[1:2, ], input_names = factor(c("a", "b")))
  expect_error(LIME$new(model, data, data[1:2, ], output_names = c("a", "d"))) # wrong output_names
  LIME$new(model, data, data[1:2, ], output_names = factor(c("a", "d", "c")))

  expect_error(LIME$new(model, data, data[1:2, ], output_idx = c(1, 10)))
  LIME$new(model, data, data[1:2, ], output_idx = c(1, 2))
  expect_error(LIME$new(model, data, data[1:2, ], output_idx = list(c(1, 10))))
  LIME$new(model, data, data[1:2, ], output_idx = list(c(1, 3)))
  expect_error(LIME$new(model, data, data[1:2, ], output_idx = list(NULL, c(1, 2))))
  expect_error(LIME$new(model, data, data[1:2, ], output_label = c(1, 2)))
  expect_error(LIME$new(model, data, data[1:2, ], output_label = c("A", "b")))
  LIME$new(model, data, data[1:2, ], output_label = c("setosa", "virginica"))
  LIME$new(model, data, data[1:2, ], output_label = as.factor(c("setosa", "virginica")))
  LIME$new(model, data, data[1:2, ], output_label = list(c("setosa", "virginica")))
  expect_error(LIME$new(model, data, data[1:2, ],
                        output_label =  c("setosa", "virginica"),
                        output_idx = c(1, 2)))
  LIME$new(model, data, data[1:2, ],
           output_label =  c("setosa", "virginica"),
           output_idx = c(1, 3))

  # Forwarding arguments to lime::explain
  lime <- LIME$new(model, data, data[1:10, ], n_permutations = 100, gower_power = 3)

  # get_result()
  res <- get_result(lime)
  expect_array(res)
  res <- get_result(lime, "data.frame")
  expect_data_frame(res)
  res <- get_result(lime, "torch_tensor")
  expect_class(res, "torch_tensor")

  # Plots

  # Non-existing data points
  expect_error(plot(lime, data_idx = c(1,11)))
  expect_error(boxplot(lime, data_idx = 1:11))
  # Non-existing class
  expect_error(plot(lime, output_idx = c(5)))
  expect_error(boxplot(lime, output_idx = c(5)))

  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(lime, data_idx = 1:3)
  boxp <- boxplot(lime, data_idx = 1:4)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(lime, data_idx = 1:3, output_idx = 1:3)
  boxp <- boxplot(lime, data_idx = 1:3, output_idx = 1:3)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  boxp <- boxplot(lime, ref_data_idx = c(4))

  # plotly
  library(plotly)

  p <- plot(lime, as_plotly = TRUE)
  boxp <- boxplot(lime, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(lime, data_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(lime, data_idx = 1:4, as_plotly = TRUE, individual_max = 2,
                  individual_data_idx = c(1,2,5,6))
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(lime, data_idx = 1:3, output_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(lime, data_idx = 1:5, output_idx = 1:3, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")


})

test_that("LIME: Dense-Net (Neuralnet)", {
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
  lime <- LIME$new(nn, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 3))
  p <- plot(lime, output_idx = c(2,3))
  boxp <- boxplot(lime, output_idx = c(2,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(nn)
  lime <- LIME$new(conv, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 3))
  p <- plot(lime, output_idx = c(2,3))
  boxp <- boxplot(lime, output_idx = c(2,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
})


test_that("LIME: Dense-Net (keras)", {
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
  lime <- LIME$new(model, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 3))
  p <- plot(lime, output_idx = c(1,3))
  boxp <- boxplot(lime, output_idx = c(1,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  lime <- LIME$new(conv, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 3))
  p <- plot(lime, output_idx = c(1,3))
  boxp <- boxplot(lime, output_idx = c(1,3))
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
  lime <- LIME$new(model, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  lime <- LIME$new(conv, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result -------------------------------------------------------------
  res_array <- lime$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- lime$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- lime$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
  expect_error(lime$get_result(type = "adsf"))
})


test_that("LIME: Conv1D-Net (keras)", {
  library(keras)
  library(torch)

  # Classification -------------------------------------------------------------
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

  # Normal model
  lime <- LIME$new(model, data, data[1:2,, ], channels_first = FALSE)
  expect_equal(dim(lime$get_result()), c(2, 64, 3, 1))
  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  lime <- LIME$new(conv, data, data[1:2,, ], channels_first = FALSE)
  expect_equal(dim(lime$get_result()), c(2, 64, 3, 1))
  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
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
    layer_dense(units = 2, activation = "linear")

  # Normal model
  lime <- LIME$new(model, data, data[1:2,, ], channels_first = FALSE)
  expect_equal(dim(lime$get_result()), c(2, 64, 3, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  lime <- LIME$new(conv, data, data[1:2,, ], channels_first = FALSE)
  expect_equal(dim(lime$get_result()), c(2, 64, 3, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result ------------------------------------------------------------
  res_array <- lime$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- lime$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- lime$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})

test_that("LIME: Conv2D-Net (keras)", {
  library(keras)
  library(torch)

  # Classification -------------------------------------------------------------
  data <- array(rnorm(4 * 10 * 10 * 3), dim = c(4, 10, 10, 3))
  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(10, 10, 3), kernel_size = 4, filters = 8,
      activation = "softplus"
    ) %>%
    layer_conv_2d(kernel_size = 4, filters = 2, activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # Normal model
  lime <- LIME$new(model, data, data[1:2,,, ], channels_first = FALSE)
  expect_equal(dim(lime$get_result()), c(2, 10, 10, 3, 1))
  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  lime <- LIME$new(conv, data, data[1:2,,, ], channels_first = FALSE)
  expect_equal(dim(lime$get_result()), c(2, 10, 10, 3, 1))
  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- array(rnorm(4 * 10 * 10 * 3), dim = c(4, 10, 10, 3))
  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(10, 10, 3), kernel_size = 4, filters = 8,
      activation = "softplus"
    ) %>%
    layer_conv_2d(kernel_size = 4, filters = 2, activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 2, activation = "linear")

  # Normal model
  lime <- LIME$new(model, data, data[1:2,,, ], channels_first = FALSE)
  expect_equal(dim(lime$get_result()), c(2, 10, 10, 3, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model)
  lime <- LIME$new(conv, data, data[1:2,,, ], channels_first = FALSE)
  expect_equal(dim(lime$get_result()), c(2, 10, 10, 3, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result ------------------------------------------------------------
  res_array <- lime$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- lime$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- lime$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})


test_that("LIME: Dense-Net (torch)", {
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
  lime <- LIME$new(model, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 3))
  p <- plot(lime, output_idx = c(1,3))
  boxp <- boxplot(lime, output_idx = c(1,3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(4))
  lime <- LIME$new(conv, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 3))
  p <- plot(lime, output_idx = c(1,3))
  boxp <- boxplot(lime, output_idx = c(1,3))
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
  lime <- LIME$new(model, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(4))
  lime <- LIME$new(conv, data, data[1:2, ])
  expect_equal(dim(lime$get_result()), c(2, 4, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result -------------------------------------------------------------
  res_array <- lime$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- lime$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- lime$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})


test_that("LIME: Conv1D-Net (torch)", {
  library(torch)

  # Classification -------------------------------------------------------------
  data <- array(rnorm(4 * 64 * 3), dim = c(4, 3, 64))
  model <- nn_sequential(
    nn_conv1d(3, 8, 16),
    nn_softplus(),
    nn_conv1d(8, 4, 16),
    nn_tanh(),
    nn_conv1d(4, 2, 16),
    nn_relu(),
    nn_flatten(),
    nn_linear(38, 16),
    nn_relu(),
    nn_linear(16, 1),
    nn_sigmoid()
  )

  # Normal model
  lime <- LIME$new(model, data, data[1:2,, ], channels_first = TRUE)
  expect_equal(dim(lime$get_result()), c(2, 3, 64, 1))
  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(3, 64))
  lime <- LIME$new(conv, data, data[1:2,, ], channels_first = TRUE)
  expect_equal(dim(lime$get_result()), c(2, 3, 64, 1))
  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- array(rnorm(4 * 64 * 3), dim = c(4, 3, 64))
  model <- nn_sequential(
    nn_conv1d(3, 8, 16),
    nn_softplus(),
    nn_conv1d(8, 4, 16),
    nn_tanh(),
    nn_conv1d(4, 2, 16),
    nn_relu(),
    nn_flatten(),
    nn_linear(38, 16),
    nn_relu(),
    nn_linear(16, 2)
  )

  # Normal model
  lime <- LIME$new(model, data, data[1:2,, ], channels_first = TRUE)
  expect_equal(dim(lime$get_result()), c(2, 3, 64, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(3, 64))
  lime <- LIME$new(conv, data, data[1:2,, ], channels_first = TRUE)
  expect_equal(dim(lime$get_result()), c(2, 3, 64, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result ------------------------------------------------------------
  res_array <- lime$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- lime$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- lime$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})

test_that("LIME: Conv2D-Net (torch)", {
  library(keras)
  library(torch)

  # Classification -------------------------------------------------------------
  data <- array(rnorm(4 * 10 * 10 * 3), dim = c(4, 3, 10, 10))
  model <- nn_sequential(
    nn_conv2d(3, 8, c(4, 4)),
    nn_softplus(),
    nn_conv2d(8, 2, c(4, 4)),
    nn_relu(),
    nn_flatten(),
    nn_linear(32, 16),
    nn_relu(),
    nn_linear(16, 1),
    nn_sigmoid()
  )

  # Normal model
  lime <- LIME$new(model, data, data[1:2,,, ], channels_first = TRUE)
  expect_equal(dim(lime$get_result()), c(2, 3, 10, 10, 1))
  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(3, 10, 10))
  lime <- LIME$new(conv, data, data[1:2,,, ], channels_first = TRUE)
  expect_equal(dim(lime$get_result()), c(2, 3, 10, 10, 1))
  p <- plot(lime)
  boxp <- boxplot(lime)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Regression -----------------------------------------------------------------
  data <- array(rnorm(4 * 10 * 10 * 3), dim = c(4, 3, 10, 10))
  model <- nn_sequential(
    nn_conv2d(3, 8, c(4, 4)),
    nn_softplus(),
    nn_conv2d(8, 2, c(4, 4)),
    nn_relu(),
    nn_flatten(),
    nn_linear(32, 16),
    nn_relu(),
    nn_linear(16, 2)
  )

  # Normal model
  lime <- LIME$new(model, data, data[1:2,,, ], channels_first = TRUE)
  expect_equal(dim(lime$get_result()), c(2, 3, 10, 10, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Converter
  conv <- Converter$new(model, input_dim = c(3, 10, 10))
  lime <- LIME$new(conv, data, data[1:2,,, ], channels_first = TRUE)
  expect_equal(dim(lime$get_result()), c(2, 3, 10, 10, 2))
  p <- plot(lime, output_idx = c(2))
  boxp <- boxplot(lime, output_idx = c(2))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # Test get_result ------------------------------------------------------------
  res_array <- lime$get_result()
  expect_true(is.array(res_array))
  res_dataframe <- lime$get_result(type = "data.frame")
  expect_true(is.data.frame(res_dataframe))
  res_torch <- lime$get_result(type = "torch.tensor")
  expect_true(inherits(res_torch, "torch_tensor"))
})


test_that("LIME: Keras multiple input or output layers", {
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

  expect_error(LIME$new(model, data))

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


  expect_error(LIME$new(model, data))
})

test_that("Custom model", {

  # Ranger model and iris dataset
  library(ranger)

  model <- ranger(Species ~ ., data = iris, probability = TRUE)

  pred_fun <- function(newdata, ...) {
    predict(model, newdata, ...)$predictions
  }

  lime <- LIME$new(model, iris[, -5], iris[c(1,70, 111), -5],
                   output_type = "classification",
                   pred_fun = pred_fun,
                   output_names = levels(iris$Species))

  res <- get_result(lime)
  expect_equal(dim(res), c(3, 4, 3))

  p <- plot(lime, output_idx = c(1, 3))
  boxp <- boxplot(lime, output_idx = c(1, 3))
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
})
