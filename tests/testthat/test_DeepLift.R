
test_that("DeepLift: General errors", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  expect_error(DeepLift$new(model, data))
  expect_error(DeepLift$new(converter, model))
  expect_error(DeepLift$new(converter, data, channels_first = NULL))
  expect_error(DeepLift$new(converter, data, rule_name = "asdf"))
  expect_error(DeepLift$new(converter, data, rule_param = "asdf"))
  expect_error(DeepLift$new(converter, data, dtype = NULL))
  expect_error(DeepLift$new(converter, data, ignore_last_act = c(1)))
})

test_that("DeepLift: Plot and Boxplot", {
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
  d <- DeepLift$new(converter, data,
                    dtype = "double",
                    ignore_last_act = FALSE
  )

  # ggplot2

  # Non-existing data points
  expect_error(plot(d, data_idx = c(1,11)))
  expect_error(boxplot(d, data_idx = 1:11))
  # Non-existing class
  expect_error(plot(d, output_idx = c(5)))
  expect_error(boxplot(d, output_idx = c(5)))

  p <- plot(d)
  boxp <- boxplot(d)
  expect_true("ggplot" %in% class(p))
  expect_true("ggplot" %in% class(boxp))
  p <- plot(d, data_idx = 1:3)
  boxp <- boxplot(d, data_idx = 1:4)
  expect_true("ggplot" %in% class(p))
  expect_true("ggplot" %in% class(boxp))
  p <- plot(d, data_idx = 1:3, output_idx = 1:3)
  boxp <- boxplot(d, data_idx = 1:5, output_idx = 1:3)
  expect_true("ggplot" %in% class(p))
  expect_true("ggplot" %in% class(boxp))

  # plotly
  library(plotly)

  p <- plot(d, as_plotly = TRUE)
  boxp <- boxplot(d, as_plotly = TRUE)
  expect_true("plotly" %in% class(p))
  expect_true("plotly" %in% class(boxp))
  p <- plot(d, data_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(d, data_idx = 1:4, as_plotly = TRUE)
  expect_true("plotly" %in% class(p))
  expect_true("plotly" %in% class(boxp))
  p <- plot(d, data_idx = 1:3, output_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(d, data_idx = 1:5, output_idx = 1:3, as_plotly = TRUE)
  expect_true("plotly" %in% class(p))
  expect_true("plotly" %in% class(boxp))

})

test_that("DeepLift: Dense-Net (Neuralnet)", {
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

  # Rescale Rule
  d <- DeepLift$new(converter, data,
                    x_ref = x_ref,
                    dtype = "double",
                    ignore_last_act = FALSE
  )

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deeplift_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rescale), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deeplift_rescale$sum(dim = 2) - contrib_true)^2)), 1e-12
  )

  d <-
    DeepLift$new(converter, data,
                 x_ref = x_ref,
                 dtype = "double",
                 ignore_last_act = TRUE
    )
  deeplift_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rescale_no_last_act), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deeplift_rescale_no_last_act$sum(dim = 2) -
                        contrib_no_last_act_true)^2)), 1e-12
  )

  # Reveal-Cancel rule
  d <- DeepLift$new(converter, data,
                    x_ref = x_ref,
                    dtype = "float",
                    ignore_last_act = FALSE,
                    rule_name = "reveal_cancel"
  )
  deeplift_rc <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rc), c(10, 4, 3))

  d <- DeepLift$new(converter, data,
                    x_ref = x_ref,
                    dtype = "double",
                    ignore_last_act = TRUE,
                    rule_name = "reveal_cancel"
  )
  deeplift_rc_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rc_no_last_act), c(10, 4, 3))
  expect_lt(as.array(mean(abs(deeplift_rc_no_last_act$sum(dim = 2) -
                                contrib_no_last_act_true)^2)), 1e-12)
})

test_that("DeepLift: Dense-Net (keras)", {
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

  # Rescale Rule
  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "double",
    ignore_last_act = FALSE
  )

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deeplift_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rescale), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deeplift_rescale$sum(dim = 2) - contrib_true)^2)), 1e-12
  )

  d <-
    DeepLift$new(converter, data,
      x_ref = x_ref,
      dtype = "float",
      ignore_last_act = TRUE
    )
  deeplift_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rescale_no_last_act), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deeplift_rescale_no_last_act$sum(dim = 2) -
      contrib_no_last_act_true)^2)), 1e-12
  )

  # Reveal-Cancel rule
  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "float",
    ignore_last_act = FALSE,
    rule_name = "reveal_cancel"
  )
  deeplift_rc <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rc), c(10, 4, 3))

  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "double",
    ignore_last_act = TRUE,
    rule_name = "reveal_cancel"
  )
  deeplift_rc_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rc_no_last_act), c(10, 4, 3))
  expect_lt(as.array(mean(abs(deeplift_rc_no_last_act$sum(dim = 2) -
    contrib_no_last_act_true)^2)), 1e-12)
})

test_that("DeepLift: Conv1D-Net", {
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

  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "double",
    channels_first = FALSE,
    ignore_last_act = FALSE
  )

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deeplift_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rescale), c(4, 64, 3, 1))
  expect_lt(as.array(mean(abs(deeplift_rescale$sum(dim = c(2, 3)) -
    contrib_true)^2)), 1e-12)

  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "float",
    ignore_last_act = TRUE,
    channels_first = FALSE
  )
  deeplift_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rescale_no_last_act), c(4, 64, 3, 1))
  expect_lt(as.array(mean(abs(deeplift_rescale_no_last_act$sum(dim = c(2, 3)) -
    contrib_no_last_act_true)^2)), 1e-12)

  # Reveal-Cancel rule
  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "float",
    ignore_last_act = FALSE,
    rule_name = "reveal_cancel",
    channels_first = FALSE
  )
  deeplift_rc <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rc), c(4, 64, 3, 1))

  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "double",
    ignore_last_act = TRUE,
    rule_name = "reveal_cancel",
    channels_first = FALSE
  )
  deeplift_rc_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rc_no_last_act), c(4, 64, 3, 1))
  expect_lt(as.array(mean(abs(deeplift_rc_no_last_act$sum(dim = c(2, 3)) -
    contrib_no_last_act_true)^2)), 1e-12)
})

test_that("DeepLift: Conv2D-Net", {
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

  converter <- Converter$new(model)
  x_ref <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))

  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "double",
    channels_first = FALSE,
    ignore_last_act = FALSE
  )

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deeplift_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rescale), c(4, 32, 32, 3, 2))
  expect_lt(as.array(mean(abs(deeplift_rescale$sum(dim = c(2, 3, 4)) -
    contrib_true)^2)), 1e-12)

  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "float",
    ignore_last_act = TRUE,
    channels_first = FALSE
  )
  deeplift_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rescale_no_last_act), c(4, 32, 32, 3, 2))
  expect_lt(
    as.array(mean(abs(deeplift_rescale_no_last_act$sum(dim = c(2, 3, 4)) -
      contrib_no_last_act_true)^2)), 1e-12
  )

  # Reveal-Cancel rule
  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "float",
    ignore_last_act = FALSE,
    rule_name = "reveal_cancel",
    channels_first = FALSE
  )
  deeplift_rc <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rc), c(4, 32, 32, 3, 2))

  d <- DeepLift$new(converter, data,
    x_ref = x_ref,
    dtype = "double",
    ignore_last_act = TRUE,
    rule_name = "reveal_cancel",
    channels_first = FALSE
  )
  deeplift_rc_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deeplift_rc_no_last_act), c(4, 32, 32, 3, 2))
  expect_lt(
    as.array(mean(abs(deeplift_rc_no_last_act$sum(dim = c(2, 3, 4)) -
      contrib_no_last_act_true)^2)), 1e-12
  )
})
