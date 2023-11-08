
test_that("DeepSHAP: General errors", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)

  expect_error(DeepSHAP$new(model, data))
  expect_error(DeepSHAP$new(converter, model))
  expect_error(DeepSHAP$new(converter, data, channels_first = NULL))
  expect_error(DeepSHAP$new(converter, data, rule_name = "asdf"))
  expect_error(DeepSHAP$new(converter, data, rule_param = "asdf"))
  expect_error(DeepSHAP$new(converter, data, dtype = NULL))
  expect_error(DeepSHAP$new(converter, data, ignore_last_act = c(1)))
  expect_error(DeepSHAP$new(converter, data, data_ref = c(1,2,3)))
})

test_that("DeepSHAP: Plot and Boxplot", {
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
  d <- DeepSHAP$new(converter, data,
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
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(d, data_idx = 1:3)
  boxp <- boxplot(d, data_idx = 1:4)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")
  p <- plot(d, data_idx = 1:3, output_idx = 1:3)
  boxp <- boxplot(d, data_idx = 1:5, output_idx = 1:3)
  expect_s4_class(p, "innsight_ggplot2")
  expect_s4_class(boxp, "innsight_ggplot2")

  # plotly
  library(plotly)

  p <- plot(d, as_plotly = TRUE)
  boxp <- boxplot(d, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(d, data_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(d, data_idx = 1:4, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
  p <- plot(d, data_idx = 1:3, output_idx = 1:3, as_plotly = TRUE)
  boxp <- boxplot(d, data_idx = 1:5, output_idx = 1:3, as_plotly = TRUE)
  expect_s4_class(p, "innsight_plotly")
  expect_s4_class(boxp, "innsight_plotly")
})

test_that("DeepSHAP: Dense-Net (Neuralnet)", {
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

  # Rescale Rule (no reference dataset)
  d <- DeepSHAP$new(converter, data,
                    ignore_last_act = FALSE
  )
  res <- converter$model(torch_tensor(t(data))$t(), TRUE,
                         TRUE, TRUE, TRUE)
  res <- converter$model$update_ref(torch_zeros(c(1,4)),
                                    TRUE, TRUE, TRUE, TRUE)

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deepshap_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deepshap_rescale$sum(dim = 2) - contrib_true)^2)), 1e-8
  )

  d <-
    DeepSHAP$new(converter, data,
                 ignore_last_act = TRUE
    )
  deepshap_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale_no_last_act), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deepshap_rescale_no_last_act$sum(dim = 2) -
                        contrib_no_last_act_true)^2)), 1e-8
  )

  # Rescale Rule (with reference dataset)
  data_ref <- matrix(rnorm(4 * 20), nrow = 20)
  d <- DeepSHAP$new(converter, data,
                    data_ref = data_ref,
                    ignore_last_act = FALSE
  )
  res <- converter$model(torch_tensor(t(data))$t(), TRUE,
                         TRUE, TRUE, TRUE)
  res <- converter$model$update_ref(torch_tensor(data_ref),
                                    TRUE, TRUE, TRUE, TRUE)

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deepshap_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deepshap_rescale$sum(dim = 2) - contrib_true)^2)), 1e-8
  )

  d <-
    DeepSHAP$new(converter, data,
                 data_ref = data_ref,
                 ignore_last_act = TRUE
    )
  deepshap_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale_no_last_act), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deepshap_rescale_no_last_act$sum(dim = 2) -
                        contrib_no_last_act_true)^2)), 1e-8
  )
})

test_that("DeepSHAP: Dense-Net (keras)", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 8, activation = "tanh") %>%
    layer_dense(units = 3, activation = "softmax")

  converter <- Converter$new(model)
  data_ref <- matrix(rnorm(4 * 20), nrow = 20)

  # Rescale Rule (no reference dataset)
  d <- DeepSHAP$new(converter, data,
                    ignore_last_act = FALSE
  )

  converter$model(torch_tensor(data), TRUE, TRUE, TRUE, TRUE)
  converter$model$update_ref(torch_zeros(c(1,4)), TRUE, TRUE, TRUE, TRUE)

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)

  deepshap_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deepshap_rescale$sum(dim = 2) - contrib_true)^2)), 1e-8
  )

  d <-
    DeepSHAP$new(converter, data,
                 ignore_last_act = TRUE
    )
  deepshap_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale_no_last_act), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deepshap_rescale_no_last_act$sum(dim = 2) -
                        contrib_no_last_act_true)^2)), 1e-8
  )

  # Rescale Rule (with reference dataset)
  d <- DeepSHAP$new(converter, data,
                    data_ref = data_ref,
                    ignore_last_act = FALSE
  )

  converter$model(torch_tensor(data), TRUE, TRUE, TRUE, TRUE)
  converter$model$update_ref(torch_tensor(data_ref), TRUE, TRUE, TRUE, TRUE)

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)

  deepshap_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deepshap_rescale$sum(dim = 2) - contrib_true)^2)), 1e-8
  )

  d <-
    DeepSHAP$new(converter, data,
                 data_ref = data_ref,
                 ignore_last_act = TRUE
    )
  deepshap_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale_no_last_act), c(10, 4, 3))
  expect_lt(
    as.array(mean(abs(deepshap_rescale_no_last_act$sum(dim = 2) -
                        contrib_no_last_act_true)^2)), 1e-8
  )
})

test_that("DeepSHAP: Conv1D-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(4 * 32 * 3), dim = c(4, 32, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(32, 3), kernel_size = 16, filters = 8,
      activation = "softplus"
    ) %>%
    layer_conv_1d(kernel_size = 8, filters = 4, activation = "tanh") %>%
    layer_conv_1d(kernel_size = 8, filters = 4, activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)

  # Rescale rule (no reference dataset)
  d <- DeepSHAP$new(converter, data,
                    channels_first = FALSE,
                    ignore_last_act = FALSE
  )

  converter$model(torch_tensor(data), FALSE, TRUE, TRUE, TRUE)
  converter$model$update_ref(torch_zeros(c(1, 32, 3)), FALSE, TRUE, TRUE, TRUE)

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deepshap_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale), c(4, 32, 3, 1))
  expect_lt(as.array(mean(abs(deepshap_rescale$sum(dim = c(2, 3)) -
                                contrib_true)^2)), 1e-8)

  d <- DeepSHAP$new(converter, data,
                    ignore_last_act = TRUE,
                    channels_first = FALSE
  )
  deepshap_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale_no_last_act), c(4, 32, 3, 1))
  expect_lt(as.array(mean(abs(deepshap_rescale_no_last_act$sum(dim = c(2, 3)) -
                                contrib_no_last_act_true)^2)), 1e-8)

  # Rescale rule (with reference dataset)
  data_ref <- array(rnorm(32 * 3 * 5), dim = c(5, 32, 3))

  d <- DeepSHAP$new(converter, data,
                    data_ref = data_ref,
                    channels_first = FALSE,
                    ignore_last_act = FALSE
  )

  converter$model(torch_tensor(data), FALSE, TRUE, TRUE, TRUE)
  converter$model$update_ref(torch_tensor(data_ref), FALSE, TRUE, TRUE, TRUE)

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deepshap_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale), c(4, 32, 3, 1))
  expect_lt(as.array(mean(abs(deepshap_rescale$sum(dim = c(2, 3)) -
                                contrib_true)^2)), 1e-8)

  d <- DeepSHAP$new(converter, data,
                    data_ref = data_ref,
                    ignore_last_act = TRUE,
                    channels_first = FALSE
  )
  deepshap_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale_no_last_act), c(4, 32, 3, 1))
  expect_lt(as.array(mean(abs(deepshap_rescale_no_last_act$sum(dim = c(2, 3)) -
                                contrib_no_last_act_true)^2)), 1e-8)
})

test_that("DeepSHAP: Conv2D-Net", {
  library(keras)
  library(torch)

  data <- array(rnorm(4 * 16 * 16 * 3), dim = c(4, 16, 16, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(16, 16, 3), kernel_size = 8, filters = 8,
      activation = "softplus", padding = "same"
    ) %>%
    layer_conv_2d(
      kernel_size = 4, filters = 4, activation = "relu") %>%
    layer_conv_2d(
      kernel_size = 2, filters = 2, activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 2, activation = "softmax")

  # test non-fitted model
  converter <- Converter$new(model)

  # Rescale rule (no reference data)
  d <- DeepSHAP$new(converter, data,
                    channels_first = FALSE,
                    ignore_last_act = FALSE
  )

  converter$model(torch_tensor(data), FALSE, TRUE, TRUE, TRUE)
  converter$model$update_ref(torch_zeros(c(1, 16, 16, 3)),
                             FALSE, TRUE, TRUE, TRUE)

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deepshap_rescale <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale), c(4, 16, 16, 3, 2))
  expect_lt(as.array(mean(abs(deepshap_rescale$sum(dim = c(2, 3, 4)) -
                                contrib_true)^2)), 1e-8)

  d <- DeepSHAP$new(converter, data,
                    ignore_last_act = TRUE,
                    channels_first = FALSE
  )
  deepshap_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale_no_last_act), c(4, 16, 16, 3, 2))
  expect_lt(
    as.array(mean(abs(deepshap_rescale_no_last_act$sum(dim = c(2, 3, 4)) -
                        contrib_no_last_act_true)^2)), 1e-8
  )

  # Rescale rule (with reference data)
  data_ref <- array(rnorm(10 * 16 * 16 * 3), dim = c(10, 16, 16, 3))
  d <- DeepSHAP$new(converter, data,
                    data_ref = data_ref,
                    channels_first = FALSE,
                    ignore_last_act = FALSE
  )

  converter$model(torch_tensor(data), FALSE, TRUE, TRUE, TRUE)
  converter$model$update_ref(torch_tensor(data_ref), FALSE, TRUE, TRUE, TRUE)

  last_layer <- rev(converter$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <-
    last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- converter$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)

  deepshap_rescale <- d$get_result(type = "torch.tensor")
  as.array(mean(abs(deepshap_rescale$sum(dim = c(2, 3, 4)) -
                      contrib_true)^2))

  expect_equal(dim(deepshap_rescale), c(4, 16, 16, 3, 2))
  expect_lt(as.array(mean(abs(deepshap_rescale$sum(dim = c(2, 3, 4)) -
                                contrib_true)^2)), 1e-8)

  d <- DeepSHAP$new(converter, data,
                    data_ref = data_ref,
                    ignore_last_act = TRUE,
                    channels_first = FALSE
  )
  deepshap_rescale_no_last_act <- d$get_result(type = "torch.tensor")

  expect_equal(dim(deepshap_rescale_no_last_act), c(4, 16, 16, 3, 2))
  expect_lt(
    as.array(mean(abs(deepshap_rescale_no_last_act$sum(dim = c(2, 3, 4)) -
                        contrib_no_last_act_true)^2)), 1e-8
  )
})



test_that("DeepSHAP: Keras model with two inputs + two outputs (concat)", {
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

  # Check DeepSHAP with rescale rule and ignoring last activation
  data <- lapply(list(c(5), c(10,10,2)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_ref <- lapply(list(c(5), c(10,10,2)),
                  function(x) array(rnorm(10 * prod(x)), dim = c(5, x)))

  deepshap <- DeepSHAP$new(converter, data, data_ref = data_ref,
                           channels_first = FALSE, output_idx = list(c(2), c(1,3)))
  result <- deepshap$get_result()
  expect_equal(length(result), 2)
  expect_equal(length(result[[1]]), 2)
  expect_equal(dim(result[[1]][[1]]), c(10,5,1))
  expect_equal(dim(result[[1]][[2]]), c(10,10,10,2,1))
  expect_equal(length(result[[2]]), 2)
  expect_equal(dim(result[[2]][[1]]), c(10,5,2))
  expect_equal(dim(result[[2]][[2]]), c(10,10,10,2,2))

  # Check correctness of DeepSHAP rescale rule without ignoring the last
  # activation
  data <- lapply(list(c(5), c(10,10,2)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_ref <- lapply(list(c(5), c(10,10,2)),
                  function(x) array(rnorm(10 * prod(x)), dim = c(5, x)))
  deepshap <- DeepSHAP$new(converter, data, data_ref = data_ref, ignore_last_act = FALSE,
                           channels_first = FALSE, output_idx = list(c(2), c(1,3)))

  y <- converter$model(data, channels_first = FALSE)
  y_ref <- converter$model$update_ref(data_ref, channels_first = FALSE)
  contrib_true <- list(as.array(y[[1]][, 2] - mean(y_ref[[1]][, 2])),
                       as.array(y[[2]][, 1] - mean(y_ref[[2]][, 1])),
                       as.array(y[[2]][, 3] - mean(y_ref[[2]][, 3])))

  result <- deepshap$get_result("torch_tensor")
  contrib_1 <- as.array(result$Output_1$Input_1$sum(c(2,3)) +
                          result$Output_1$Input_2$sum(c(2,3,4,5)))
  contrib_2 <- as.array(result$Output_2$Input_1[,,1]$sum(c(2)) +
                          result$Output_2$Input_2[,,,,1]$sum(c(2,3,4)))
  contrib_3 <- as.array(result$Output_2$Input_1[,,2]$sum(c(2)) +
                          result$Output_2$Input_2[,,,,2]$sum(c(2,3,4)))

  expect_lt(mean((contrib_true[[1]] - contrib_1)^2), 1e-8)
  expect_lt(mean((contrib_true[[2]] - contrib_2)^2), 1e-8)
  expect_lt(mean((contrib_true[[3]] - contrib_3)^2), 1e-8)
})



test_that("DeepSHAP: Keras model with three inputs + one output (add)", {
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

  # Check DeepSHAP with rescale rule and ignoring last activation
  data <- lapply(list(c(12,15,3), c(20), c(10)),
                 function(x) torch_randn(c(10,x)))
  data_ref <- lapply(list(c(12,15,3), c(20), c(10)),
                  function(x) torch_randn(c(5,x)))

  deepshap <- DeepSHAP$new(converter, data, data_ref = data_ref,
                           channels_first = FALSE)
  result <- deepshap$get_result()
  expect_equal(length(result), 3)
  expect_equal(dim(result[[1]]), c(10,12,15,3,1))
  expect_equal(dim(result[[2]]), c(10,20,1))
  expect_equal(dim(result[[3]]), c(10,10,1))

  # Check correctness of DeepSHAP rescale rule without ignoring the last
  # activation
  data <- lapply(list(c(12,15,3), c(20), c(10)),
                 function(x) torch_randn(c(10,x)))
  data_ref <- lapply(list(c(12,15,3), c(20), c(10)),
                  function(x) torch_randn(c(5,x)))
  deepshap <- DeepSHAP$new(converter, data, data_ref = data_ref,
                           ignore_last_act = FALSE,
                           channels_first = FALSE)

  y <- converter$model(data, channels_first = FALSE)
  y_ref <- converter$model$update_ref(data_ref, channels_first = FALSE)
  contrib_true <- as.array(y[[1]] - mean(y_ref[[1]]))

  result <- deepshap$get_result("torch_tensor")
  contrib <- as.array(
    result$Input_1$sum(c(2,3,4,5)) +
      result$Input_2$sum(c(2,3)) +
      result$Input_3$sum(c(2,3)))

  expect_lt(mean((contrib_true - contrib)^2), 1e-8)
})

