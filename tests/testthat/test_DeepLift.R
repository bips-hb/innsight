
test_that("DeepLift: General errors",{

  data <- matrix(rnorm(4*10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
    layer_dense(units = 8, activation = 'relu') %>%
    layer_dense(units = 3, activation = 'softmax')

  analyzer <- Analyzer$new(model)

  expect_error(DeepLift$new(model, data))
  expect_error(DeepLift$new(analyzer, model))
  expect_error(DeepLift$new(analyzer, data, channels_first = NULL))
  expect_error(DeepLift$new(analyzer, data, rule_name = "asdf"))
  expect_error(DeepLift$new(analyzer, data, rule_param = "asdf"))
  expect_error(DeepLift$new(analyzer, data, dtype = NULL))
  expect_error(DeepLift$new(analyzer, data, ignore_last_act = c(1)))
})

test_that("DeepLift: Dense-Net",{

  data <- matrix(rnorm(4*10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
    layer_dense(units = 8, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax')

  analyzer = Analyzer$new(model)
  x_ref <- matrix(rnorm(4), nrow = 1)

  # Rescale Rule
  d <- DeepLift$new(analyzer, data, x_ref = x_ref, dtype = "double", ignore_last_act = FALSE)

  last_layer <- rev(analyzer$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <- last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- analyzer$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deeplift_rescale <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rescale), c(10,4,3))
  expect_lt(as.array(mean(abs(deeplift_rescale$sum(dim = 2) - contrib_true)^2)), 1e-12)

  d <- DeepLift$new(analyzer, data, x_ref = x_ref,  dtype = "float", ignore_last_act = TRUE)
  deeplift_rescale_no_last_act <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rescale_no_last_act), c(10,4,3))
  expect_lt(as.array(mean(abs(deeplift_rescale_no_last_act$sum(dim = 2) - contrib_no_last_act_true)^2)), 1e-12)

  # Reveal-Cancel rule
  d <- DeepLift$new(analyzer, data, x_ref = x_ref,  dtype = "float", ignore_last_act = FALSE, rule_name = "reveal_cancel")
  deeplift_rc <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rc), c(10,4,3))
  #expect_lt(as.array(mean(abs(deeplift_rc$sum(dim = 2) - contrib_true)^2)), 1e-12)

  d <- DeepLift$new(analyzer, data, x_ref = x_ref,  dtype = "double", ignore_last_act = TRUE,  rule_name = "reveal_cancel")
  deeplift_rc_no_last_act <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rc_no_last_act), c(10,4,3))
  expect_lt(as.array(mean(abs(deeplift_rc_no_last_act$sum(dim = 2) - contrib_no_last_act_true)^2)), 1e-12)
})

test_that("DeepLift: Conv1D-Net",{

  data <- array(rnorm(4*64*3), dim = c(4,64,3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(input_shape = c(64,3), kernel_size = 16, filters = 8, activation = "softplus") %>%
    layer_conv_1d(kernel_size = 16, filters = 4,  activation = "tanh") %>%
    layer_conv_1d(kernel_size = 16, filters = 2,  activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  analyzer = Analyzer$new(model)
  x_ref <- array(rnorm(64*3), dim = c(1,64,3))

  d <- DeepLift$new(analyzer, data, x_ref = x_ref,  dtype = "double", channels_first = FALSE, ignore_last_act = FALSE)

  last_layer <- rev(analyzer$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <- last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- analyzer$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deeplift_rescale <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rescale), c(4,64,3,1))
  expect_lt(as.array(mean(abs(deeplift_rescale$sum(dim = c(2,3)) - contrib_true)^2)), 1e-12)

  d <- DeepLift$new(analyzer, data, x_ref = x_ref,  dtype = "float", ignore_last_act = TRUE, channels_first = FALSE)
  deeplift_rescale_no_last_act <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rescale_no_last_act), c(4,64,3,1))
  expect_lt(as.array(mean(abs(deeplift_rescale_no_last_act$sum(dim = c(2,3)) - contrib_no_last_act_true)^2)), 1e-12)

  # Reveal-Cancel rule
  d <- DeepLift$new(analyzer, data, x_ref = x_ref, dtype = "float", ignore_last_act = FALSE, rule_name = "reveal_cancel", channels_first = FALSE)
  deeplift_rc <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rc), c(4,64,3,1))
  #expect_lt(as.array(mean(abs(deeplift_rc$sum(dim = c(2,3)) - contrib_true)^2)), 1e-12)

  d <- DeepLift$new(analyzer, data, x_ref = x_ref, dtype = "double", ignore_last_act = TRUE,  rule_name = "reveal_cancel", channels_first = FALSE)
  deeplift_rc_no_last_act <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rc_no_last_act), c(4,64,3,1))
  expect_lt(as.array(mean(abs(deeplift_rc_no_last_act$sum(dim = c(2,3)) - contrib_no_last_act_true)^2)), 1e-12)
})

test_that("DeepLift: Conv2D-Net",{

  data <- array(rnorm(4*32*32*3), dim = c(4,32,32,3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(input_shape = c(32,32,3), kernel_size = 8, filters = 8, activation = "softplus", padding = "same") %>%
    layer_conv_2d(kernel_size = 8, filters = 4,  activation = "tanh", padding = "same") %>%
    layer_conv_2d(kernel_size = 4, filters = 2,  activation = "relu", padding = "same") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 2, activation = "softmax")

  # test non-fitted model
  analyzer = Analyzer$new(model)

  analyzer = Analyzer$new(model)
  x_ref <- array(rnorm(32*32*3), dim = c(1,32,32,3))

  d <- DeepLift$new(analyzer, data, x_ref = x_ref,  dtype = "double", channels_first = FALSE, ignore_last_act = FALSE)

  last_layer <- rev(analyzer$model$modules_list)[[1]]
  contrib_true <- last_layer$output - last_layer$output_ref
  contrib_no_last_act_true <- last_layer$preactivation - last_layer$preactivation_ref

  first_layer <- analyzer$model$modules_list[[1]]
  input_diff <- (first_layer$input - first_layer$input_ref)$unsqueeze(-1)


  deeplift_rescale <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rescale), c(4,32,32,3,2))
  expect_lt(as.array(mean(abs(deeplift_rescale$sum(dim = c(2,3,4)) - contrib_true)^2)), 1e-12)

  d <- DeepLift$new(analyzer, data, x_ref = x_ref,  dtype = "float", ignore_last_act = TRUE, channels_first = FALSE)
  deeplift_rescale_no_last_act <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rescale_no_last_act), c(4,32,32,3,2))
  expect_lt(as.array(mean(abs(deeplift_rescale_no_last_act$sum(dim = c(2,3,4)) - contrib_no_last_act_true)^2)), 1e-12)

  # Reveal-Cancel rule
  d <- DeepLift$new(analyzer, data, x_ref = x_ref, dtype = "float", ignore_last_act = FALSE, rule_name = "reveal_cancel", channels_first = FALSE)
  deeplift_rc <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rc), c(4,32,32,3,2))
  #expect_lt(as.array(mean(abs(deeplift_rc$sum(dim = c(2,3)) - contrib_true)^2)), 1e-12)

  d <- DeepLift$new(analyzer, data, x_ref = x_ref, dtype = "double", ignore_last_act = TRUE,  rule_name = "reveal_cancel", channels_first = FALSE)
  deeplift_rc_no_last_act <- d$get_result(as_torch = TRUE)

  expect_equal(dim(deeplift_rc_no_last_act), c(4,32,32,3,2))
  expect_lt(as.array(mean(abs(deeplift_rc_no_last_act$sum(dim = c(2,3,4)) - contrib_no_last_act_true)^2)), 1e-12)
})
