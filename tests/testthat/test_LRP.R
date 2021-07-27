
test_that("LRP: General errors",{

  data <- matrix(rnorm(4*10), nrow = 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
    layer_dense(units = 8, activation = 'relu') %>%
    layer_dense(units = 3, activation = 'softmax')

  analyzer <- Analyzer$new(model)

  expect_error(LRP$new(model, data))
  expect_error(LRP$new(analyzer, model))
  expect_error(LRP$new(analyzer, data, channels_first = NULL))
  expect_error(LRP$new(analyzer, data, rule_name = "asdf"))
  expect_error(LRP$new(analyzer, data, rule_param = "asdf"))
  expect_error(LRP$new(analyzer, data, dtype = NULL))
})


test_that("LRP: Dense-Net",{

  data <- matrix(rnorm(4*10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
    layer_dense(units = 8, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax')

  analyzer = Analyzer$new(model)

  expect_error(LRP$new(analyzer, array(rnorm(4*2*3), dim = c(2,3,4))))

  # Simple Rule
  lrp_simple <- LRP$new(analyzer, data)
  expect_equal(dim(lrp_simple$get_result()), c(10,4,3))
  expect_true(lrp_simple$get_result(as_torch = TRUE)$dtype == torch_float())

  # Epsilon Rule
  lrp_eps_default <- LRP$new(analyzer, data, rule_name = "epsilon", dtype = "double")
  expect_equal(dim(lrp_eps_default$get_result()), c(10,4,3))
  expect_true(lrp_eps_default$get_result(as_torch = TRUE)$dtype == torch_double())

  lrp_eps_1 <- LRP$new(analyzer, data, rule_name = "epsilon", rule_param = 1)
  expect_equal(dim(lrp_eps_1$get_result()), c(10,4,3))
  expect_true(lrp_eps_1$get_result(as_torch = TRUE)$dtype == torch_float())

  # Alpha-Beta Rule
  lrp_ab_default <- LRP$new(analyzer, data, rule_name = "epsilon", dtype = "double")
  expect_equal(dim(lrp_ab_default$get_result()), c(10,4,3))
  expect_true(lrp_ab_default$get_result(as_torch = TRUE)$dtype == torch_double())

  lrp_ab_2 <- LRP$new(analyzer, data, rule_name = "epsilon", rule_param = 2)
  expect_equal(dim(lrp_ab_2$get_result()), c(10,4,3))
  expect_true(lrp_ab_2$get_result(as_torch = TRUE)$dtype == torch_float())
})

test_that("LRP: Conv1D-Net",{

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

  expect_error(LRP$new(analyzer, array(rnorm(4*2*3), dim = c(2,3,4))))

  # Simple Rule
  lrp_simple <- LRP$new(analyzer, data, channels_first = FALSE)
  expect_equal(dim(lrp_simple$get_result()), c(4,64,3,1))
  expect_true(lrp_simple$get_result(as_torch = TRUE)$dtype == torch_float())

  # Epsilon Rule
  lrp_eps_default <- LRP$new(analyzer, data, rule_name = "epsilon", dtype = "double", channels_first = FALSE)
  expect_equal(dim(lrp_eps_default$get_result()), c(4,64,3,1))
  expect_true(lrp_eps_default$get_result(as_torch = TRUE)$dtype == torch_double())

  lrp_eps_1 <- LRP$new(analyzer, data, rule_name = "epsilon", rule_param = 1, channels_first = FALSE)
  expect_equal(dim(lrp_eps_1$get_result()), c(4,64,3,1))
  expect_true(lrp_eps_1$get_result(as_torch = TRUE)$dtype == torch_float())

  # Alpha-Beta Rule
  lrp_ab_default <- LRP$new(analyzer, data, rule_name = "epsilon", dtype = "double", channels_first = FALSE)
  expect_equal(dim(lrp_ab_default$get_result()), c(4,64,3,1))
  expect_true(lrp_ab_default$get_result(as_torch = TRUE)$dtype == torch_double())

  lrp_ab_2 <- LRP$new(analyzer, data, rule_name = "epsilon", rule_param = 2, channels_first = FALSE)
  expect_equal(dim(lrp_ab_2$get_result()), c(4,64,3,1))
  expect_true(lrp_ab_2$get_result(as_torch = TRUE)$dtype == torch_float())
})

test_that("LRP: Conv2D-Net",{

  data <- array(rnorm(4*32*32*3), dim = c(4,32,32,3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(input_shape = c(32,32,3), kernel_size = 8, filters = 8, activation = "softplus", padding = "same") %>%
    layer_conv_2d(kernel_size = 8, filters = 4,  activation = "tanh", padding = "same") %>%
    layer_conv_2d(kernel_size = 4, filters = 2,  activation = "relu", padding = "same") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 2, activation = "sigmoid")

  # test non-fitted model
  analyzer = Analyzer$new(model)

  expect_error(LRP$new(analyzer, array(rnorm(4*32*31,3), dim = c(4,32,31,3)), channels_first = FALSE))

  # Simple Rule
  lrp_simple <- LRP$new(analyzer, data, channels_first = FALSE)
  expect_equal(dim(lrp_simple$get_result()), c(4,32,32,3,2))
  expect_true(lrp_simple$get_result(as_torch = TRUE)$dtype == torch_float())

  # Epsilon Rule
  lrp_eps_default <- LRP$new(analyzer, data, rule_name = "epsilon", dtype = "double", channels_first = FALSE)
  expect_equal(dim(lrp_eps_default$get_result()), c(4,32,32,3,2))
  expect_true(lrp_eps_default$get_result(as_torch = TRUE)$dtype == torch_double())

  lrp_eps_1 <- LRP$new(analyzer, data, rule_name = "epsilon", rule_param = 1, channels_first = FALSE)
  expect_equal(dim(lrp_eps_1$get_result()), c(4,32,32,3,2))
  expect_true(lrp_eps_1$get_result(as_torch = TRUE)$dtype == torch_float())

  # Alpha-Beta Rule
  lrp_ab_default <- LRP$new(analyzer, data, rule_name = "epsilon", dtype = "double", channels_first = FALSE)
  expect_equal(dim(lrp_ab_default$get_result()), c(4,32,32,3,2))
  expect_true(lrp_ab_default$get_result(as_torch = TRUE)$dtype == torch_double())

  lrp_ab_2 <- LRP$new(analyzer, data, rule_name = "epsilon", rule_param = 2, channels_first = FALSE)
  expect_equal(dim(lrp_ab_2$get_result()), c(4,32,32,3,2))
  expect_true(lrp_ab_2$get_result(as_torch = TRUE)$dtype == torch_float())
})


test_that("LRP: Correctness", {
  data <- array(rnorm(10*32*32*3), dim = c(10,32,32,3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(input_shape = c(32,32,3), kernel_size = 8, filters = 8, activation = "softplus", padding = "valid", use_bias = FALSE) %>%
    layer_conv_2d(kernel_size = 8, filters = 4,  activation = "tanh", padding = "valid", use_bias = FALSE) %>%
    layer_conv_2d(kernel_size = 4, filters = 2,  activation = "relu", padding = "valid", use_bias = FALSE) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 16, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE)

  # test non-fitted model
  analyzer = Analyzer$new(model)

  lrp <- LRP$new(analyzer, data, channels_first = FALSE)
  out <- analyzer$model$modules_list[[7]]$preactivation
  lrp_result_sum <- lrp$get_result(as_torch = TRUE)$sum(dim = c(2,3,4))
  expect_lt(as.array(mean(abs(lrp_result_sum - out)^2)), 1e-5)
})
