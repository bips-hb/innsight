
test_that("Connection_Weights: General errors",{

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
    layer_dense(units = 8, activation = 'relu') %>%
    layer_dense(units = 3, activation = 'softmax')

  analyzer <- Analyzer$new(model)

  expect_error(Connection_Weights$new(model))
  expect_error(Connection_Weights$new(analyzer, channels_first = NULL))
  expect_error(Connection_Weights$new(analyzer, dtype = "asdf"))
})


test_that("Connection_Weights: Dense-Net",{

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
    layer_dense(units = 8, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax')

  analyzer = Analyzer$new(model)

  cw <- Connection_Weights$new(analyzer)
  expect_equal(dim(cw$get_result()), c(4,3))
  expect_true(cw$get_result(TRUE)$dtype == torch_float())

  cw_channels_last <- Connection_Weights$new(analyzer, channels_first = FALSE)
  expect_equal(dim(cw_channels_last$get_result()), c(4,3))
  expect_true(cw_channels_last$get_result(TRUE)$dtype == torch_float())

  cw_double <- Connection_Weights$new(analyzer, dtype = 'double')
  expect_equal(dim(cw$get_result()), c(4,3))
  expect_true(cw_double$get_result(TRUE)$dtype == torch_double())

  cw_channels_last_double <- Connection_Weights$new(analyzer, channels_first = FALSE, dtype = 'double')
  expect_equal(dim(cw_channels_last_double$get_result()), c(4,3))
  expect_true(cw_channels_last_double$get_result(TRUE)$dtype == torch_double())

})

test_that("Connection_Weights: Conv1D-Net",{

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(input_shape = c(64,3), kernel_size = 16, filters = 8, activation = "softplus") %>%
    layer_conv_1d(kernel_size = 16, filters = 4,  activation = "tanh") %>%
    layer_conv_1d(kernel_size = 16, filters = 2,  activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  analyzer = Analyzer$new(model)

  cw <- Connection_Weights$new(analyzer)
  expect_equal(dim(cw$get_result()), c(3,64,1))
  expect_true(cw$get_result(TRUE)$dtype == torch_float())

  cw_channels_last <- Connection_Weights$new(analyzer, channels_first = FALSE)
  expect_equal(dim(cw_channels_last$get_result()), c(64,3,1))
  expect_true(cw_channels_last$get_result(TRUE)$dtype == torch_float())

  cw_double <- Connection_Weights$new(analyzer, dtype = 'double')
  expect_equal(dim(cw$get_result()), c(3,64,1))
  expect_true(cw_double$get_result(TRUE)$dtype == torch_double())

  cw_channels_last_double <- Connection_Weights$new(analyzer, channels_first = FALSE, dtype = 'double')
  expect_equal(dim(cw_channels_last_double$get_result()), c(64,3,1))
  expect_true(cw_channels_last_double$get_result(TRUE)$dtype == torch_double())
})

test_that("Connection_Weights: Conv2D-Net",{

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(input_shape = c(32,32,3), kernel_size = 8, filters = 8, activation = "softplus", padding = "same") %>%
    layer_conv_2d(kernel_size = 8, filters = 4,  activation = "tanh", padding = "same") %>%
    layer_conv_2d(kernel_size = 4, filters = 2,  activation = "relu", padding = "same") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 2, activation = "sigmoid")

  analyzer = Analyzer$new(model)

  cw <- Connection_Weights$new(analyzer)
  expect_equal(dim(cw$get_result()), c(3,32,32,2))
  expect_true(cw$get_result(TRUE)$dtype == torch_float())

  cw_channels_last <- Connection_Weights$new(analyzer, channels_first = FALSE)
  expect_equal(dim(cw_channels_last$get_result()), c(32,32,3,2))
  expect_true(cw_channels_last$get_result(TRUE)$dtype == torch_float())

  cw_double <- Connection_Weights$new(analyzer, dtype = 'double')
  expect_equal(dim(cw$get_result()), c(3,32,32,2))
  expect_true(cw_double$get_result(TRUE)$dtype == torch_double())

  cw_channels_last_double <- Connection_Weights$new(analyzer, channels_first = FALSE, dtype = 'double')
  expect_equal(dim(cw_channels_last_double$get_result()), c(32,32,3,2))
  expect_true(cw_channels_last_double$get_result(TRUE)$dtype == torch_double())
})
