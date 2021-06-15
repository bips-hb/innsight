
test_that("Test general errors",{
  expect_error(Analyzer$new(NULL))
  expect_error(Analyzer$new(NA))
  expect_error(Analyzer$new(c(3)))
  expect_error(Analyzer$new("124"))
})



test_that("Test neuralnet model", {
  library(neuralnet)
  library(torch)
  data(iris)
  #
  # --------------------- positive tests ---------------------------------------
  #

  nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
                  iris, linear.output = FALSE,
                  hidden = c(3,2), act.fct = "tanh", rep = 1)
  analyzer = Analyzer$new(nn)

  # forward method
  idx <- sample(nrow(iris), 10)
  y_true <- as.vector(predict(nn, iris))
  y <- as.vector(analyzer$forward(as.matrix(iris[,3:4])))
  expect_true(mean(abs(y_true - y)) < 1e-6)

  # update_ref method
  x_ref <- iris[sample(nrow(iris), 1), 3:4]
  y_true <- as.vector(predict(nn, x_ref))
  analyzer$update_ref(as.matrix(x_ref))
  expect_true(abs(y_true - as_array(rev(analyzer$model$modules_list)[[1]]$output_ref)) < 1e-6)

  # doesn't converge
  expect_warning(nn <- neuralnet(Species ~ .,
                                 iris, linear.output = TRUE,
                                 hidden = c(3,2), act.fct = "tanh", rep = 1, stepmax = 1e+01))
  expect_error(Analyzer$new(nn))

})


test_that("Test keras model", {

  library(keras)
  library(torch)

  #
  # --------------------- Dense Model -----------------------------------------
  #

  data <- matrix(rnorm(4*10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
    layer_dropout(0.1) %>%
    layer_dense(units = 8, activation = 'relu') %>%
    layer_dropout(0.1) %>%
    layer_dense(units = 3, activation = 'softmax')

  # test non-fitted model
  analyzer = Analyzer$new(model)

  # forward method
  y_true <- predict(model, data)
  y <- analyzer$forward(data)
  expect_true(mean(abs(y_true - y)) < 1e-6)

  # update
  x_ref <- matrix(rnorm(4), nrow=1, ncol=4)
  analyzer$update_ref(x_ref)
  y_true <- as.array(model(x_ref))
  y <- as_array(rev(analyzer$model$modules_list)[[1]]$output_ref)
  expect_true(mean(abs(y_true - y)) < 1e-6)

  ## other attributes
  # input dimension
  expect_equal(analyzer$input_dim, 4)
  # output dimension
  expect_equal(analyzer$output_dim, 3)

  analyzer$forward(data)

  for (module in analyzer$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }



  #
  # --------------------- CNN (1D) Model -----------------------------------------
  #

  data <- array(rnorm(64*128*4), dim = c(64,128,4))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(input_shape = c(128,4), kernel_size = 16, filters = 8, activation = "softplus") %>%
    layer_conv_1d(kernel_size = 16, filters = 4,  activation = "tanh") %>%
    layer_conv_1d(kernel_size = 16, filters = 2,  activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  analyzer = Analyzer$new(model)

  # forward method
  y_true <- predict(model, data)
  y <- analyzer$forward(data, channels_first = FALSE)
  expect_true(mean(abs(y_true - y)) < 1e-6)

  # update
  x_ref <- array(rnorm(128*4), dim=c(1,128,4))
  analyzer$update_ref(x_ref, channels_first = FALSE)
  y_true <- as.array(model(x_ref))
  y <- as_array(rev(analyzer$model$modules_list)[[1]]$output_ref)
  expect_true(mean(abs(y_true - y)) < 1e-6)

  ## other attributes
  # input dimension
  expect_equal(analyzer$input_dim, c(128,4))
  # output dimension
  expect_equal(analyzer$output_dim, 1)

  for (module in analyzer$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }

})
