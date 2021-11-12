
test_that("Test general errors", {
  library(torch)

  expect_error(Converter$new(NULL))
  expect_error(Converter$new(NA))
  expect_error(Converter$new(c(3)))
  expect_error(Converter$new("124"))

  model <- nn_sequential(
    nn_linear(5, 20),
    nn_relu(),
    nn_linear(20, 10, FALSE),
    nn_tanh(),
    nn_linear(10, 3),
    nn_softmax(2)
  )

  Converter$new(model, input_dim = c(5))
  expect_error(Converter$new(model, input_dim = c(4)))
  Converter$new(model, input_dim = c(5),
                input_names = list(c("a", "b", "c", "d", "e")))
  expect_error(Converter$new(model, input_dim = c(5),
                             input_names = list(c("a", "b", "c", "d"))))
  Converter$new(model, input_dim = c(5),
                output_names = list(c("a", "b", "c")))
  expect_error(Converter$new(model, input_dim = c(5),
                             output_names = list(c("a"))))

  # Test non classification output
  model <- NULL
  model$input_dim <- c(3,5,5)
  model$layers$Layer_1 <-
    list(
      type = "AveragePooling2D",
      strides = NULL,
      kernel_size = c(2,2)
    )
  expect_error(Converter$new(model))

  # Test for too many input dimensions
  model <- NULL
  model$input_dim <- c(3,5,5,5)
  model$layers$Layer_1 <-
    list(
      type = "Flatten"
    )
  expect_error(Converter$new(model))
})

test_that("Test non sequential model", {
  library(torch)

  net <- nn_module(
    "class_net",
    initialize = function() {
      self$linear1 <- nn_linear(4,8)
      self$linear2 <- nn_linear(8,16)
      self$linear3 <- nn_linear(16,3)
    },

    forward = function(x){
      x %>%
        self$linear1() %>%
        nnf_relu() %>%
        self$linear2() %>%
        nnf_relu() %>%
        self$linear3() %>%
        nnf_softmax(2)
    }
  )
  model <- net()

  expect_error(Converter$new(model))
})

test_that("Test torch sequential model: Dense", {
  library(torch)

  model <- nn_sequential(
    nn_linear(5, 20),
    nn_relu(),
    nn_linear(20, 10, FALSE),
    nn_tanh(),
    nn_linear(10, 1),
    nn_sigmoid()
  )
  input <- torch_randn(10, 5)

  expect_error(Converter$new(model))

  converter <- Converter$new(model, input_dim = c(5))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(input))

  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)

  # Dropout layer
  model <- nn_sequential(
    nn_linear(5, 20),
    nn_leaky_relu(),
    nn_linear(20, 10, FALSE),
    nn_tanh(),
    nn_dropout(),
    nn_linear(10, 1),
    nn_sigmoid()
  )

  Converter$new(model, input_dim = c(5))

  # Unknown layer
  model <- nn_sequential(
    nn_linear(5,5),
    nn_batch_norm1d(5))
  expect_error(Converter$new(model, input_dim = c(5)))

})

test_that("Test torch sequential model: 1D Conv", {
  library(torch)

  # See issue #716 (https://github.com/mlverse/torch/issues/716)
  nn_flatten <- nn_module(
    classname = "nn_flatten",
    initialize = function(start_dim = 2, end_dim = -1) {
      self$start_dim <- start_dim
      self$end_dim <- end_dim
    },
    forward = function(x) {
      torch_flatten(x, start_dim = self$start_dim, end_dim = self$end_dim)
    }
  )

  input <- torch_randn(10, 3, 100)

  model <- nn_sequential(
    nn_conv1d(3,10,10),
    nn_relu(),
    nn_conv1d(10,8,8, stride = 2),
    nn_softplus(),
    nn_conv1d(8,6,6, padding = 2),
    nn_softplus(),
    nn_max_pool1d(kernel_size = 1),
    nn_conv1d(6,4,4, dilation = 2),
    nn_softplus(),
    nn_avg_pool1d(kernel_size = 1),
    nn_conv1d(4,2,2, bias = FALSE),
    nn_softplus(),
    nn_flatten(),
    nn_linear(68, 32),
    nn_tanh(),
    nn_linear(32, 2)
  )

  expect_error(Converter$new(model))

  converter <- Converter$new(model, input_dim = c(3, 100))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(input))

  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)

  # Test failures
  # padding mode
  model <- nn_sequential(
    nn_conv1d(3,2,10, padding_mode = "reflect"),
    nn_relu(),
    nn_flatten(),
    nn_linear(22, 2)
  )
  expect_error(Converter$new(model, input_dim = c(3,20)))
  # padding for pooling layers
  model <- nn_sequential(
    nn_conv1d(3,2,10),
    nn_relu(),
    nn_avg_pool1d(2, padding = c(1)),
    nn_flatten(),
    nn_linear(12, 2)
  )
  expect_error(Converter$new(model, input_dim = c(3,20)))
  model <- nn_sequential(
    nn_conv1d(3,2,10),
    nn_relu(),
    nn_max_pool1d(2, padding = c(1)),
    nn_flatten(),
    nn_linear(12, 2)
  )
  expect_error(Converter$new(model, input_dim = c(3,20)))
})


test_that("Test torch sequential model: 2D Conv", {
  library(torch)

  # See issue #716 (https://github.com/mlverse/torch/issues/716)
  nn_flatten <- nn_module(
    classname = "nn_flatten",
    initialize = function(start_dim = 2, end_dim = -1) {
      self$start_dim <- start_dim
      self$end_dim <- end_dim
    },
    forward = function(x) {
      torch_flatten(x, start_dim = self$start_dim, end_dim = self$end_dim)
    }
  )

  input <- torch_randn(10, 3, 30, 30)

  model <- nn_sequential(
    nn_conv2d(3,10,5),
    nn_relu(),
    nn_conv2d(10,8,4, stride = c(2,1)),
    nn_relu(),
    nn_conv2d(8,8,4, stride = 2),
    nn_relu(),
    nn_conv2d(8,6,3, padding = c(5,4)),
    nn_relu(),
    nn_conv2d(6,6,3, padding = 3),
    nn_relu(),
    nn_conv2d(6,4,2, dilation = 2),
    nn_relu(),
    nn_conv2d(4,4,2, dilation = c(1,2)),
    nn_relu(),
    nn_conv2d(4,2,1, bias = FALSE),
    nn_relu(),
    nn_flatten(),
    nn_linear(448, 64),
    nn_linear(64, 2)
  )

  expect_error(Converter$new(model))

  converter <- Converter$new(model, input_dim = c(3, 30, 30))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(input))

  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)

  # Test failures
  # padding mode
  model <- nn_sequential(
    nn_conv2d(3,2,5, padding_mode = "reflect"),
    nn_relu(),
    nn_flatten(),
    nn_linear(72, 2)
  )
  expect_error(Converter$new(model, input_dim = c(3,10,10)))
  # padding for pooling layers
  model <- nn_sequential(
    nn_conv2d(3,2,5),
    nn_relu(),
    nn_avg_pool2d(2, padding = c(1)),
    nn_flatten(),
    nn_linear(32, 2)
  )
  expect_error(Converter$new(model, input_dim = c(3,10,10)))
  model <- nn_sequential(
    nn_conv2d(3,2,5),
    nn_relu(),
    nn_max_pool2d(2, padding = c(1)),
    nn_flatten(),
    nn_linear(32, 2)
  )
  expect_error(Converter$new(model, input_dim = c(3,10,10)))

})

test_that("Test torch sequential model: 2D Conv with pooling", {
  library(torch)

  # See issue #716 (https://github.com/mlverse/torch/issues/716)
  nn_flatten <- nn_module(
    classname = "nn_flatten",
    initialize = function(start_dim = 2, end_dim = -1) {
      self$start_dim <- start_dim
      self$end_dim <- end_dim
    },
    forward = function(x) {
      torch_flatten(x, start_dim = self$start_dim, end_dim = self$end_dim)
    }
  )

  input <- torch_randn(10, 3, 30, 30)

  model <- nn_sequential(
    nn_conv2d(3,10,5),
    nn_relu(),
    nn_avg_pool2d(c(2,2)),
    nn_relu(),
    nn_conv2d(10,8,4, padding = c(4, 5)),
    nn_relu(),
    nn_max_pool2d(c(2,2), stride = c(2,3)),
    nn_relu(),
    nn_flatten(),
    nn_linear(504, 64),
    nn_linear(64, 2)
  )

  expect_error(Converter$new(model))

  converter <- Converter$new(model, input_dim = c(3, 30, 30))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(input, TRUE, TRUE, TRUE, TRUE))

  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)

  x_ref <- array(rnorm(3 * 30 * 30), dim = c(1, 3, 30, 30))
  y_ref <- as.array(converter$model$update_ref(torch_tensor(x_ref)))
  dim_y_ref <- dim(y_ref)
  y_ref_true <- as.array(model(x_ref))
  dim_y_ref_true <- dim(y_ref_true)

  expect_equal(dim_y_ref_true, dim_y_ref)
  expect_lt(mean((y_ref - y_ref_true)^2), 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$model_dict$input_dim, c(3, 30, 30))
  # output dimension
  expect_equal(converter$model_dict$output_dim, 2)

  for (module in converter$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }

})

test_that("Test neuralnet model", {
  library(neuralnet)
  library(torch)

  data(iris)
  #
  # --------------------- positive tests --------------------------------------
  #
  nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
                  iris,
                  linear.output = TRUE,
                  hidden = c(3, 2), act.fct = "tanh", rep = 3
  )
  converter <- Converter$new(nn)

  nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
                  iris,
                  linear.output = FALSE,
                  hidden = c(3, 2), act.fct = "tanh", rep = 1
  )
  converter <- Converter$new(nn)

  # forward method
  y_true <- as.vector(predict(nn, iris))
  dim_y_true <- c(150, 1)
  y <- as.array(converter$model(torch_tensor(as.matrix(iris[, 3:4]))))
  dim_y <- dim(y)

  expect_equal(dim_y, dim_y_true)
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref method
  x_ref <- iris[sample(nrow(iris), 1), 3:4]
  y_ref_true <- as.vector(predict(nn, x_ref))
  y_ref <- as.array(converter$model$update_ref(torch_tensor(as.matrix(x_ref))))
  dim_y_ref <- dim(y_ref)
  expect_equal(dim_y_ref, c(1, 1))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  # doesn't converge
  expect_warning(
    nn_not_converged <-
      neuralnet(
        Species ~ ., iris, linear.output = TRUE, hidden = c(3, 2),
        act.fct = "tanh", rep = 1, stepmax = 1e+01
  ))
  expect_error(Converter$new(nn_not_converged))

  # custom activation function
  nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
                  iris,
                  linear.output = FALSE,
                  hidden = c(3, 2), act.fct = function(x) tanh(x), rep = 1
  )
  expect_error(Converter$new(nn))
})

test_that("Test list model: Dense", {
  library(torch)

  model <- NULL
  model$input_dim <- 5
  model$input_names <- list(c("Feat1", "Feat2", "Feat3", "Feat4", "Feat5"))
  model$output_dim <- 2
  model$output_names <- list(c("Cat", "no-Cat"))
  model$layers$Layer_1 <-
    list(
      type = "Dense",
      weight = matrix(rnorm(5 * 20), 20, 5),
      bias = rnorm(20),
      activation_name = "tanh",
      dim_in = 5L,
      dim_out = 20L
    )
  model$layers$Layer_2 <-
    list(
      type = "Dense",
      weight = matrix(rnorm(20 * 2), 2, 20),
      bias = rnorm(2),
      activation_name = "softmax",
      dim_in = 20L,
      dim_out = 2L
    )

  # Convert the model
  converter <- Converter$new(model)
  expect_true("Converter" %in% class(converter))

  # get the model
  model_torch <- converter$model

  # test output dimension
  input <- torch_randn(10,5)
  out <- model_torch(input)
  expect_equal(dim(out), c(10, 2))

  # Test failure
  # type
  model$layers$Layer_1$type <- "asdf"
  expect_error(Converter$new(model))
  model$layers$Layer_1$type <- "Dense"
  # weight
  model$layers$Layer_1$weight <- t(model$layers$Layer_1$weight)
  expect_error(Converter$new(model))
  model$layers$Layer_1$weight <- t(model$layers$Layer_1$weight)
  # activation function
  model$layers$Layer_1$activation_name <- "asdf"
  expect_error(Converter$new(model))

})


test_that("Test list model: 1D Convolution", {
  library(torch)

  model <- NULL
  model$input_dim <- c(3, 10)
  model$output_dim <- 2
  model$output_names <- list(c("Cat", "no-Cat"))
  model$layers$Layer_1 <-
    list(
      type = "Conv1D",
      weight = array(rnorm(8*3*2), dim = c(8,3,2)),
      bias = rnorm(8),
      activation_name = "tanh",
      dim_in = c(3L, 10L),
      dim_out = c(8L, 9L)
    )
  model$layers$Layer_2 <-
    list(
      type = "Conv1D",
      weight = array(rnorm(2*8*2), dim = c(2,8,2)),
      bias = rnorm(2),
      activation_name = "tanh",
      dim_in = c(8L, 9L),
      dim_out = c(2L, 8L)
    )
  model$layers$Layer_3 <-
    list(
      type = "Flatten",
      dim_in = c(2,8),
      dim_out = 16L
    )
  model$layers$Layer_4 <-
    list(
      type = "Dense",
      weight = matrix(rnorm(16 * 2), 2, 16),
      bias = rnorm(2),
      activation_name = "softmax",
      dim_in = 16,
      dim_out = 2L
    )

  # Convert the model
  converter <- Converter$new(model)
  expect_true("Converter" %in% class(converter))

  # get the model
  model_torch <- converter$model

  # test output dimension
  input <- torch_randn(10,3,10)
  out <- model_torch(input)
  expect_equal(dim(out), c(10, 2))

  # Test failures
  # Padding
  model$layers$Layer_1$padding <- c(1,2,3)
  expect_error(Converter$new(model))
  model$layers$Layer_1$padding <- NULL
  # Forward pass
  dim(model$layers$Layer_1$weight) <- c(3,8,2)
  expect_error(Converter$new(model))


})

test_that("Test list model: 2D Convolution", {
  library(torch)

  model <- NULL
  model$input_dim <- c(3, 10, 10)
  model$output_dim <- 2
  model$output_names <- list(c("Cat", "no-Cat"))
  model$layers$Layer_1 <-
    list(
      type = "Conv2D",
      weight = array(rnorm(8*3*2*2), dim = c(8,3,2,2)),
      bias = rnorm(8),
      activation_name = "tanh",
      dim_in = c(3L, 10L, 10L),
      dim_out = c(8L, 9L, 9L)
    )
  model$layers$Layer_2 <-
    list(
      type = "Conv2D",
      weight = array(rnorm(2*8*2*2), dim = c(2,8,2,2)),
      bias = rnorm(2),
      padding = c(1),
      stride = c(1),
      activation_name = "tanh",
      dim_in = c(8L, 9L, 9L),
      dim_out = c(2L, 10L, 10L)
    )
  model$layers$Layer_3 <-
    list(
      type = "Conv2D",
      weight = array(rnorm(2*2*2*2), dim = c(2,2,2,2)),
      bias = rnorm(2),
      padding = c(1, 0),
      dilation = c(1),
      activation_name = "tanh",
      dim_in = c(2L, 10L, 10L),
      dim_out = c(2L, 9L, 11L)
    )
  model$layers$Layer_4 <-
    list(
      type = "Flatten",
      dim_in = c(2,9,11),
      dim_out = 198
    )
  model$layers$Layer_5 <-
    list(
      type = "Dense",
      weight = matrix(rnorm(198 * 2), 2, 198),
      bias = rnorm(2),
      activation_name = "softmax",
      dim_in = 198,
      dim_out = 2L
    )

  # Convert the model
  converter <- Converter$new(model)
  expect_true("Converter" %in% class(converter))

  # get the model
  model_torch <- converter$model

  # test output dimension
  input <- torch_randn(10,3,10,10)
  out <- model_torch(input)
  expect_equal(dim(out), c(10, 2))

  # Test failures
  # padding
  model$layers$Layer_1$padding <- c(1,2,3)
  expect_error(Converter$new(model))
  model$layers$Layer_1$padding <- NULL
  # stride
  model$layers$Layer_1$stride <- c(1,2,3)
  expect_error(Converter$new(model))
  model$layers$Layer_1$stride <- NULL
  # dilation
  model$layers$Layer_1$dilation <- c(1,2,3)
  expect_error(Converter$new(model))
  model$layers$Layer_1$dilation <- NULL
  # Forward pass
  dim(model$layers$Layer_1$weight) <- c(3,8,2,2)
  expect_error(Converter$new(model))
})



test_that("Test keras model: Dense", {
  library(keras)
  library(torch)

  data <- matrix(rnorm(4 * 10), nrow = 10)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(4)) %>%
    layer_dropout(0.1) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(units = 3, activation = "softmax")

  # test non-fitted model
  converter <- Converter$new(model)

  # forward method
  y_true <- as.array(model(data))
  dim_y_true <- dim(y_true)
  y <- as.array(converter$model(torch_tensor(data)))
  dim_y <- dim(y)

  expect_equal(dim_y, dim_y_true)
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref <- matrix(rnorm(4), nrow = 1, ncol = 4)
  y_ref <- as.array(converter$model$update_ref(torch_tensor(x_ref)))
  dim_y_ref <- dim(y_ref)
  y_ref_true <- as.array(model(x_ref))
  dim_y_ref_true <- dim(y_ref_true)

  expect_equal(dim_y_ref, dim_y_ref_true)
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

  ## other attributes
  # input dimension
  converter_input_dim <- converter$model_dict$input_dim
  expect_equal(converter_input_dim, 4)
  # output dimension
  converter_output_dim <- converter$model_dict$output_dim
  expect_equal(converter_output_dim, 3)

  converter$model(torch_tensor(data), TRUE, TRUE, TRUE, TRUE)

  for (module in converter$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }
})

test_that("Test keras model: General", {
  library(keras)

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(3, 2, input_shape = c(10,3)) %>%
    layer_average_pooling_1d(padding = "same") %>%
    layer_flatten()

  expect_error(Converter$new(model))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(3, c(3,4), strides = 2, input_shape = c(10,10,3),
                  padding = "same") %>%
    layer_conv_2d(3, c(1,2), strides = 2, padding = "same") %>%
    layer_flatten()
  c <- Converter$new(model)
})


test_that("Test keras model: Conv1D with 'valid' padding", {
  library(keras)
  library(torch)

  data <- array(rnorm(64 * 128 * 4), dim = c(64, 128, 4))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(128, 4), kernel_size = 16, filters = 8,
      activation = "softplus"
    ) %>%
    layer_max_pooling_1d() %>%
    layer_conv_1d(kernel_size = 16, filters = 4, activation = "tanh") %>%
    layer_average_pooling_1d() %>%
    layer_conv_1d(kernel_size = 16, filters = 2, activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)

  # forward method
  y_true <- as.array(model(data))
  dim_y_true <- dim(y_true)
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE,
                                TRUE, TRUE, TRUE))
  dim_y <- dim(y)

  expect_equal(dim_y, dim_y_true)
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref <- array(rnorm(128 * 4), dim = c(1, 128, 4))
  y_ref <- as.array(converter$model$update_ref(torch_tensor(x_ref),
                                               channels_first = FALSE
  ))
  dim_y_ref <- dim(y_ref)
  y_ref_true <- as.array(model(x_ref))
  dim_y_ref_true <- dim(y_ref_true)

  expect_equal(dim_y_ref_true, dim_y_ref)
  expect_lt(mean((y_ref - y_ref_true)^2), 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$model_dict$input_dim, c(4, 128))
  # output dimension
  expect_equal(converter$model_dict$output_dim, 1)

  for (module in converter$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }
})

test_that("Test keras model: Conv1D with 'same' padding", {
  library(keras)
  library(torch)

  data <- array(rnorm(64 * 128 * 4), dim = c(64, 128, 4))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(128, 4), kernel_size = 16, filters = 8,
      activation = "softplus", padding = "same"
    ) %>%
    layer_conv_1d(
      kernel_size = 16, filters = 4, activation = "tanh",
      padding = "same"
    ) %>%
    layer_conv_1d(
      kernel_size = 16, filters = 2, activation = "relu",
      padding = "same"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)

  # forward method
  y_true <- as.array(model(data))
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE,
                                TRUE, TRUE, TRUE))
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(128 * 4), dim = c(1, 128, 4))
  y_ref <-
    as.array(converter$model$update_ref(torch_tensor(x_ref),
                                        channels_first = FALSE
    ))
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$model_dict$input_dim, c(4, 128))
  # output dimension
  expect_equal(converter$model_dict$output_dim, 1)

  for (module in converter$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }
})


test_that("Test keras model: Conv2D with 'valid' padding", {
  library(keras)
  library(torch)

  data <- array(rnorm(64 * 32 * 32 * 3), dim = c(64, 32, 32, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
      activation = "softplus", padding = "valid"
    ) %>%
    layer_max_pooling_2d() %>%
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh",
      padding = "valid"
    ) %>%
    layer_average_pooling_2d(pool_size = c(1,1)) %>%
    layer_conv_2d(
      kernel_size = 4, filters = 2, activation = "relu",
      padding = "valid"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)

  # forward method
  y_true <- as.array(model(data))
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE,
                                TRUE, TRUE, TRUE))
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))
  y_ref <- as.array(converter$model$update_ref(torch_tensor(x_ref),
                                               channels_first = FALSE
  ))
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$model_dict$input_dim, c(3, 32, 32))
  # output dimension
  expect_equal(converter$model_dict$output_dim, 1)

  for (module in converter$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }
})

test_that("Test keras model: Conv2D with 'same' padding", {
  library(keras)
  library(torch)

  data <- array(rnorm(64 * 32 * 32 * 3), dim = c(64, 32, 32, 3))

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
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)

  # forward method
  y_true <- as.array(model(data))
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE,
                                TRUE, TRUE, TRUE))
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))
  y_ref <- as.array(converter$model$update_ref(torch_tensor(x_ref),
                                               channels_first = FALSE
  ))
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$model_dict$input_dim, c(3, 32, 32))
  # output dimension
  expect_equal(converter$model_dict$output_dim, 1)

  for (module in converter$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }
})

test_that("Test keras model: CNN with average pooling", {
  library(torch)
  library(keras)

  data <- array(rnorm(64 * 32 * 32 * 3), dim = c(64, 32, 32, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 4, filters = 8,
      activation = "softplus", padding = "valid"
    ) %>%
    layer_average_pooling_2d(strides = 3) %>%
    layer_conv_2d(
      kernel_size = 4, filters = 4, activation = "tanh",
      padding = "valid"
    ) %>%
    layer_average_pooling_2d(pool_size = c(1, 3)) %>%
    layer_conv_2d(
      kernel_size = 2, filters = 2, activation = "relu",
      padding = "valid"
    ) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  converter <- Converter$new(model)

  # forward method
  y_true <- as.array(model(data))
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE,
                                TRUE, TRUE, TRUE))
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))
  y_ref <- as.array(converter$model$update_ref(torch_tensor(x_ref),
                                               channels_first = FALSE
  ))
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$model_dict$input_dim, c(3, 32, 32))
  # output dimension
  expect_equal(converter$model_dict$output_dim, 1)

  for (module in converter$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }

})
