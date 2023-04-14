
################################################################################
#                            General Errors
################################################################################

test_that("Test general errors", {
  library(torch)

  expect_error(Converter$new(dtype = "adsf"))  # dtype
  expect_error(Converter$new(save_model_as_list = "No")) # save_model_as_list
  expect_error(Converter$new(NULL)) # not torch, keras or neuralnet
  expect_error(Converter$new(c(3))) # not torch, keras or neuralnet

  layers <- list(list(type = "Dense", weight = matrix(c(2), 1,2), bias = 1,
                      activation_name = "relu"))

  # No entry 'layers'
  model <- list(NULL)
  expect_error(Converter$new(model))
  # No entry 'input_dim'
  model <- list(layers = layers)
  expect_error(Converter$new(model))
  # 'input_dim' not as numeric
  model <- list(layers = layers, input_dim = c("as"))
  expect_error(Converter$new(model))
  # 'input_layers' missing
  tmp_layers <- layers
  tmp_layers[[1]]$output_layers <- -1
  model <- list(layers = tmp_layers, input_dim = list(c(2)), input_nodes = 1,
                output_nodes = 1)
  expect_warning(Converter$new(model))
  # 'output_layers' missing
  tmp_layers <- layers
  tmp_layers[[1]]$input_layers <- 0
  model <- list(layers = tmp_layers, input_dim = list(c(2)), input_nodes = 1,
                output_nodes = 1)
  expect_warning(Converter$new(model))
  # 'input_nodes' missing
  tmp_layers[[1]]$output_layers <- -1
  model <- list(layers = tmp_layers, input_dim = list(c(2)), output_nodes = 1)
  expect_warning(Converter$new(model))
  # 'output_nodes' missing
  model <- list(layers = tmp_layers, input_dim = list(c(2)), input_nodes = 1)
  expect_warning(Converter$new(model))
  # 'input_nodes' out of range
  model <- list(layers = tmp_layers, input_dim = list(c(2)), input_nodes = 2)
  expect_error(Converter$new(model))
  # 'input_nodes' wrong
  model <- list(layers = tmp_layers, input_dim = list(c(2)),
                input_nodes = "asdf")
  expect_error(Converter$new(model))
  # 'output_nodes' out of range
  model <- list(layers = tmp_layers, input_dim = list(c(2)),
                input_nodes = c(1), output_nodes = c(3))
  expect_error(Converter$new(model))
  # 'output_nodes' wrong
  model <- list(layers = tmp_layers, input_dim = list(c(2)),
                input_nodes = c(1), output_nodes = list(c("a")))
  expect_error(Converter$new(model))
  # 'output_dim' not numeric
  model <- list(layers = tmp_layers, input_dim = list(c(2)),
                input_nodes = c(1), output_nodes = c(1), output_dim = "adf")
  expect_error(Converter$new(model))
  # 'input_names' not characters
  model <- list(layers = tmp_layers, input_dim = list(c(2)),
                input_nodes = c(1), output_nodes = c(1), input_names = c(1,2,3))
  expect_error(Converter$new(model))
  # 'output_names' not characters
  model <- list(layers = tmp_layers, input_dim = list(c(2)),
                input_nodes = c(1), output_nodes = c(1), output_names = c(1,2,3))
  expect_error(Converter$new(model))

  # Define model
  create_model <- function(type, input_layers = NULL, output_layers = NULL) {
    list(
      input_dim = c(2),
      input_nodes = c(1),
      output_nodes = c(1),
      layers = list(
        list(
          type = type,
          weight = array(rnorm(2*3), dim = c(3,2)),
          bias = rnorm(3),
          activation_name = "relu",
          input_layers = input_layers,
          output_layers = output_layers
        )
      )
    )
  }

  # Checks for converting layers

  # 'type' wrong
  expect_error(Converter$new(create_model("asd")))
  # 'input_layers' missing
  expect_warning(expect_warning(Converter$new(create_model("Dense"))))
  # 'input_layers' wrong
  expect_error(Converter$new(create_model("Dense", "sadf")))
  # 'output_layers' missing
  expect_warning(Converter$new(create_model("Dense", c(0))))
  # 'output_layers' wrong
  expect_error(Converter$new(create_model("Dense", c(0), NA)))
  # 'output_dim' wrong
  model <- create_model("Dense", c(0), c(-1))
  model$output_dim <- c(2)
  expect_error(Converter$new(model))
  # Test non classification/regression output
  model <- NULL
  model$input_dim <- c(3,5,5)
  model$input_nodes <- c(1)
  model$output_nodes <- c(1)
  model$layers$Layer_1 <-
    list(
      type = "AveragePooling2D",
      strides = NULL,
      kernel_size = c(2,2),
      input_layers = 0,
      output_layers = -1
    )
  expect_error(Converter$new(model))
  # 'input_names' wrong dimensions
  model <- create_model("Dense", c(0), c(-1))
  model$input_names <- c("A", "B", "C")
  expect_error(Converter$new(model))
  # 'output_names' wrong dimensions
  model <- create_model("Dense", c(0), c(-1))
  model$output_names <- c("A", "B")
  expect_error(Converter$new(model))

  # Without error but saving model as list
  model <- create_model("Dense", c(0), c(-1))
  conv <- Converter$new(model, save_model_as_list = TRUE)

  # Test for too many input dimensions
  model <- NULL
  model$input_dim <- c(3,5,5,5)
  model$input_nodes <- c(1)
  model$output_nodes <- c(1)
  model$layers$Layer_1 <-
    list(
      type = "Flatten",
      input_layers = 0,
      output_layers = -1
    )
  expect_error(Converter$new(model))
})

test_that("Torch: Test non sequential model", {
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



################################################################################
#                            Package: Neuralnet
################################################################################
test_that("Test package Neuralnet", {
  library(neuralnet)
  library(torch)

  data(iris)
  nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
                  iris,
                  linear.output = FALSE,
                  hidden = c(3, 2), act.fct = "tanh", rep = 1
  )
  converter <- Converter$new(nn)
  # Converter with input dim as vector
  converter <- Converter$new(nn, input_dim = c(2))
  # Converter with input dim as list
  converter <- Converter$new(nn, input_dim = list(2))

  # Test if converting was successful

  # Forward pass
  y_true <- predict(nn, iris)
  y_pred <- as_array(converter$model(
    list(torch_tensor(as.matrix(iris[,c(3,4)]))))[[1]])
  expect_equal(dim(y_true), dim(y_pred))
  expect_lt(mean((y_true - y_pred)^2), 1e-10)

  # update_ref method
  x_ref <- iris[sample(nrow(iris), 1), 3:4]
  y_ref_true <- as.vector(predict(nn, x_ref))
  y_ref <- as.array(converter$model$update_ref(torch_tensor(as.matrix(x_ref)))[[1]])
  dim_y_ref <- dim(y_ref)
  expect_equal(dim_y_ref, c(1, 1))
  expect_lt((y_ref_true - y_ref)^2, 1e-10)

})


################################################################################
#                            Package: torch
################################################################################

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

  # input dim as numeric
  converter <- Converter$new(model, input_dim = c(5))
  # input dim as list
  converter <- Converter$new(model, input_dim = list(5))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(list(input))[[1]])

  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)
})


test_that("Test torch sequential model: Dense with dropout", {
  library(torch)

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
  model$eval()
  input <- torch_randn(10, 5)

  expect_error(Converter$new(model))

  converter <- Converter$new(model, input_dim = c(5))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(list(input))[[1]])

  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)
})

test_that("Test torch sequential model: 1D Conv", {
  library(torch)
  input <- torch_randn(10, 3, 100)

  model <- nn_sequential(
    nn_conv1d(3,10,10),
    nn_relu(),
    nn_conv1d(10,8,8, stride = 2),
    nn_softplus(),
    nn_batch_norm1d(8),
    nn_conv1d(8,6,6, padding = 2),
    nn_softplus(),
    nn_max_pool1d(kernel_size = 1),
    nn_batch_norm1d(6),
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
  model$eval()

  expect_error(Converter$new(model))

  # input dim as vector
  converter <- Converter$new(model, input_dim = c(3, 100))
  # input dim as list
  converter <- Converter$new(model, input_dim = list(c(3, 100)))
  # input dim not channels first
  expect_error(Converter$new(model, input_dim = c(100, 3)))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(list(input))[[1]])

  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)
})

test_that("Test torch sequential model: 1D Conv failures", {
  # unsupported padding mode
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

  # Padding for pooling layers is not supported
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

  input <- torch_randn(10, 3, 30, 30)

  model <- nn_sequential(
    nn_conv2d(3,10,5),
    nn_relu(),
    nn_conv2d(10,8,4, stride = c(2,1)),
    nn_relu(),
    nn_conv2d(8,8,4, stride = 2),
    nn_relu(),
    nn_batch_norm2d(8),
    nn_conv2d(8,6,3, padding = c(5,4)),
    nn_relu(),
    nn_conv2d(6,6,3, padding = 3),
    nn_relu(),
    nn_batch_norm2d(6),
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

  model$eval()

  expect_error(Converter$new(model))

  # input dim as vector
  converter <- Converter$new(model, input_dim = c(3, 30, 30))
  # input dim as list
  converter <- Converter$new(model, input_dim = list(c(3, 30, 30)))
  # input dim not channels first
  expect_error(Converter$new(model, input_dim = c(30, 30, 3)))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(list(input))[[1]])

  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)
})


test_that("Test torch sequential model: 2D Conv with pooling", {
  library(torch)

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

  # forward pass
  converter <- Converter$new(model, input_dim = c(3, 30, 30))
  y_true <- as_array(model(input))
  y <- as_array(converter$model(list(input), TRUE, TRUE, TRUE, TRUE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y - y_true)^2), 1e-12)

  # update x_ref
  x_ref <- array(rnorm(3 * 30 * 30), dim = c(1, 3, 30, 30))
  y_ref <- as_array(converter$model$update_ref(torch_tensor(x_ref))[[1]])
  dim_y_ref <- dim(y_ref)
  y_ref_true <- as.array(model(x_ref))
  dim_y_ref_true <- dim(y_ref_true)

  expect_equal(dim_y_ref_true, dim_y_ref)
  expect_lt(mean((y_ref - y_ref_true)^2), 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$input_dim[[1]], c(3, 30, 30))
  # output dimension
  expect_equal(converter$output_dim[[1]], 2)
})

test_that("Test torch sequential model: 1D Conv failures", {
  # unsupported padding mode
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

  # padding in pooling layer
  model <- nn_sequential(
    nn_conv2d(3,2,5),
    nn_relu(),
    nn_max_pool2d(2, padding = c(1)),
    nn_flatten(),
    nn_linear(32, 2)
  )
  expect_error(Converter$new(model, input_dim = c(3,10,10)))
})

################################################################################
#                            Package: Keras
################################################################################

#
# Sequential Models
#

test_that("Test keras sequential: Dense", {
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

  converter <- Converter$new(model)
  # input dim as vector
  converter <- Converter$new(model, input_dim = c(4))
  # input dim as list
  converter <- Converter$new(model, input_dim = list(4))

  # forward method
  y_true <- as.array(model(data))
  dim_y_true <- dim(y_true)
  y <- as_array(converter$model(list(torch_tensor(data)))[[1]])
  dim_y <- dim(y)

  expect_equal(dim_y, dim_y_true)
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref <- matrix(rnorm(4), nrow = 1, ncol = 4)
  y_ref <- as_array(converter$model$update_ref(list(torch_tensor(x_ref)))[[1]])
  dim_y_ref <- dim(y_ref)
  y_ref_true <- as.array(model(x_ref))
  dim_y_ref_true <- dim(y_ref_true)

  expect_equal(dim_y_ref, dim_y_ref_true)
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

  ## other attributes
  # input dimension
  converter_input_dim <- converter$input_dim[[1]]
  expect_equal(converter_input_dim, 4)
  # output dimension
  converter_output_dim <- converter$output_dim[[1]]
  expect_equal(converter_output_dim, 3)
})


test_that("Test keras sequential: Conv1D with 'valid' padding", {
  library(keras)
  library(torch)

  data <- array(rnorm(10 * 128 * 4), dim = c(10, 128, 4))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(128, 4), kernel_size = 16, filters = 8,
      activation = "softplus"
    ) %>%
    layer_max_pooling_1d() %>%
    layer_conv_1d(kernel_size = 16, filters = 4, activation = "tanh") %>%
    layer_zero_padding_1d(padding = c(1,2)) %>%
    layer_average_pooling_1d() %>%
    layer_conv_1d(kernel_size = 16, filters = 2, activation = "relu") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  converter <- Converter$new(model)
  # input dim as vector
  converter <- Converter$new(model, input_dim = c(4, 128))
  # input dim as list
  converter <- Converter$new(model, input_dim = list(c(4, 128)))
  # not channels first
  expect_error(Converter$new(model, input_dim = list(c(128, 4))))

  # forward method
  y_true <- as.array(model(data))
  dim_y_true <- dim(y_true)
  y <- as_array(converter$model(list(torch_tensor(data)), channels_first = FALSE)[[1]])
  dim_y <- dim(y)

  expect_equal(dim_y, dim_y_true)
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref <- array(rnorm(128 * 4), dim = c(1, 128, 4))
  y_ref <- as_array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                               channels_first = FALSE)[[1]])
  dim_y_ref <- dim(y_ref)
  y_ref_true <- as.array(model(x_ref))
  dim_y_ref_true <- dim(y_ref_true)

  expect_equal(dim_y_ref_true, dim_y_ref)
  expect_lt(mean((y_ref - y_ref_true)^2), 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$input_dim[[1]], c(4, 128))
  # output dimension
  expect_equal(converter$output_dim[[1]], 1)
})

test_that("Test keras sequential: Conv1D with 'same' padding", {
  library(keras)
  library(torch)

  data <- array(rnorm(10 * 128 * 4), dim = c(10, 128, 4))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(128, 4), kernel_size = 16, filters = 8,
      activation = "softplus", padding = "same"
    ) %>%
    layer_batch_normalization() %>%
    layer_conv_1d(
      kernel_size = 16, filters = 4, activation = "tanh",
      padding = "same"
    ) %>%
    layer_batch_normalization() %>%
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
  y <- as.array(converter$model(list(torch_tensor(data)),
                                channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(128 * 4), dim = c(1, 128, 4))
  y_ref <-
    as.array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                        channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$input_dim[[1]], c(4, 128))
  # output dimension
  expect_equal(converter$output_dim[[1]], 1)
})


test_that("Test keras sequential: Conv2D with 'valid' padding", {
  library(keras)
  library(torch)

  data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
      activation = "softplus", padding = "valid"
    ) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d() %>%
    layer_zero_padding_2d(padding = list(c(2,2), c(5,3))) %>%
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh",
      padding = "valid"
    ) %>%
    layer_average_pooling_2d(pool_size = c(1,1)) %>%
    layer_batch_normalization() %>%
    layer_zero_padding_2d(padding = c(3,5)) %>%
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
  y <- as.array(converter$model(list(torch_tensor(data)),
                                channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))
  y_ref <- as.array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                               channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$input_dim[[1]], c(3, 32, 32))
  # output dimension
  expect_equal(converter$output_dim[[1]], 1)
})

test_that("Test keras sequential: Conv2D with 'same' padding", {
  library(keras)
  library(torch)

  data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))

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
  y <- as.array(converter$model(list(torch_tensor(data)),
                                channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))
  y_ref <- as.array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                               channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$input_dim[[1]], c(3, 32, 32))
  # output dimension
  expect_equal(converter$output_dim[[1]], 1)
})

test_that("Test keras sequential: CNN with average pooling", {
  library(torch)
  library(keras)

  data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))

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
  y <- as.array(converter$model(list(torch_tensor(data)),
                                channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32 * 32 * 3), dim = c(1, 32, 32, 3))
  y_ref <- as.array(converter$model$update_ref(list(torch_tensor(x_ref)),
                                               channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt((y_ref_true - y_ref)^2, 1e-12)

  ## other attributes
  # input dimension
  expect_equal(converter$input_dim[[1]], c(3, 32, 32))
  # output dimension
  expect_equal(converter$output_dim[[1]], 1)
})


#
# Other Models
#
test_that("Test keras model: Sequential", {
  library(keras)

  main_input <- layer_input(shape = c(10,10,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2)) %>%
    layer_flatten() %>%
    layer_dense(units = 4)
  main_output <- lstm_out %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 4, activation = 'tanh') %>%
    layer_dense(units = 2, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  model <- keras_model(
    inputs = c(main_input),
    outputs = c(main_output)
  )

  conv <- Converter$new(model)
  data <- lapply(list(c(10,10,2)), function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_torch <- lapply(data, torch_tensor)

  # forward method
  y_true <- as.array(model(data))
  y <- as_array(conv$model(data_torch, channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update
  x_ref <- lapply(list(c(10,10,2)), function(x) array(rnorm(prod(x)), dim = c(1, x)))
  x_ref_torch <- lapply(x_ref, torch_tensor)
  y_ref <- as_array(conv$model$update_ref(x_ref_torch, channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)
})


test_that("Test keras model: Two inputs + one output", {
  library(keras)

  main_input <- layer_input(shape = c(10,10,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2)) %>%
    layer_flatten() %>%
    layer_dense(units = 4)
  auxiliary_input <- layer_input(shape = c(5), name = 'aux_input')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 4, activation = 'tanh') %>%
    layer_dense(units = 2, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  model <- keras_model(
    inputs = c(main_input, auxiliary_input),
    outputs = c(main_output)
  )

  conv <- Converter$new(model)
  # input dim as list
  conv <- Converter$new(model, input_dim = list(c(2,10,10), c(5)))
  # not channels first
  expect_error(Converter$new(model, input_dim = list(c(10,10,2), c(5))))
  data <- lapply(list(c(10,10,2), c(5)), function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_torch <- lapply(data, torch_tensor)

  # forward method
  y_true <- as.array(model(data))
  y <- as_array(conv$model(data_torch, channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update
  x_ref <- lapply(list(c(10,10,2), c(5)), function(x) array(rnorm(prod(x)), dim = c(1, x)))
  x_ref_torch <- lapply(x_ref, torch_tensor)
  y_ref <- as_array(conv$model$update_ref(x_ref_torch, channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)
})

test_that("Test keras model: Two inputs + two output", {
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
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  model <- keras_model(
    inputs = c(auxiliary_input, main_input),
    outputs = c(auxiliary_output, main_output)
  )

  conv <- Converter$new(model)
  data <- lapply(list(c(5), c(10,10,2)), function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_torch <- lapply(data, torch_tensor)

  # forward method
  y_true <- lapply(model(data), as.array)
  y <- lapply(conv$model(data_torch, channels_first = FALSE), as_array)
  expect_equal(lapply(y, dim), lapply(y_true, dim))
  expect_lt(mean(unlist(lapply(seq_along(y),
                               function(i) mean((y_true[[i]] - y[[i]])^2)))),
            1e-12)

  # update
  x_ref <- lapply(list(c(10,10,2), c(5)), function(x) array(rnorm(prod(x)), dim = c(1, x)))
  x_ref_torch <- lapply(x_ref, torch_tensor)
  y_ref <-lapply(conv$model(data_torch, channels_first = FALSE), as_array)
  y_ref_true <- lapply(model(data), as.array)
  expect_equal(lapply(y_ref, dim), lapply(y_ref_true, dim))
  expect_lt(mean(unlist(lapply(seq_along(y_ref),
                               function(i) mean((y_ref_true[[i]] - y_ref[[i]])^2)))),
            1e-12)
})



test_that("Test keras model: Two inputs + two output (second)", {
  library(keras)

  main_input <- layer_input(shape = c(12,15,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2)) %>%
    layer_flatten() %>%
    layer_dense(units = 11)
  auxiliary_input <- layer_input(shape = c(11), name = 'aux_input')
  auxiliary_input_2 <- layer_input(shape = c(16), name = 'aux_input_2')
  seq_test <- auxiliary_input_2 %>%
    layer_dense(units = 11, activation = "relu")
  seq_test_2 <- layer_add(c(seq_test, auxiliary_input)) %>%
    layer_dense(units = 11, activation = "relu")
  auxiliary_output <- layer_concatenate(c(lstm_out, seq_test, seq_test_2)) %>%
    layer_dense(units = 2, activation = 'linear', name = 'aux_output')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  model <- keras_model(
    inputs = c(auxiliary_input, main_input, auxiliary_input_2),
    outputs = c(auxiliary_output, main_output)
  )

  conv <- Converter$new(model)
  data <- lapply(list(c(11), c(12,15,2), c(16)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_torch <- lapply(data, torch_tensor)

  # forward method
  y_true <- lapply(model(data), as.array)
  y <- lapply(conv$model(data_torch, channels_first = FALSE), as_array)
  expect_equal(lapply(y, dim), lapply(y_true, dim))
  expect_lt(mean(unlist(lapply(seq_along(y),
                               function(i) mean((y_true[[i]] - y[[i]])^2)))),
            1e-12)

  # update
  x_ref <- lapply(list(c(11), c(12,15,2), c(16)), function(x) array(rnorm(prod(x)), dim = c(1, x)))
  x_ref_torch <- lapply(x_ref, torch_tensor)
  y_ref <- lapply(conv$model(x_ref_torch, channels_first = FALSE), as_array)
  y_ref_true <- lapply(model(x_ref), as.array)
  expect_equal(lapply(y_ref, dim), lapply(y_ref_true, dim))
  expect_lt(mean(unlist(lapply(seq_along(y_ref),
                               function(i) mean((y_ref_true[[i]] - y_ref[[i]])^2)))),
            1e-12)
})


test_that("Test keras model: Sequential as submodule", {
  library(keras)
  library(innsight)
  library(torch)

  input <- layer_input(shape = c(10))
  seq_model <- keras_model_sequential() %>%
    layer_dense(units = 32) %>%
    layer_activation('relu') %>%
    layer_dense(units = 16) %>%
    layer_activation('relu') %>%
    layer_dense(units = 10) %>%
    layer_activation('relu')

  out <-  seq_model(input) %>%
    layer_dense(32, activation = "relu") %>%
    layer_dense(1, activation = "sigmoid")

  model <- keras_model(inputs = input, outputs = out)
  conv <- Converter$new(model)

  data <- matrix(rnorm(4 * 10), nrow = 4)

  # forward method
  y_true <- as.array(model(data))
  dim_y_true <- dim(y_true)
  y <- as_array(conv$model(list(torch_tensor(data)))[[1]])
  dim_y <- dim(y)

  expect_equal(dim_y, dim_y_true)
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref <- matrix(rnorm(10), nrow = 1, ncol = 10)
  y_ref <- as_array(conv$model$update_ref(list(torch_tensor(x_ref)))[[1]])
  dim_y_ref <- dim(y_ref)
  y_ref_true <- as.array(model(x_ref))
  dim_y_ref_true <- dim(y_ref_true)

  expect_equal(dim_y_ref, dim_y_ref_true)
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

})



#
# Predefined models
#


# VGG16
test_that("Test keras predefiend Model: VGG16", {
  library(keras)

  model <- application_vgg16(weights = NULL, input_shape = c(32,32,3))

  conv <- Converter$new(model)
  data <- array(rnorm(10 * 32* 32* 3), dim = c(10, 32, 32, 3))
  data_torch <- torch_tensor(data)

  # forward method
  y_true <- as.array(model(data))
  y <- as_array(conv$model(data_torch, channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32* 32* 3), dim = c(1, 32, 32, 3))
  x_ref_torch <- torch_tensor(x_ref)
  y_ref <- as_array(conv$model(x_ref_torch, channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

  # Gradient method
  grad <- Gradient$new(conv, x_ref, channels_first = FALSE, times_input = FALSE)
  grad_t_input <- Gradient$new(conv, x_ref, channels_first = FALSE, times_input = TRUE)

  # LRP
  lrp_simple <- LRP$new(conv, x_ref, channels_first = FALSE, output_idx = c(1,2),
                        rule_name = "simple")
  lrp_eps <- LRP$new(conv, x_ref, channels_first = FALSE, output_idx = c(1,2),
                     rule_name = "epsilon")
  lrp_ab <- LRP$new(conv, x_ref, channels_first = FALSE, output_idx = c(1,2),
                    rule_name = "alpha_beta")

  # DeepLift
  deeplift_rescale <- DeepLift$new(conv, data, x_ref = x_ref, channels_first = FALSE,
                                   output_idx = c(1,2), rule_name = "rescale", ignore_last_act = FALSE)
  deeplift_rc <- DeepLift$new(conv, x_ref, channels_first = FALSE,
                              output_idx = c(1,2), rule_name = "reveal_cancel")

  # ConnectionWeights
  connect_weights <- ConnectionWeights$new(conv, channels_first = FALSE)
})

# ResNet50
test_that("Test keras predefiend Model: Resnet50", {
  library(keras)

  model <- application_resnet50(weights = NULL, input_shape = c(32,32,3))

  conv <- Converter$new(model)
  data <- array(rnorm(10 * 32* 32* 3), dim = c(10, 32, 32, 3))
  data_torch <- torch_tensor(data)

  # forward method
  y_true <- as.array(model(data))
  y <- as_array(conv$model(data_torch, channels_first = FALSE)[[1]])
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32* 32* 3), dim = c(1, 32, 32, 3))
  x_ref_torch <- torch_tensor(x_ref)
  y_ref <- as_array(conv$model(x_ref_torch, channels_first = FALSE)[[1]])
  y_ref_true <- as.array(model(x_ref))
  expect_equal(dim(y_ref), dim(y_ref_true))
  expect_lt(mean((y_ref_true - y_ref)^2), 1e-12)

  # Gradient method
  grad <- Gradient$new(conv, x_ref, channels_first = FALSE, times_input = FALSE)
  grad_t_input <- Gradient$new(conv, x_ref, channels_first = FALSE, times_input = TRUE)

  # LRP
  lrp_simple <- LRP$new(conv, x_ref, channels_first = FALSE, output_idx = c(1,2),
                        rule_name = "simple")
  lrp_eps <- LRP$new(conv, x_ref, channels_first = FALSE, output_idx = c(1,2),
                     rule_name = "epsilon")
  lrp_ab <- LRP$new(conv, x_ref, channels_first = FALSE, output_idx = c(1,2),
                        rule_name = "alpha_beta")

  # DeepLift
  deeplift_rescale <- DeepLift$new(conv, data, x_ref = x_ref, channels_first = FALSE,
                                   output_idx = c(1,2), rule_name = "rescale", ignore_last_act = FALSE)
  deeplift_rc <- DeepLift$new(conv, x_ref, channels_first = FALSE,
                              output_idx = c(1,2), rule_name = "reveal_cancel")

  # ConnectionWeights
  connect_weights <- ConnectionWeights$new(conv, channels_first = FALSE)
})

test_that("Test keras model: Two inputs + two output with VGG16 as submodule", {
  library(keras)

  main_input <- layer_input(shape = c(32,32,3))
  vgg16_model <- application_vgg16(include_top = FALSE, weights = NULL,
                                   input_shape = c(32,32,3))
  lstm_out <- main_input %>%
    vgg16_model %>%
    layer_flatten() %>%
    layer_dense(units = 11)
  auxiliary_input <- layer_input(shape = c(11), name = 'aux_input')
  auxiliary_input_2 <- layer_input(shape = c(16), name = 'aux_input_2')
  seq_test <- auxiliary_input_2 %>%
    layer_dense(units = 11, activation = "relu")
  seq_test_2 <- layer_add(c(seq_test, auxiliary_input)) %>%
    layer_dense(units = 11, activation = "relu")
  auxiliary_output <- layer_concatenate(c(lstm_out, seq_test, seq_test_2)) %>%
    layer_dense(units = 2, activation = 'linear', name = 'aux_output')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'tanh') %>%
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  model <- keras_model(
    inputs = c(auxiliary_input, main_input, auxiliary_input_2),
    outputs = c(auxiliary_output, main_output)
  )

  conv <- Converter$new(model)
  data <- lapply(list(c(11), c(32,32,3), c(16)),
                 function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))
  data_torch <- lapply(data, torch_tensor)

  # forward method
  y_true <- lapply(model(data), as.array)
  y <- lapply(conv$model(data_torch, channels_first = FALSE), as_array)
  expect_equal(lapply(y, dim), lapply(y_true, dim))
  expect_lt(mean(unlist(lapply(seq_along(y),
                               function(i) mean((y_true[[i]] - y[[i]])^2)))),
            1e-12)

  # update
  x_ref <- lapply(list(c(11), c(32,32,3), c(16)), function(x) array(rnorm(prod(x)), dim = c(1, x)))
  x_ref_torch <- lapply(x_ref, torch_tensor)
  y_ref <- lapply(conv$model(x_ref_torch, channels_first = FALSE), as_array)
  y_ref_true <- lapply(model(x_ref), as.array)
  expect_equal(lapply(y_ref, dim), lapply(y_ref_true, dim))
  expect_lt(mean(unlist(lapply(seq_along(y_ref),
                               function(i) mean((y_ref_true[[i]] - y_ref[[i]])^2)))),
            1e-12)


})
