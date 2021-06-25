
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
  y_true <- as.vector(predict(nn, iris))
  y <- analyzer$forward(as.matrix(iris[,3:4]))

  expect_equal(dim(y), c(nrow(iris),1))
  expect_lt(mean((y_true - y)^2),  1e-12)

  # update_ref method
  x_ref <- iris[sample(nrow(iris), 1), 3:4]
  y_true <- as.vector(predict(nn, x_ref))
  y <- analyzer$update_ref(as.matrix(x_ref))
  expect_equal(dim(y), c(1,1))
  expect_lt((y_true - y)^2, 1e-12)

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
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update_ref
  x_ref <- matrix(rnorm(4), nrow=1, ncol=4)
  y <- analyzer$update_ref(x_ref)
  y_true <- as.array(model(x_ref))
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

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
  # --------------------- CNN (1D) Model ("valid" padding) ---------------------
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
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(128*4), dim=c(1,128,4))
  y <- analyzer$update_ref(x_ref, channels_first = FALSE)
  y_true <- as.array(model(x_ref))
  expect_equal(dim(y_true), dim(y))
  expect_lt(mean((y_true - y)^2), 1e-12)

  ## other attributes
  # input dimension
  expect_equal(analyzer$input_dim, c(4,128))
  # output dimension
  expect_equal(analyzer$output_dim, 1)

  for (module in analyzer$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }



  #
  # --------------------- CNN (1D) Model ("same" padding) ---------------------
  #

  data <- array(rnorm(64*128*4), dim = c(64,128,4))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(input_shape = c(128,4), kernel_size = 16, filters = 8, activation = "softplus", padding = "same") %>%
    layer_conv_1d(kernel_size = 16, filters = 4,  activation = "tanh", padding = "same") %>%
    layer_conv_1d(kernel_size = 16, filters = 2,  activation = "relu", padding = "same") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  analyzer = Analyzer$new(model)

  # forward method
  y_true <- predict(model, data)
  y <- analyzer$forward(data, channels_first = FALSE)
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(128*4), dim=c(1,128,4))
  y <- analyzer$update_ref(x_ref, channels_first = FALSE)
  y_true <- as.array(model(x_ref))
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  ## other attributes
  # input dimension
  expect_equal(analyzer$input_dim, c(4,128))
  # output dimension
  expect_equal(analyzer$output_dim, 1)

  for (module in analyzer$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }



  #
  # --------------------- CNN (2D) Model ("valid" padding) ---------------------
  #

  data <- array(rnorm(64*32*32*3), dim = c(64,32,32,3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(input_shape = c(32,32,3), kernel_size = 8, filters = 8, activation = "softplus", padding = "valid") %>%
    layer_conv_2d(kernel_size = 8, filters = 4,  activation = "tanh", padding = "valid") %>%
    layer_conv_2d(kernel_size = 4, filters = 2,  activation = "relu", padding = "valid") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  analyzer = Analyzer$new(model)

  # forward method
  y_true <- predict(model, data)
  y <- analyzer$forward(data, channels_first = FALSE)
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean((y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32*32*3), dim=c(1,32,32,3))
  y <- analyzer$update_ref(x_ref, channels_first = FALSE)
  y_true <- as.array(model(x_ref))
  expect_equal(dim(y), dim(y_true))
  expect_lt((y_true - y)^2, 1e-12)

  ## other attributes
  # input dimension
  expect_equal(analyzer$input_dim, c(3,32,32))
  # output dimension
  expect_equal(analyzer$output_dim, 1)

  for (module in analyzer$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }


  #
  # --------------------- CNN (2D) Model ("same" padding) ---------------------
  #

  data <- array(rnorm(64*32*32*3), dim = c(64,32,32,3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(input_shape = c(32,32,3), kernel_size = 8, filters = 8, activation = "softplus", padding = "same") %>%
    layer_conv_2d(kernel_size = 8, filters = 4,  activation = "tanh", padding = "same") %>%
    layer_conv_2d(kernel_size = 4, filters = 2,  activation = "relu", padding = "same") %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

  # test non-fitted model
  analyzer = Analyzer$new(model)

  # forward method
  y_true <- predict(model, data)
  y <- analyzer$forward(data, channels_first = FALSE)
  expect_equal(dim(y), dim(y_true))
  expect_lt(mean(abs(y_true - y)^2), 1e-12)

  # update
  x_ref <- array(rnorm(32*32*3), dim=c(1,32,32,3))
  y <- analyzer$update_ref(x_ref, channels_first = FALSE)
  y_true <- as.array(model(x_ref))
  expect_equal(dim(y), dim(y_true))
  expect_lt((y_true - y)^2, 1e-12)

  ## other attributes
  # input dimension
  expect_equal(analyzer$input_dim, c(3,32,32))
  # output dimension
  expect_equal(analyzer$output_dim, 1)

  for (module in analyzer$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }
})


test_that("Test keras net with Conv1D", {
  library(torch)
  library(keras)

  batch_size <- 20
  in_channels <- 3

  for (in_length in c(200,201)) {
    for (kernel_length in c(6,7)) {
      for (padding in c("valid", "same")) {
        for (ds in c(1,2,3,4,5)) {

          # strides
          model <- keras_model_sequential()
          model %>%
            layer_conv_1d(input_shape = c(in_length,in_channels), kernel_size = kernel_length, filters = 8, strides = ds, activation = "softplus", padding = padding) %>%
            layer_conv_1d(kernel_size = kernel_length, filters = 4,  activation = "softplus", padding = padding, strides = ds) %>%
            layer_conv_1d(kernel_size = kernel_length, filters = 2,  activation = "softplus", padding = padding, strides = ds) %>%
            layer_flatten() %>%
            layer_dense(units = 64, activation = "softplus") %>%
            layer_dense(units = 16, activation = "softplus") %>%
            layer_dense(units = 1, activation = "linear")

          analyzer <- Analyzer$new(model)

          # forward
          input <- as_array(torch_randn(batch_size, in_length, in_channels))
          y <- analyzer$forward(input, channels_first = FALSE)
          y_true <- as.array(model(input))

          expect_lt(mean((y - y_true)^2), 1e-10)

          # update_ref
          input <- as_array(torch_randn(1, in_length, in_channels))
          y <- analyzer$update_ref(input, channels_first = FALSE)
          y_true <- as.array(model(input))

          expect_lt(((y - y_true)^2), 1e-10)

          # layer dimension
          for (layer in analyzer$model$modules_list) {
            expect_equal(dim(layer$input)[-1], layer$input_dim)
            expect_equal(dim(layer$output)[-1], layer$output_dim)
          }


          # dilation
          model <- keras_model_sequential()
          model %>%
            layer_conv_1d(input_shape = c(in_length,in_channels), kernel_size = kernel_length, filters = 8, dilation_rate = ds, activation = "softplus", padding = padding) %>%
            layer_conv_1d(kernel_size = kernel_length, filters = 4,  activation = "softplus", padding = padding, dilation_rate = ds) %>%
            layer_conv_1d(kernel_size = kernel_length, filters = 2,  activation = "softplus", padding = padding, dilation_rate = ds) %>%
            layer_flatten() %>%
            layer_dense(units = 64, activation = "softplus") %>%
            layer_dense(units = 16, activation = "softplus") %>%
            layer_dense(units = 1, activation = "linear")

          analyzer <- Analyzer$new(model)

          # forward
          input <- as_array(torch_randn(batch_size, in_length, in_channels))
          y <- analyzer$forward(input, channels_first = FALSE)
          y_true <- as.array(model(input))

          expect_lt(mean((y - y_true)^2), 1e-10)

          # update_ref
          input <- as_array(torch_randn(1, in_length, in_channels))
          y <- analyzer$update_ref(input, channels_first = FALSE)
          y_true <- as.array(model(input))

          expect_lt(mean((y - y_true)^2), 1e-10)

          # layer dimensions
          for (layer in analyzer$model$modules_list) {
            expect_equal(dim(layer$input)[-1], layer$input_dim)
            expect_equal(dim(layer$output)[-1], layer$output_dim)
          }
        }
      }
    }
  }
})



test_that("Test keras net with Conv2D", {
  library(torch)
  library(keras)
  batch_size <- 20
  in_channels <- 3
  i <- 1

  for (in_height in c(200,201)) {
    for (in_width in c(200,201, 220,221)) {
      for (kernel_height in c(3,4)) {
        for (kernel_width in c(3,4,6,7)) {
          for (padding in c("same", "valid")) {
            for (ds_h in c(1,2,3)) {
              for (ds_w in c(1,3,4)) {

                # strides
                model <- keras_model_sequential()
                model %>%
                  layer_conv_2d(input_shape = c(in_height, in_width, in_channels), kernel_size = c(kernel_height, kernel_width), filters = 8, strides = c(ds_h, ds_w), activation = "softplus", padding = padding) %>%
                  layer_conv_2d(kernel_size = c(kernel_height, kernel_width), filters = 4,  activation = "softplus", padding = padding, strides = c(ds_h, ds_w)) %>%
                  layer_conv_2d(kernel_size = c(kernel_height, kernel_width), filters = 2,  activation = "softplus", padding = padding, strides = c(ds_h, ds_w)) %>%
                  layer_flatten() %>%
                  layer_dense(units = 64, activation = "softplus") %>%
                  layer_dense(units = 16, activation = "softplus") %>%
                  layer_dense(units = 1, activation = "linear")

                analyzer <- Analyzer$new(model)

                # forward
                input <- as_array(torch_randn(batch_size, in_height, in_width, in_channels))
                y <- analyzer$forward(input, channels_first = FALSE)
                y_true <- as.array(model(input))

                expect_lt(mean((y - y_true)^2), 1e-8)

                # update_ref
                input <- as_array(torch_randn(1, in_height, in_width, in_channels))
                y <- analyzer$update_ref(input, channels_first = FALSE)
                y_true <- as.array(model(input))

                expect_lt(mean((y - y_true)^2), 1e-8)

                # layer dimension
                for (layer in analyzer$model$modules_list) {
                  expect_equal(dim(layer$input)[-1], layer$input_dim)
                  expect_equal(dim(layer$output)[-1], layer$output_dim)
                }


                # dilation
                model <- keras_model_sequential()
                model %>%
                  layer_conv_2d(input_shape = c(in_height, in_width, in_channels), kernel_size = c(kernel_height, kernel_width), filters = 8, dilation_rate = c(ds_h, ds_w), activation = "softplus", padding = padding) %>%
                  layer_conv_2d(kernel_size = c(kernel_height, kernel_width), filters = 4,  activation = "softplus", padding = padding, dilation_rate = c(ds_h, ds_w)) %>%
                  layer_conv_2d(kernel_size = c(kernel_height, kernel_width), filters = 2,  activation = "softplus", padding = padding, dilation_rate = c(ds_h, ds_w)) %>%
                  layer_flatten() %>%
                  layer_dense(units = 64, activation = "softplus") %>%
                  layer_dense(units = 16, activation = "softplus") %>%
                  layer_dense(units = 1, activation = "linear")

                analyzer <- Analyzer$new(model)

                # forward
                input <- as_array(torch_randn(batch_size, in_height, in_width, in_channels))
                y <- analyzer$forward(input, channels_first = FALSE)
                y_true <- as.array(model(input))

                expect_lt(mean((y - y_true)^2), 1e-8)

                # update_ref
                input <- as_array(torch_randn(1, in_height, in_width, in_channels))
                y <- analyzer$update_ref(input, channels_first = FALSE)
                y_true <- as.array(model(input))

                expect_lt(mean((y - y_true)^2), 1e-8)

                # layer dimension
                for (layer in analyzer$model$modules_list) {
                  expect_equal(dim(layer$input)[-1], layer$input_dim)
                  expect_equal(dim(layer$output)[-1], layer$output_dim)
                }
              }
            }
          }
        }
      }
    }
  }
})
