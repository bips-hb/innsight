
test_that("Test general errors", {

  expect_error(Converter$new(NULL))
  expect_error(Converter$new(NA))
  expect_error(Converter$new(c(3)))
  expect_error(Converter$new("124"))
})


test_that("Test neuralnet model", {
  skip_on_os("windows")

  data(iris)
  #
  # --------------------- positive tests --------------------------------------
  #

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
  expect_warning(nn_not_converged <- neuralnet(Species ~ .,
                                               iris,
                                               linear.output = TRUE,
                                               hidden = c(3, 2), act.fct = "tanh", rep = 1, stepmax = 1e+01
  ))
  expect_error(Converter$new(nn_not_converged))
})

test_that("Test list model: Dense", {
  skip_on_os("windows")

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
  model <- converter$model

  # test output dimension
  input <- torch::torch_randn(10,5)
  out <- model(input)
  expect_equal(dim(out), c(10, 2))

})

test_that("Test list model: 2D Convolution", {
  skip_on_os("windows")

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
      activation_name = "tanh",
      dim_in = c(8L, 9L, 9L),
      dim_out = c(2L, 8L, 8L)
    )
  model$layers$Layer_3 <-
    list(
      type = "Flatten",
      dim_in = c(2L,8L,8L),
      dim_out = 128L
    )
  model$layers$Layer_4 <-
    list(
      type = "Dense",
      weight = matrix(rnorm(128 * 2), 2, 128),
      bias = rnorm(2),
      activation_name = "softmax",
      dim_in = 128L,
      dim_out = 2L
    )

  # Convert the model
  converter <- Converter$new(model)
  expect_true("Converter" %in% class(converter))

  # get the model
  model <- converter$model

  # test output dimension
  input <- torch::torch_randn(10,3,10,10)
  out <- model(input)
  expect_equal(dim(out), c(10, 2))

})



test_that("Test keras model: Dense", {
  skip_on_os("windows")
  skip_on_cran()
  #
  # --------------------- Dense Model -----------------------------------------
  #

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

  converter$model(torch_tensor(data))

  for (module in converter$model$modules_list) {
    expect_equal(module$input_dim, dim(module$input)[-1])
    expect_equal(module$output_dim, dim(module$output)[-1])
  }
})



test_that("Test keras model: Conv1D with 'valid' padding", {
  skip_on_os("windows")
  skip_on_cran()
  #
  # --------------------- CNN (1D) Model ("valid" padding) --------------------
  #

  data <- array(rnorm(64 * 128 * 4), dim = c(64, 128, 4))

  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(
      input_shape = c(128, 4), kernel_size = 16, filters = 8,
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

  # forward method
  y_true <- as.array(model(data))
  dim_y_true <- dim(y_true)
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE))
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
  skip_on_os("windows")
  skip_on_cran()
  #
  # --------------------- CNN (1D) Model ("same" padding) ---------------------
  #

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
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE))
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
  skip_on_os("windows")
  skip_on_cran()
  #
  # --------------------- CNN (2D) Model ("valid" padding) --------------------
  #

  data <- array(rnorm(64 * 32 * 32 * 3), dim = c(64, 32, 32, 3))

  model <- keras_model_sequential()
  model %>%
    layer_conv_2d(
      input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
      activation = "softplus", padding = "valid"
    ) %>%
    layer_conv_2d(
      kernel_size = 8, filters = 4, activation = "tanh",
      padding = "valid"
    ) %>%
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
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE))
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
  skip_on_os("windows")
  skip_on_cran()
  #
  # --------------------- CNN (2D) Model ("same" padding) --------------------
  #

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
  y <- as.array(converter$model(torch_tensor(data), channels_first = FALSE))
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


