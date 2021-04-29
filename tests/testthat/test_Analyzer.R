context("Class Analyzer")

test_that("Test general errors",{
  expect_error(Analyzer$new(NULL))
  expect_error(Analyzer$new(NA))
  expect_error(Analyzer$new(c(3)))
  expect_error(Analyzer$new("124"))
})

test_that("Test neuralnet model", {
  library(neuralnet)
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
  y <- as.vector(analyzer$forward(as.matrix(iris[,3:4]))$out)
  expect_equal(y_true, y, ignore_attr = TRUE)

  # update method
  idx <- sample(nrow(iris), 10)
  y_true <- as.vector(predict(nn, iris))
  analyzer$update(as.matrix(iris[,3:4]))
  expect_equal(y_true, rev(analyzer$layers)[[1]]$outputs, ignore_attr = TRUE)


  #
  # ----------------------------- negative tests -------------------------------
  #

  # custom activation function
  softplus <- function(x) log(1+exp(x))
  nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
                  iris, linear.output = FALSE,
                  hidden = c(3,2), act.fct = softplus, rep = 1)
  expect_error(Analyzer$new(nn))

  # doesn't converge
  expect_warning(nn <- neuralnet(Species ~ .,
                  iris, linear.output = TRUE,
                  hidden = c(3,2), act.fct = "tanh", rep = 1, stepmax = 1e+01))
  expect_error(Analyzer$new(nn))

})

test_that("Test keras model", {
  #
  # ----------------------- positive tests ------------------------------------
  #
  library(keras)
  data(iris)

  iris[,5] <- as.numeric(iris[,5]) -1
  # Turn `iris` into a matrix
  iris <- as.matrix(iris)
  # Set iris `dimnames` to `NULL`
  dimnames(iris) <- NULL
  # Determine sample size
  ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))
  # Split the `iris` data
  iris.training <- iris[ind==1, 1:4]
  iris.test <- iris[ind==2, 1:4]
  # Split the class attribute
  iris.trainingtarget <- iris[ind==1, 5]
  iris.testtarget <- iris[ind==2, 5]
  # One hot encode training target values
  iris.trainLabels <- to_categorical(iris.trainingtarget)
  # One hot encode test target values
  iris.testLabels <- to_categorical(iris.testtarget)

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
    layer_dropout(0.1) %>%
    layer_dense(units = 8, activation = 'relu') %>%
    layer_dropout(0.1) %>%
    layer_dense(units = 3, activation = 'softmax')

  # test non-fitted model
  analyzer = Analyzer$new(model)

  # test compiled model
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
  )
  analyzer = Analyzer$new(model)

  # test fitted model
  history <- model %>% fit(
    iris.training,
    iris.trainLabels,
    epochs = 50,
    batch_size = 5,
    validation_split = 0.2, verbose = 0
  )
  analyzer = Analyzer$new(model)

  # forward method
  y_true <- predict(model, iris.test)
  y <- analyzer$forward(iris.test)$out
  expect_equal(y_true, y, tolerance = 1e-6, ignore_attr = TRUE)

  # update method
  analyzer$update(iris.test)
  expect_equal(y_true, rev(analyzer$layers)[[1]]$outputs, tolerance = 1e-6, ignore_attr = TRUE)

  #
  # ------------------------- negative method ---------------------------------
  #

  # activation function in own layer
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'linear', input_shape = c(4)) %>%
    layer_activation_relu() %>%
    layer_dense(units = 3, activation = 'softmax')

  expect_error(Analyzer$new(model))

  # not implemented layer
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 16, activation = 'linear', input_shape = c(4)) %>%
    layer_batch_normalization() %>%
    layer_dense(units = 3, activation = 'softmax')

  expect_error(Analyzer$new(model))
})
