context("Test Analyzer")

test_that("Analyzer Initialization", {
  ###--------------neuralnet-----------------
  library(neuralnet)
  library(datasets)
  data(iris)

  # Binary classification
  nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, linear.output = FALSE, rep = 1) # one rep
  expect_error(Analyzer$new(nn), NA)

  nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, linear.output = FALSE, rep = 5) # multiple rep
  expect_error(Analyzer$new(nn), NA)

  nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, linear.output = FALSE, hidden = c(5,3), rep = 5) # more hidden layers
  expect_error(Analyzer$new(nn), NA)

  ###--------------keras model-----------------

  #
  # toDo
  #

})

test_that("Forward method", {
  ###--------------neuralnet-----------------
  for (act in c("logistic", "tanh")) {
    nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, act.fct = act, linear.output = FALSE, hidden = c(5,3), rep = 1) # more hidden layers

    analyzer <- Analyzer$new(nn)
    iris_test <- iris[1:20, ]

    nn_pred <- predict(nn, iris_test)
    for (i in 1:20) {
      analyzer_pred <- analyzer$forward(as.vector(t(iris[i,c(3,4)])))$out
      expect_equal(analyzer_pred, nn_pred[i])
    }
  }
  ###--------------keras model-----------------

  #
  # toDo
  #
})

test_that("Connection Weights", {
  ##------------------- neuralnet ---------------------
  model_neuralnet <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length + Petal.Width, iris, linear.output = FALSE, hidden = c(5,4), rep = 2)

  # create an analyzer for this model
  analyzer = Analyzer$new(model_neuralnet)

  expect_error(analyzer$Connection_Weights(), NA)
  expect_error(analyzer$Connection_Weights(out_class = 2), NA)

  expect_equal(is.matrix(analyzer$Connection_Weights()), TRUE)
  expect_equal(is.vector(analyzer$Connection_Weights(out_class = 2)), TRUE)

  ###--------------keras model-----------------

  #
  # toDo
  #
})

test_that("Layerwise Relevance Propagation", {
  ##------------------- neuralnet ---------------------
  model_neuralnet <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length + Petal.Width, iris, linear.output = FALSE, hidden = c(5,4), rep = 2)

  # create an analyzer for this model
  analyzer = Analyzer$new(model_neuralnet)

  input <- as.vector(t(iris[1,-5]))

  expect_error(analyzer$LRP(input), NA)
  expect_error(analyzer$LRP(input, out_class = 2), NA)

  expect_equal(is.matrix(analyzer$LRP(input)), TRUE)
  expect_equal(is.vector(analyzer$LRP(input, out_class = 2)), TRUE)

  ###--------------keras model-----------------

  #
  # toDo
  #
})



test_that("DeepLIFT", {
  ##------------------- neuralnet ---------------------
  model_neuralnet <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length + Petal.Width, iris, linear.output = FALSE, hidden = c(5,4), rep = 2)

  # create an analyzer for this model
  analyzer = Analyzer$new(model_neuralnet)

  input <- as.vector(t(iris[1,-5]))
  input_ref <- rnorm(4)

  expect_error(analyzer$DeepLift(input, input_ref), NA)
  expect_error(analyzer$DeepLift(input, input_ref, out_class = 2), NA)

  expect_equal(is.matrix(analyzer$DeepLift(input, input_ref)), TRUE)
  expect_equal(is.vector(analyzer$DeepLift(input, input_ref, out_class = 2)), TRUE)

  ###--------------keras model-----------------

  #
  # toDo
  #
})

