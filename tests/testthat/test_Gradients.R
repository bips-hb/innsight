context("Gradients")

test_that("Test Gradient method",{
  library(neuralnet)
  nn <- neuralnet(Species ~ .,
                  iris, linear.output = FALSE,
                  hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
  analyzer = Analyzer$new(nn)

  expect_error(Gradient(NULL, NULL))
  expect_error(Gradient(analyzer, NULL))
  expect_error(Gradient(analyzer, iris[1:10,]))
  expect_error(Gradient(analyzer, iris[1:10,-5], times_input = NULL))
  expect_error(Gradient(analyzer, iris[1:10,-5], ignore_last_act = NULL))

  expect_error(Gradient(analyzer, iris[1:10,-5]), NA)
  expect_error(Gradient(analyzer, iris[1:10,-5], times_input = FALSE), NA)
  expect_error(Gradient(analyzer, data = iris[1:10,-5], times_input = FALSE), NA)

  # test with data.frame
  result <- Gradient(analyzer, iris[1:10,-5])
  expect_true("Gradient" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))
  expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
  expect_true("ggplot" %in% class(plot(result, scale = TRUE)))

  result <- Gradient(analyzer, iris[1:10,-5], times_input = FALSE)
  expect_true("Gradient" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))
  expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
  expect_true("ggplot" %in% class(plot(result, scale = TRUE)))

  # test with matrix
  d <- t(as.matrix(t(iris[1:10,-5])))
  dimnames(d) <- c()
  result <- Gradient(analyzer, d)
  expect_true("Gradient" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))
  expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
  expect_true("ggplot" %in% class(plot(result, scale = TRUE)))
})


test_that("Test SmoothGrad method",{
  library(neuralnet)
  nn <- neuralnet(Species ~ .,
                  iris, linear.output = FALSE,
                  hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
  analyzer = Analyzer$new(nn)

  expect_error(SmoothGrad(NULL, NULL))
  expect_error(SmoothGrad(analyzer, NULL))
  expect_error(SmoothGrad(analyzer, iris[1:10,]))
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], times_input = NULL))
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], ignore_last_act = NULL))
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], n = NULL))
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], n = -1))
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], n = 3.5))
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], noise_level = NULL))
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], noise_level = -10))

  expect_error(SmoothGrad(analyzer, iris[1:10,-5]), NA)
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], times_input = FALSE), NA)
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], n = 10), NA)
  expect_error(SmoothGrad(analyzer, iris[1:10,-5], noise_level = 2), NA)

  # test with data.frame
  result <- SmoothGrad(analyzer, iris[1:10,-5])
  expect_true("Gradient" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))
  expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
  expect_true("ggplot" %in% class(plot(result, scale = TRUE)))

  result <- SmoothGrad(analyzer, iris[1:10,-5], times_input = FALSE)
  expect_true("Gradient" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))
  expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
  expect_true("ggplot" %in% class(plot(result, scale = TRUE)))

  # test with matrix
  d <- t(as.matrix(t(iris[1:10,-5])))
  dimnames(d) <- c()
  result <- SmoothGrad(analyzer, d)
  expect_true("Gradient" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))
  expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
  expect_true("ggplot" %in% class(plot(result, scale = TRUE)))
})

