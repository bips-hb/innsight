context("Connection Weights")

test_that("Test Connection Weights method",{
  library(neuralnet)
  nn <- neuralnet(Species ~ .,
                  iris, linear.output = FALSE,
                  hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
  analyzer = Analyzer$new(nn)

  expect_error(Connection_Weights(NULL))

  result <- Connection_Weights(analyzer)
  expect_true(is.matrix(result))
  expect_equal(dim(result), c(4,3))
  expect_true("ConnectionWeights" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))
})
