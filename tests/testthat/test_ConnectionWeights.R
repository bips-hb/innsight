context("Connection Weights")

test_that("Test Connection Weights method",{
  library(neuralnet)
  nn <- neuralnet(Species ~ .,
                  iris, linear.output = FALSE,
                  hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
  analyzer = Analyzer$new(nn)

  expect_error(Connection_Weights(NULL, out_class = 1))
  expect_error(Connection_Weights(NULL, out_class = 10))
  expect_error(Connection_Weights(analyzer, out_class = 10))
  expect_error(Connection_Weights(analyzer, out_class = 1.4))
  expect_error(Connection_Weights(analyzer, out_class = -2))

  result <- Connection_Weights(analyzer, out_class = NULL)
  expect_true("ConnectionWeights" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))

  result <- Connection_Weights(analyzer, out_class = 1)
  expect_true("ConnectionWeights" %in% class(result))
  expect_true("ggplot" %in% class(plot(result)))
})
