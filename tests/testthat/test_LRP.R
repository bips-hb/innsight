context("Layerwise Relevance Propagation")

test_that("Test LRP method",{
  library(neuralnet)
  nn <- neuralnet(Species ~ .,
                  iris, linear.output = FALSE,
                  hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
  analyzer = Analyzer$new(nn)

  expect_error(LRP(NULL, iris[1:10, -5]))
  expect_error(LRP(analyzer))
  expect_error(LRP(analyzer, iris))
  expect_error(LRP(analyzer, iris[1:10,-c(4,5)]))
  expect_error(LRP(analyzer, iris[1:10,-5], rule_name = "asdf"))

  # test with data.frame
  for (rule in c("simple", "eps", "ab", "ww")) {
    result <- LRP(analyzer, iris[1:10,-5], rule_name = rule)
    expect_true(is.array(result))
    expect_equal(dim(result), c(4,3,10))
    expect_true("LRP" %in% class(result))
    expect_true("ggplot" %in% class(plot(result)))
    expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
    expect_true("ggplot" %in% class(plot(result, scale = TRUE)))
  }

  # test with matrix
  d <- t(as.matrix(t(iris[1:10,-5])))
  dimnames(d) <- c()
  for (rule in c("simple", "eps", "ab", "ww")) {
    result <- LRP(analyzer, d, rule_name = rule)
    expect_true(is.array(result))
    expect_equal(dim(result), c(4,3,10))
    expect_true("LRP" %in% class(result))
    expect_true("ggplot" %in% class(plot(result)))
    expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
    expect_true("ggplot" %in% class(plot(result, scale = TRUE)))
  }
})
