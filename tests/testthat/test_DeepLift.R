context("DeepLift")

test_that("Test DeepLift method",{
  library(neuralnet)
  nn <- neuralnet(Species ~ .,
                  iris, linear.output = FALSE,
                  hidden = c(10,6), act.fct = "tanh", rep = 1, threshold = 0.1 )
  analyzer = Analyzer$new(nn)

  expect_error(DeepLift(NULL, iris[1:10, -5]))
  expect_error(DeepLift(analyzer))
  expect_error(DeepLift(analyzer, iris))
  expect_error(DeepLift(analyzer, iris[1:10,-c(4,5)]))
  expect_error(DeepLift(analyzer, iris[1:10,-5], rule_name = "asdf"))

  expect_error(DeepLift(analyzer, iris[1:10,-5], x_ref = NULL), NA)
  expect_error(DeepLift(analyzer, iris[1:10,-5], x_ref = 1), NA)
  expect_error(DeepLift(analyzer, iris[1:10,-5], x_ref = -1))
  expect_error(DeepLift(analyzer, iris[1:10,-5], x_ref = iris[140,-5]), NA)
  expect_error(DeepLift(analyzer, iris[1:10,-5], x_ref = as.vector(t(iris[140,-5]))), NA)

  # test with data.frame
  for (rule in c("rescale", "revealcancel")) {
    result <- DeepLift(analyzer, iris[1:10,-5], x_ref = NULL, rule_name = rule)
    expect_true(is.array(result))
    expect_equal(dim(result), c(4,3,10))
    expect_true("DeepLift" %in% class(result))
    expect_true("ggplot" %in% class(plot(result)))
    expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
    expect_true("ggplot" %in% class(plot(result, scale = TRUE)))
  }

  # test with matrix
  d <- t(as.matrix(t(iris[1:10,-5])))
  dimnames(d) <- c()
  for (rule in c("rescale", "revealcancel")) {
    result <- DeepLift(analyzer, d, x_ref = NULL, rule_name = rule)
    expect_true(is.array(result))
    expect_equal(dim(result), c(4,3,10))
    expect_true("DeepLift" %in% class(result))
    expect_true("ggplot" %in% class(plot(result)))
    expect_true("ggplot" %in% class(plot(result, rank = TRUE)))
    expect_true("ggplot" %in% class(plot(result, scale = TRUE)))
  }
})
