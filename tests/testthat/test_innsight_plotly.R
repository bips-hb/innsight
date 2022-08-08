library(torch)
library(ggplot2)


# Create data
data_tab <- torch_randn(10,12)
data_1d <- torch_randn(10,3,12)
data_2d <- torch_randn(10,3,12,12)
data_mixed <- lapply(list(c(12), c(12,15,2)),
                     function(x) array(rnorm(10 * prod(x)), dim = c(10, x)))

# Create small net
net_tab <- nn_sequential(
  nn_linear(12, 5),
  nn_relu(),
  nn_linear(5,3)
)
net_1d <- nn_sequential(
  nn_conv1d(3,3, 4),
  nn_tanh(),
  nn_conv1d(3,1,3),
  nn_flatten(),
  nn_linear(7, 3),
  nn_softmax(dim = 1)
)
net_2d <- nn_sequential(
  nn_conv2d(3,3, 4),
  nn_tanh(),
  nn_conv2d(3,1,5),
  nn_flatten(),
  nn_linear(25,3),
  nn_softmax(dim = 1)
)

if (requireNamespace("keras", quietly = FALSE)) {
  main_input <- layer_input(shape = c(12,15,2), name = 'main_input')
  lstm_out <- main_input %>%
    layer_conv_2d(2, c(2,2)) %>%
    layer_flatten() %>%
    layer_dense(units = 12)
  auxiliary_input <- layer_input(shape = c(12), name = 'aux_input')
  auxiliary_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 2, activation = 'softmax', name = 'aux_output')
  main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
    layer_dense(units = 5, activation = 'relu') %>%
    layer_dense(units = 3, activation = 'softmax', name = 'main_output')
  net_mixed <- keras_model(
    inputs = c(auxiliary_input, main_input),
    outputs = c(main_output, auxiliary_output)
  )
}

# Create results
conv_tab <- Converter$new(net_tab, input_dim = 12)
conv_1d <- Converter$new(net_1d, input_dim = c(3,12))
conv_2d <- Converter$new(net_2d, input_dim = c(3,12,12))
conv_mixed <- Converter$new(net_mixed)
res_tab <- Gradient$new(conv_tab, data_tab)
res_1d <- Gradient$new(conv_1d, data_1d)
res_2d <- Gradient$new(conv_2d, data_2d)
res_mixed <- Gradient$new(conv_mixed, data_mixed, channels_first = FALSE,
                          output_idx = list(c(1,2,3), c(1)))

###############################################################################
#                         Individual plots
###############################################################################

#----- Tabular data -----------------------------------------------------------
test_that("innsight_plotly: Tabular data (one row and one column)", {
  p <- plot(res_tab, as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,1], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,1]], "plotly"))
})


test_that("innsight_plotly: Tabular data (multiple rows and columns)", {
  p <- plot(res_tab, output_idx = c(1,2,3), data_idx = c(1,3,5),
            as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[c(2,3),c(1,3)], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,2]], "plotly"))
})

#----- Signal data -----------------------------------------------------------
test_that("innsight_plotly: Signal data (one row and one column)", {
  p <- plot(res_1d, as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,1], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,1]], "plotly"))
})


test_that("innsight_plotly: Tabular data (multiple rows and columns)", {
  p <- plot(res_1d, output_idx = c(1,2,3), data_idx = c(1,3,5),
            as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[c(1,3),c(1,2)], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,2]], "plotly"))
})


#----- Image data -----------------------------------------------------------
test_that("innsight_plotly: Signal data (one row and one column)", {
  p <- plot(res_2d, as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,1], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,1]], "plotly"))
})


test_that("innsight_plotly: Tabular data (multiple rows and columns)", {
  p <- plot(res_2d, output_idx = c(1,2,3), data_idx = c(1,3,5),
            as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[c(1,3),c(1,2)], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,2]], "plotly"))
})

#----- Mixed data -----------------------------------------------------------
test_that("innsight_plotly: Mixed data", {
  skip_if_not_installed("keras")

  p <- plot(res_mixed, output_idx = list(c(1,2,3), c(1)), data_idx = c(1,3,5),
            as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[c(1,3),c(3,4,6)], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,1]], "plotly"))
})


###############################################################################
#                               Boxplot plots
###############################################################################

#----- Tabular data -----------------------------------------------------------
test_that("innsight_plotly: Tabular data (one column)", {
  p <- boxplot(res_tab, as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,1], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,1]], "plotly"))
})


test_that("innsight_plotly: Tabular data (multiple columns)", {
  p <- boxplot(res_tab, output_idx = c(1,2,3), as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,c(1,3)], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,2]], "plotly"))
})

#----- Signal data -----------------------------------------------------------
test_that("innsight_plotly: Signal data (one column)", {
  p <- boxplot(res_1d, as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,1], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,1]], "plotly"))
})


test_that("innsight_plotly: Tabular data (multiple columns)", {
  p <- boxplot(res_1d, output_idx = c(1,2,3), data_idx = c(1,3,5),
               as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,c(1,2)], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,2]], "plotly"))
})


#----- Image data -----------------------------------------------------------
test_that("innsight_plotly: Signal data (one column)", {
  p <- boxplot(res_2d, as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,1], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,1]], "plotly"))
})


test_that("innsight_plotly: Tabular data (multiple columns)", {
  p <- boxplot(res_2d, output_idx = c(1,2,3), as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,c(1,2)], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,2]], "plotly"))
})

#----- Mixed data -----------------------------------------------------------
test_that("innsight_plotly: Mixed data", {
  skip_if_not_installed("keras")

  p <- boxplot(res_mixed, output_idx = list(c(1,2,3), c(1)), as_plotly = TRUE)

  # Check class
  expect_s4_class(p, "innsight_plotly")

  # Check plot, show and print
  expect_true(inherits(print(p), "plotly"))
  expect_true(inherits(show(p), "plotly"))
  expect_true(inherits(print(p), "plotly"))
  expect_invisible(print(p))

  # Single [ index
  expect_s4_class(p[1,c(3,4,6)], "innsight_plotly")

  # Double [[ index
  expect_true(inherits(p[[1,1]], "plotly"))
})
