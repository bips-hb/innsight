#' @examplesIf torch::torch_is_installed()
#' # ------------------------- Example 1: Torch -------------------------------
#' library(torch)
#'
#' # Create nn_sequential model and data
#' model <- nn_sequential(
#'   nn_linear(5, 10),
#'   nn_relu(),
#'   nn_linear(10, 2),
#'   nn_sigmoid()
#' )
#' data <- torch_randn(25, 5)
#'
#' # Create Converter
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # Calculate the smoothed Gradients
#' smoothgrad <- SmoothGrad$new(converter, data)
#'
#' # Print the result as a data.frame for first 5 rows
#' smoothgrad$get_result("data.frame")[1:5, ]
#'
#' # Plot the result for both classes
#' plot(smoothgrad, output_idx = 1:2)
#'
#' # Plot the boxplot of all datapoints
#' boxplot(smoothgrad, output_idx = 1:2)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' library(neuralnet)
#' data(iris)
#'
#' # Train a neural network
#' nn <- neuralnet(Species ~ ., iris,
#'   linear.output = FALSE,
#'   hidden = c(10, 5),
#'   act.fct = "logistic",
#'   rep = 1
#' )
#'
#' # Convert the trained model
#' converter <- Converter$new(nn)
#'
#' # Calculate the smoothed gradients
#' smoothgrad <- SmoothGrad$new(converter, iris[, -5], times_input = FALSE)
#'
#' # Plot the result for the first and 60th data point and all classes
#' plot(smoothgrad, data_idx = c(1, 60), output_idx = 1:3)
#'
#' # Calculate SmoothGrad x Input and do not ignore the last activation
#' smoothgrad <- SmoothGrad$new(converter, iris[, -5], ignore_last_act = FALSE)
#'
#' # Plot the result again
#' plot(smoothgrad, data_idx = c(1, 60), output_idx = 1:3)
#'
#' # ------------------------- Example 3: Keras -------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   data <- array(rnorm(64 * 60 * 3), dim = c(64, 60, 3))
#'
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_1d(
#'       input_shape = c(60, 3), kernel_size = 8, filters = 8,
#'       activation = "softplus", padding = "valid"
#'     ) %>%
#'     layer_conv_1d(
#'       kernel_size = 8, filters = 4, activation = "tanh",
#'       padding = "same"
#'     ) %>%
#'     layer_conv_1d(
#'       kernel_size = 4, filters = 2, activation = "relu",
#'       padding = "valid"
#'     ) %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 3, activation = "softmax")
#'
#'   # Convert the model
#'   converter <- Converter$new(model)
#'
#'   # Apply the SmoothGrad method
#'   smoothgrad <- SmoothGrad$new(converter, data, channels_first = FALSE)
#'
#'   # Plot the result for the first datapoint and all classes
#'   plot(smoothgrad, output_idx = 1:3)
#'
#'   # Plot the result as boxplots for first two classes
#'   boxplot(smoothgrad, output_idx = 1:2)
#'
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'
#'   # Result as boxplots
#'   boxplot(smoothgrad, as_plotly = TRUE)
#'
#'   # Result of the second data point
#'   plot(smoothgrad, data_idx = 2, as_plotly = TRUE)
#' }
