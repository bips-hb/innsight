#' @examplesIf torch::torch_is_installed()
#' #----------------------- Example 1: Torch ----------------------------------
#' library(torch)
#'
#' # Create nn_sequential model and data
#' model <- nn_sequential(
#'   nn_linear(5, 12),
#'   nn_relu(),
#'   nn_linear(12, 2),
#'   nn_softmax(dim = 2)
#' )
#' data <- torch_randn(25, 5)
#'
#' # Create Converter with input and output names
#' converter <- convert(model,
#'   input_dim = c(5),
#'   input_names = list(c("Car", "Cat", "Dog", "Plane", "Horse")),
#'   output_names = list(c("Buy it!", "Don't buy it!"))
#' )
#'
#' # Calculate the Gradients
#' grad <- Gradient$new(converter, data)
#'
#' # You can also use the helper function `run_grad` for initializing
#' # an R6 Gradient object
#' grad <- run_grad(converter, data)
#'
#' # Print the result as a data.frame for first 5 rows
#' get_result(grad, "data.frame")[1:5,]
#'
#' # Plot the result for both classes
#' plot(grad, output_idx = 1:2)
#'
#' # Plot the boxplot of all datapoints
#' boxplot(grad, output_idx = 1:2)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' if (require("neuralnet")) {
#'   library(neuralnet)
#'   data(iris)
#'
#'   # Train a neural network
#'   nn <- neuralnet(Species ~ ., iris,
#'     linear.output = FALSE,
#'     hidden = c(10, 5),
#'     act.fct = "logistic",
#'     rep = 1
#'   )
#'
#'   # Convert the trained model
#'   converter <- convert(nn)
#'
#'   # Calculate the gradients
#'   gradient <- run_grad(converter, iris[, -5])
#'
#'   # Plot the result for the first and 60th data point and all classes
#'   plot(gradient, data_idx = c(1, 60), output_idx = 1:3)
#'
#'   # Calculate Gradients x Input and do not ignore the last activation
#'   gradient <- run_grad(converter, iris[, -5],
#'                        ignore_last_act = FALSE,
#'                        times_input = TRUE)
#'
#'   # Plot the result again
#'   plot(gradient, data_idx = c(1, 60), output_idx = 1:3)
#' }
#'
#' @examplesIf torch::torch_is_installed() & Sys.getenv("INNSIGHT_EXAMPLE_KERAS", unset = 0) == 1
#' # ------------------------- Example 3: Keras -------------------------------
#' if (require("keras") & keras::is_keras_available()) {
#'   library(keras)
#'
#'   # Make sure keras is installed properly
#'   is_keras_available()
#'
#'   data <- array(rnorm(64 * 60 * 3), dim = c(64, 60, 3))
#'
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_1d(
#'       input_shape = c(60, 3), kernel_size = 8, filters = 8,
#'       activation = "softplus", padding = "valid") %>%
#'     layer_conv_1d(
#'       kernel_size = 8, filters = 4, activation = "tanh",
#'       padding = "same") %>%
#'     layer_conv_1d(
#'       kernel_size = 4, filters = 2, activation = "relu",
#'       padding = "valid") %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 3, activation = "softmax")
#'
#'   # Convert the model
#'   converter <- convert(model)
#'
#'   # Apply the Gradient method
#'   gradient <- run_grad(converter, data, channels_first = FALSE)
#'
#'   # Plot the result for the first datapoint and all classes
#'   plot(gradient, output_idx = 1:3)
#'
#'   # Plot the result as boxplots for first two classes
#'   boxplot(gradient, output_idx = 1:2)
#' }
#'
#' @examplesIf torch::torch_is_installed() & Sys.getenv("RENDER_PLOTLY", unset = 0) == 1
#' #------------------------- Plotly plots ------------------------------------
#' if (require("plotly")) {
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'
#'   # Result as boxplots
#'   boxplot(gradient, as_plotly = TRUE)
#'
#'   # Result of the second data point
#'   plot(gradient, data_idx = 2, as_plotly = TRUE)
#' }
