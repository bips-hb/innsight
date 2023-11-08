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
#' # Create a reference dataset for the estimation of the conditional
#' # expectation
#' ref <- torch_randn(5, 5)
#'
#' # Create Converter
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # Apply method DeepSHAP
#' deepshap <- DeepSHAP$new(converter, data, data_ref = ref)
#'
#' # Print the result as a torch tensor for first two data points
#' get_result(deepshap, "torch.tensor")[1:2]
#'
#' # Plot the result for both classes
#' plot(deepshap, output_idx = 1:2)
#'
#' # Plot the boxplot of all datapoints and for both classes
#' boxplot(deepshap, output_idx = 1:2)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' if (require("neuralnet")) {
#'   library(neuralnet)
#'   data(iris)
#'
#'   # Train a neural network
#'   nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#'     iris,
#'     linear.output = FALSE,
#'     hidden = c(3, 2), act.fct = "tanh", rep = 1
#'   )
#'
#'   # Convert the model
#'   converter <- Converter$new(nn)
#'
#'   # Apply DeepSHAP with rescale-rule and a 100 (default of `limit_ref`)
#'   # instances as the reference dataset
#'   deepshap <- DeepSHAP$new(converter, iris[, c(3, 4)],
#'                            data_ref = iris[, c(3, 4)])
#'
#'   # Get the result as a dataframe and show first 5 rows
#'   get_result(deepshap, type = "data.frame")[1:5, ]
#'
#'   # Plot the result for the first datapoint in the data
#'   plot(deepshap, data_idx = 1)
#'
#'   # Plot the result as boxplots
#'   boxplot(deepshap)
#' }
#'
#' @examplesIf keras::is_keras_available() & torch::torch_is_installed()
#' # ------------------------- Example 3: Keras -------------------------------
#' if (require("keras")) {
#'   library(keras)
#'
#'   # Make sure keras is installed properly
#'   is_keras_available()
#'
#'   data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))
#'
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_2d(
#'       input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
#'       activation = "softplus", padding = "valid") %>%
#'     layer_conv_2d(
#'       kernel_size = 8, filters = 4, activation = "tanh",
#'       padding = "same") %>%
#'     layer_conv_2d(
#'       kernel_size = 4, filters = 2, activation = "relu",
#'       padding = "valid") %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 2, activation = "softmax")
#'
#'   # Convert the model
#'   converter <- Converter$new(model)
#'
#'   # Apply the DeepSHAP method with zero baseline (wich is equivalent to
#'   # DeepLift with zero baseline)
#'   deepshap <- DeepSHAP$new(converter, data, channels_first = FALSE)
#'
#'   # Plot the result for the first image and both classes
#'   plot(deepshap, output_idx = 1:2)
#'
#'   # Plot the result as boxplots for first class
#'   boxplot(deepshap, output_idx = 1)
#' }
#' @examplesIf torch::torch_is_installed() & Sys.getenv("RENDER_PLOTLY", unset = 0) == 1
#' #------------------------- Plotly plots ------------------------------------
#' if (require("plotly")) {
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'   boxplot(deepshap, as_plotly = TRUE)
#' }
