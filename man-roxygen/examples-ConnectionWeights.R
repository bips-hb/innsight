#' @examplesIf torch::torch_is_installed()
#' #----------------------- Example 1: Torch ----------------------------------
#' library(torch)
#'
#' # Create nn_sequential model
#' model <- nn_sequential(
#'   nn_linear(5, 12),
#'   nn_relu(),
#'   nn_linear(12, 1),
#'   nn_sigmoid()
#' )
#'
#' # Create Converter with input names
#' converter <- Converter$new(model,
#'   input_dim = c(5),
#'   input_names = list(c("Car", "Cat", "Dog", "Plane", "Horse"))
#' )
#'
#' # Apply method Connection Weights
#' cw <- ConnectionWeights$new(converter)
#'
#' # Print the head of the result as a data.frame
#' head(get_result(cw, "data.frame"), 5)
#'
#' # Plot the result
#' plot(cw)
#'
#' #----------------------- Example 2: Neuralnet ------------------------------
#' if (require("neuralnet")) {
#'   library(neuralnet)
#'   data(iris)
#'
#'   # Train a Neural Network
#'   nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#'     iris,
#'     linear.output = FALSE,
#'     hidden = c(3, 2), act.fct = "tanh", rep = 1
#'   )
#'
#'   # Convert the trained model
#'   converter <- Converter$new(nn)
#'
#'   # Apply the Connection Weights method
#'   cw <- ConnectionWeights$new(converter)
#'
#'   # Get the result as a torch tensor
#'   get_result(cw, type = "torch.tensor")
#'
#'   # Plot the result
#'   plot(cw)
#' }
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
#'   # Apply the Connection Weights method
#'   cw <- ConnectionWeights$new(converter)
#'
#'   # Get the head of the result as a data.frame
#'   head(get_result(cw, type = "data.frame"), 5)
#'
#'   # Plot the result for all classes
#'   plot(cw, output_idx = 1:2)
#' }
#' @examplesIf torch::torch_is_installed() & Sys.getenv("RENDER_PLOTLY", unset = 0) == 1
#' #------------------------- Plotly plots ------------------------------------
#' if (require("plotly")) {
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'   plot(cw, as_plotly = TRUE)
#' }
