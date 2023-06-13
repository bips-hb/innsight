#' @examplesIf torch::torch_is_installed()
#'  #----------------------- Example 1: Torch ----------------------------------
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
#' # Create Converter
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # Apply method LRP with simple rule (default)
#' lrp <- LRP$new(converter, data)
#'
#' # Print the result as an array for data point one and two
#' get_result(lrp)[1:2,,]
#'
#' # Plot the result for both classes
#' plot(lrp, output_idx = 1:2)
#'
#' # Plot the boxplot of all datapoints without a preprocess function
#' boxplot(lrp, output_idx = 1:2, preprocess_FUN = identity)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' if (require("neuralnet")) {
#'   library(neuralnet)
#'   data(iris)
#'   nn <- neuralnet(Species ~ .,
#'     iris,
#'     linear.output = FALSE,
#'     hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
#'   )
#'
#'   # Create an converter for this model
#'   converter <- Converter$new(nn)
#'
#'   # Create new instance of 'LRP'
#'   lrp <- LRP$new(converter, iris[, -5], rule_name = "simple")
#'
#'   # Get the result as an array for data point one and two
#'   get_result(lrp)[1:2,,]
#'
#'   # Get the result as a torch tensor for data point one and two
#'   get_result(lrp, type = "torch.tensor")[1:2]
#'
#'   # Use the alpha-beta rule with alpha = 2
#'   lrp <- LRP$new(converter, iris[, -5],
#'     rule_name = "alpha_beta",
#'     rule_param = 2
#'   )
#'
#'   # Include the last activation into the calculation
#'   lrp <- LRP$new(converter, iris[, -5],
#'     rule_name = "alpha_beta",
#'     rule_param = 2,
#'     ignore_last_act = FALSE
#'   )
#'
#'   # Plot the result for all classes
#'   plot(lrp, output_idx = 1:3)
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
#'   data <- array(rnorm(10 * 60 * 3), dim = c(10, 60, 3))
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
#'   converter <- Converter$new(model)
#'
#'   # Apply the LRP method with the epsilon rule for the dense layers and
#'   # the alpha-beta rule for the convolutional layers
#'   lrp_comp <- LRP$new(converter, data,
#'     channels_first = FALSE,
#'     rule_name = list(Dense_Layer = "epsilon", Conv1D_Layer = "alpha_beta"),
#'     rule_param = list(Dense_Layer = 0.1, Conv1D_Layer = 1)
#'   )
#'
#'   # Plot the result for the first datapoint and all classes
#'   plot(lrp_comp, output_idx = 1:3)
#'
#'   # Plot the result as boxplots for first two classes
#'   boxplot(lrp_comp, output_idx = 1:2)
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
#'   boxplot(lrp, as_plotly = TRUE)
#'
#'   # Result of the second data point
#'   plot(lrp, data_idx = 2, as_plotly = TRUE)
#' }
