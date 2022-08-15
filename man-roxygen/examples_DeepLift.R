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
#' ref <- torch_randn(1, 5)
#'
#' # Create Converter
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # Apply method DeepLift
#' deeplift <- DeepLift$new(converter, data, x_ref = ref)
#'
#' # Print the result as a torch tensor for first two data points
#' deeplift$get_result("torch.tensor")[1:2]
#'
#' # Plot the result for both classes
#' plot(deeplift, output_idx = 1:2)
#'
#' # Plot the boxplot of all datapoints
#' boxplot(deeplift, output_idx = 1:2)
#'
#' # ------------------------- Example 2: Neuralnet ---------------------------
#' library(neuralnet)
#' data(iris)
#'
#' # Train a neural network
#' nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(3, 2), act.fct = "tanh", rep = 1
#' )
#'
#' # Convert the model
#' converter <- Converter$new(nn)
#'
#' # Apply DeepLift with rescale-rule and a reference input of the feature
#' # means
#' x_ref <- matrix(colMeans(iris[, c(3, 4)]), nrow = 1)
#' deeplift_rescale <- DeepLift$new(converter, iris[, c(3, 4)], x_ref = x_ref)
#'
#' # Get the result as a dataframe and show first 5 rows
#' deeplift_rescale$get_result(type = "data.frame")[1:5, ]
#'
#' # Plot the result for the first datapoint in the data
#' plot(deeplift_rescale, data_idx = 1)
#'
#' # Plot the result as boxplots
#' boxplot(deeplift_rescale)
#'
#' # ------------------------- Example 3: Keras -------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   data <- array(rnorm(10 * 32 * 32 * 3), dim = c(10, 32, 32, 3))
#'
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_2d(
#'       input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
#'       activation = "softplus", padding = "valid"
#'     ) %>%
#'     layer_conv_2d(
#'       kernel_size = 8, filters = 4, activation = "tanh",
#'       padding = "same"
#'     ) %>%
#'     layer_conv_2d(
#'       kernel_size = 4, filters = 2, activation = "relu",
#'       padding = "valid"
#'     ) %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 16, activation = "relu") %>%
#'     layer_dense(units = 2, activation = "softmax")
#'
#'   # Convert the model
#'   converter <- Converter$new(model)
#'
#'   # Apply the DeepLift method with reveal-cancel rule
#'   deeplift_revcancel <- DeepLift$new(converter, data,
#'     channels_first = FALSE,
#'     rule_name = "reveal_cancel"
#'   )
#'
#'   # Plot the result for the first image and both classes
#'   plot(deeplift_revcancel, output_idx = 1:2)
#'
#'   # Plot the result as boxplots for first class
#'   boxplot(deeplift_revcancel, output_idx = 1)
#'
#'   # You can also create an interactive plot with plotly.
#'   # This is a suggested package, so make sure that it is installed
#'   library(plotly)
#'   boxplot(deeplift_revcancel, as_plotly = TRUE)
#' }
#'
#' # ------------------------- Advanced: Plotly -------------------------------
#' # If you want to create an interactive plot of your results with custom
#' # changes, you can take use of the method plotly::ggplotly
#' library(ggplot2)
#' library(neuralnet)
#' library(plotly)
#' data(iris)
#'
#' nn <- neuralnet(Species ~ .,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
#' )
#' # create an converter for this model
#' converter <- Converter$new(nn)
#'
#' # create new instance of 'DeepLift'
#' deeplift <- DeepLift$new(converter, iris[, -5])
#'
#' # Get the ggplot and add your changes
#' p <- plot(deeplift, output_idx = 1, data_idx = 1:2) +
#'   theme_bw() +
#'   scale_fill_gradient2(low = "green", mid = "black", high = "blue")
#'
#' # Now apply the method plotly::ggplotly with argument tooltip = "text"
#' plotly::ggplotly(p, tooltip = "text")
