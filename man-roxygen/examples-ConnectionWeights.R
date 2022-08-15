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
#' # Print the result as a data.frame
#' cw$get_result("data.frame")
#'
#' # Plot the result
#' plot(cw)
#'
#' #----------------------- Example 2: Neuralnet ------------------------------
#' library(neuralnet)
#' data(iris)
#'
#' # Train a Neural Network
#' nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(3, 2), act.fct = "tanh", rep = 1
#' )
#'
#' # Convert the trained model
#' converter <- Converter$new(nn)
#'
#' # Apply the Connection Weights method
#' cw <- ConnectionWeights$new(converter)
#'
#' # Get the result as a torch tensor
#' cw$get_result(type = "torch.tensor")
#'
#' # Plot the result
#' plot(cw)
#'
#' #----------------------- Example 3: Keras ----------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   # Define a model
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_1d(
#'       input_shape = c(64, 3), kernel_size = 16, filters = 8,
#'       activation = "softplus"
#'     ) %>%
#'     layer_conv_1d(kernel_size = 16, filters = 4, activation = "tanh") %>%
#'     layer_conv_1d(kernel_size = 16, filters = 2, activation = "relu") %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 2, activation = "softmax")
#'
#'   # Convert the model
#'   converter <- Converter$new(model)
#'
#'   # Apply the Connection Weights method
#'   cw <- ConnectionWeights$new(converter)
#'
#'   # Get the result as data.frame
#'   cw$get_result(type = "data.frame")
#'
#'   # Plot the result for all classes
#'   plot(cw, output_idx = 1:2)
#' }
#'
#' # ------------------------- Advanced: Plotly -------------------------------
#' # If you want to create an interactive plot of your results with custom
#' # changes, you can take use of the method plotly::ggplotly
#' library(ggplot2)
#' library(plotly)
#' library(neuralnet)
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
#' # create new instance of 'LRP'
#' cw <- ConnectionWeights$new(converter)
#'
#' library(plotly)
#'
#' # Get the ggplot and add your changes
#' p <- plot(cw, output_idx = 1) +
#'   theme_bw() +
#'   scale_fill_gradient2(low = "green", mid = "black", high = "blue")
#'
#' # Now apply the method plotly::ggplotly with argument tooltip = "text"
#' plotly::ggplotly(p, tooltip = "text")
