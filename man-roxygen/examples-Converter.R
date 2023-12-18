#' @examplesIf torch::torch_is_installed()
#' #----------------------- Example 1: Torch ----------------------------------
#' library(torch)
#'
#' model <- nn_sequential(
#'   nn_linear(5, 10),
#'   nn_relu(),
#'   nn_linear(10, 2, bias = FALSE),
#'   nn_softmax(dim = 2)
#' )
#' data <- torch_randn(25, 5)
#'
#' # Convert the model (for torch models is 'input_dim' required!)
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # You can also use the helper function `convert()` for initializing a
#' # Converter object
#' converter <- convert(model, input_dim = c(5))
#'
#' # Get the converted model stored in the field 'model'
#' converted_model <- converter$model
#'
#' # Test it with the original model
#' mean(abs(converted_model(data)[[1]] - model(data)))
#'
#'
#' #----------------------- Example 2: Neuralnet ------------------------------
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
#'   converter <- convert(nn)
#'
#'   # Print all the layers
#'   converter$model$modules_list
#' }
#'
#' @examplesIf keras::is_keras_available() & torch::torch_is_installed()
#' #----------------------- Example 3: Keras ----------------------------------
#' if (require("keras")) {
#'   library(keras)
#'
#'   # Make sure keras is installed properly
#'   is_keras_available()
#'
#'   # Define a keras model
#'   model <- keras_model_sequential() %>%
#'     layer_conv_2d(
#'       input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
#'       activation = "relu", padding = "same") %>%
#'     layer_conv_2d(
#'       kernel_size = 8, filters = 4,
#'       activation = "tanh", padding = "same") %>%
#'     layer_conv_2d(
#'       kernel_size = 4, filters = 2,
#'       activation = "relu", padding = "same") %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 1, activation = "sigmoid")
#'
#'   # Convert this model and save model as list
#'   converter <- convert(model, save_model_as_list = TRUE)
#'
#'   # Print the converted model as a named list
#'   str(converter$model_as_list, max.level = 1)
#' }
#'
#' @examplesIf torch::torch_is_installed()
#' #----------------------- Example 4: List  ----------------------------------
#'
#' # Define a model
#'
#' model <- list()
#' model$input_dim <- 5
#' model$input_names <- list(c("Feat1", "Feat2", "Feat3", "Feat4", "Feat5"))
#' model$input_nodes <- c(1)
#' model$output_dim <- 2
#' model$output_names <- list(c("Cat", "no-Cat"))
#' model$output_nodes <- c(2)
#' model$layers$Layer_1 <-
#'   list(
#'     type = "Dense",
#'     weight = matrix(rnorm(5 * 20), 20, 5),
#'     bias = rnorm(20),
#'     activation_name = "tanh",
#'     dim_in = 5,
#'     dim_out = 20,
#'     input_layers = 0, # '0' means model input layer
#'     output_layers = 2
#'   )
#' model$layers$Layer_2 <-
#'   list(
#'     type = "Dense",
#'     weight = matrix(rnorm(20 * 2), 2, 20),
#'     bias = rnorm(2),
#'     activation_name = "softmax",
#'     input_layers = 1,
#'     output_layers = -1 # '-1' means model output layer
#'     #dim_in = 20, # These values are optional, but
#'     #dim_out = 2  # useful for internal checks
#'   )
#'
#' # Convert the model
#' converter <- convert(model)
#'
#' # Get the model as a torch::nn_module
#' torch_model <- converter$model
#'
#' # You can use it as a normal torch model
#' x <- torch::torch_randn(3, 5)
#' torch_model(x)
