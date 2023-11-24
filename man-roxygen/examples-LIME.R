#' @examplesIf torch::torch_is_installed()
#' #----------------------- Example 1: Torch -----------------------------------
#' library(torch)
#'
#' # Create nn_sequential model and data
#' model <- nn_sequential(
#'   nn_linear(5, 12),
#'   nn_relu(),
#'   nn_linear(12, 2),
#'   nn_softmax(dim = 2)
#'   )
#' data <- torch_randn(25, 5)
#'
#' # Calculate LIME for the first 10 instances and set the
#' # feature and outcome names
#' lime <- LIME$new(model, data, data[1:10, ],
#'                  input_names = c("Car", "Cat", "Dog", "Plane", "Horse"),
#'                  output_names = c("Buy it!", "Don't buy it!"))
#'
#' # Get the result as an array for the first two instances
#' get_result(lime)[1:2,, ]
#'
#' # Plot the result for both classes
#' plot(lime, output_idx = c(1, 2))
#'
#' # Show the boxplot over all 10 instances
#' boxplot(lime, output_idx = c(1, 2))
#'
#' # We can also forward some arguments to lime::explain, e.g. n_permutatuins
#' # to get more accurate values
#' lime <- LIME$new(model, data, data[1:10, ],
#'                  input_names = c("Car", "Cat", "Dog", "Plane", "Horse"),
#'                  output_names = c("Buy it!", "Don't buy it!"),
#'                  n_perturbations = 500)
#'
#' # Plot the boxplots again
#' boxplot(lime, output_idx = c(1, 2))
#'
#' #----------------------- Example 2: Converter object --------------------------
#' # We can do the same with an Converter object (all feature and outcome names
#' # will be extracted by the LIME method!)
#' conv <- Converter$new(model,
#'                       input_dim = c(5),
#'                       input_names = c("Car", "Cat", "Dog", "Plane", "Horse"),
#'                       output_names = c("Buy it!", "Don't buy it!"))
#'
#' # Calculate LIME for the first 10 instances
#' lime <- LIME$new(conv, data, data[1:10], n_perturbations = 400)
#'
#' # Plot the result for both classes
#' plot(lime, output_idx = c(1, 2))
#'
#' #----------------------- Example 3: Other model -------------------------------
#' if (require("neuralnet") & require("ranger")) {
#'   library(neuralnet)
#'   library(ranger)
#'   data(iris)
#'
#'   # Fit a random forest unsing the ranger package
#'   model <- ranger(Species ~ ., data = iris, probability = TRUE)
#'
#'   # There is no pre-implemented predict function for ranger models, i.e.,
#'   # we have to define it ourselves.
#'   pred_fun <- function(newdata, ...) {
#'     predict(model, newdata, ...)$predictions
#'   }
#'
#'   # Calculate LIME for the instances of index 1 and 111 and add
#'   # the outcome labels (for LIME, the output_type is required!)
#'   lime <- LIME$new(model, iris[, -5], iris[c(1, 111), -5],
#'                    pred_fun = pred_fun,
#'                    output_type = "classification",
#'                    output_names = levels(iris$Species),
#'                    n_perturbations = 500)
#'
#'   # Plot the result for the first two classes and all selected instances
#'   plot(lime, data_idx = 1:2, output_idx = 1:2)
#'
#'   # Get the result as a torch_tensor
#'   get_result(lime, "torch_tensor")
#' }
