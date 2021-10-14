#'
#' @title Superclass for Gradient-based Interpretation Methods
#' @description Superclass for gradient-based interpretation methods. This
#' class inherits from [InterpretingMethod]. It summarizes all implemented
#' gradient-based methods and provides a private function to calculate the
#' gradients w.r.t. to the input for given data. Implemented are:
#'
#' - Normal Gradients and Gradient x Input ([Gradient])
#' - SmoothGrad and SmoothGrad x Input ([SmoothGrad])
#'
GradientBased <- R6::R6Class(
  classname = "GradientBased",
  inherit = InterpretingMethod,
  public = list(

    #' @field times_input Multiplies the gradients with the input features.
    #' This method is called 'Gradient x Input'.
    times_input = NULL,

    #' @description
    #' Create a new instance of this class.
    #'
    #' @param converter The converter of class [Converter] with the stored and
    #' torch-converted model.
    #' @param data The given data in an array-like format to be interpreted
    #' with the selected gradient-based method.
    #' @param channels_first The format of the given date, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data (either `'float'` or `'double'`).
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not. In some cases, the last
    #' activation leads to a saturation problem.
    #' @param times_input Multiplies the gradients with the input features. This
    #' method is called 'Gradient x Input'.
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          dtype = "float",
                          ignore_last_act = TRUE,
                          times_input = TRUE) {
      super$initialize(converter, data, channels_first, dtype, ignore_last_act)

      checkmate::assert_logical(times_input)
      self$times_input <- times_input
    }
  ),
  private = list(
    calculate_gradients = function(input) {
      input$requires_grad <- TRUE

      out <- self$converter$model(input, channels_first = self$channels_first)


      if (self$ignore_last_act) {
        output <- rev(self$converter$model$modules_list)[[1]]$preactivation
      } else {
        output <- out
      }

      # Implemented is only the case where the output is one-dimensional
      checkmate::assertTRUE(length(dim(output)) == 2)

      res <- vector(mode = "list", length = dim(output)[2])
      out_sum <- sum(output, dim = 1)

      for (i in 1:dim(output)[2]) {
        res[[i]] <-
          torch::autograd_grad(out_sum[i], input, retain_graph = TRUE)[[1]]
      }

      torch::torch_stack(res, dim = length(dim(input)) + 1)
    }
  )
)

#' @title Gradient Method
#' @name Gradient
#'
#' @description
#' This method computes the gradients (also known as Vanilla Gradients) of
#' the outputs with respect to the input variables, i.e. for all input
#' variable \eqn{i} and output class \eqn{j}
#' \deqn{d f(x)_j / d x_i.}
#' If the argument `times_input` is `TRUE`, the gradients are multiplied by
#' the respective input value (Gradient x Input), i.e.
#' \deqn{x_i * d f(x)_j / d x_i.}
#'
#' @examples
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
#' # Calculate the gradients
#' gradient <- Gradient$new(converter, iris[, -5], times_input = FALSE)
#'
#' # Plot the result for the first and 60th data point and all classes
#' # plot(gradient, data_id = c(1, 60), class_id = 1:3)
#'
#' # Calculate Gradients x Input and do not ignore the last activation
#' gradient <- Gradient$new(converter, iris[, -5], ignore_last_act = FALSE)
#'
#' # Plot the result again
#' # plot(gradient, data_id = c(1, 60), class_id = 1:3)
#' @export
#'

Gradient <- R6::R6Class(
  classname = "Gradient",
  inherit = GradientBased,
  public = list(

    #' @description
    #' Create a new instance of this class.
    #'
    #' @param converter The converter of class [Converter] with the stored
    #' and torch-converted model.
    #' @param data The given data in an array-like format to be interpreted
    #' with the this method.
    #' @param channels_first The format of the given data, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data (either `'float'` or `'double'`).
    #' Default: `'float'`.
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not. In some cases, the
    #' last activation leads to a saturation problem. Default: `TRUE`.
    #' @param times_input Multiplies the gradients with the input features.
    #' This method is called 'Gradient x Input'. Default: `TRUE`.
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          dtype = "float",
                          ignore_last_act = TRUE,
                          times_input = TRUE) {
      super$initialize(
        converter,
        data,
        channels_first,
        dtype,
        ignore_last_act,
        times_input
      )

      self$result <- private$run()
    }
  ),
  private = list(
    run = function() {
      gradients <- private$calculate_gradients(self$data)

      if (self$times_input) {
        gradients <- gradients * self$data$unsqueeze(-1)
      }

      gradients
    }
  )
)


#' @title SmoothGrad Method
#' @name SmoothGrad
#'
#' @description
#' SmoothGrad was introduced by D. Smilkov et al. (2017) and is an extension to
#' the classical [Gradient] method. It takes the mean of the gradients for
#' \code{n} perturbations of each data point, i.e. with
#' \eqn{\epsilon ~ N(0,\sigma)}
#' \deqn{1/n \sum_n d f(x+ \epsilon)_j / d x_j.}
#'
#' @references
#' D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @examples
#' library(neuralnet)
#' data(iris)
#'
#' # Train a neural network
#' nn <- neuralnet(Species ~ ., iris,
#'   linear.output = FALSE,
#'   hidden = c(5, 3),
#'   act.fct = "logistic",
#'   rep = 1
#' )
#'
#' # Convert the trained model
#' converter <- Converter$new(nn)
#'
#' # Calculate the smoothed gradients
#' gradient <- SmoothGrad$new(converter, iris[, -5], times_input = FALSE)
#'
#' # Plot the result for the first and 60th data point and all classes
#' # plot(gradient, data_id = c(1, 60), class_id = 1:3)
#' @export
#'
SmoothGrad <- R6::R6Class(
  classname = "SmoothGrad",
  inherit = GradientBased,
  public = list(

    #' @field n Number of perturbations of the input data (default: \eqn{50}).
    #' @field noise_level The standard deviation of the gaussian
    #' perturbation, i.e. \eqn{\sigma = (max(x) - min(x)) *} `noise_level`.
    #'
    n = NULL,
    noise_level = NULL,

    #' @description
    #' Create a new instance of this class.
    #'
    #' @param converter The converter of class [Converter] with the stored and
    #' torch-converted model.
    #' @param data The given data in an array-like format to be interpreted
    #' with the this method.
    #' @param channels_first The format of the given data, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data (either `'float'` or `'double'`).
    #' Default: `'float'`.
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not. In some cases, the last
    #' activation leads to a saturation problem. Default: `TRUE`.
    #' @param times_input Multiplies the smoothed gradients with the input
    #' features. This method is called 'SmoothGrad x Input'.
    #' @param n Number of perturbations of the input data (default: \eqn{50}).
    #' @param noise_level Determines the standard deviation of the gaussian
    #' perturbation, i.e. \eqn{\sigma = (max(x) - min(x)) *} `noise_level`.
    #'
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          dtype = "float",
                          ignore_last_act = TRUE,
                          times_input = TRUE,
                          n = 50,
                          noise_level = 0.1) {
      super$initialize(
        converter,
        data,
        channels_first,
        dtype,
        ignore_last_act,
        times_input
      )

      checkmate::assertInt(n, lower = 1)
      checkmate::assertNumber(noise_level, lower = 0)
      self$n <- n
      self$noise_level <- noise_level

      self$result <- private$run()
    }
  ),
  private = list(
    run = function() {
      data <-
        torch::torch_repeat_interleave(
          self$data,
          repeats = torch::torch_tensor(self$n, dtype = torch::torch_long()),
          dim = 1
        )

      noise_scale <- self$noise_level * (max(data) - min(data))

      noise <- torch::torch_randn_like(data) * noise_scale

      gradients <- private$calculate_gradients(data + noise)

      smoothgrads <-
        torch::torch_stack(lapply(gradients$chunk(dim(self$data)[1]),
          FUN = torch::torch_mean,
          dim = 1
        ),
        dim = 1
        )

      if (self$times_input) {
        smoothgrads <- smoothgrads * self$data$unsqueeze(-1)
      }

      smoothgrads
    }
  )
)
