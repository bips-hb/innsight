
#' @title Deep Learning Important FeaTures (DeepLift) Method
#'
#' @description
#' This is an implementation of the \emph{Deep Learning Important FeaTures
#' (DeepLift)} algorithm introduced by Shrikumar et al. (2017). It's a local
#' method for interpreting a single element \eqn{x} of the dataset concerning
#' a reference value \eqn{x'} and returns the contribution of each input
#' feature from the difference of the output (\eqn{y=f(x)}) and reference
#' output (\eqn{y'=f(x')}) prediction. The basic idea of this method is to
#' decompose the difference-from-reference prediction with respect to the
#' input features, i.e.
#' \deqn{\Delta y = y - y'  = \sum_i C(x_i).}
#' Compared to \emph{Layer-wise Relevance Propagation} (see [LRP]) is the
#' DeepLIFT method an exact decomposition and not an approximation, so we
#' get real contributions of the input features to the
#' difference-from-reference prediction. There are two ways to handle
#' activation functions: \emph{Rescale-Rule} and \emph{RevealCancel-Rule}.
#'
#' @examples
#' # ------------------------- Example 1: Neuralnet ---------------------------
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
#' # Get the result as a dataframe
#' deeplift_rescale$get_result(type = "data.frame")
#'
#' # Plot the result for the first datapoint in the data
#' plot(deeplift_rescale, data_id = 1)
#'
#' # ------------------------- Example 2: Keras -------------------------------
#' library(keras)
#'
#' data <- array(rnorm(64 * 32 * 32 * 3), dim = c(64, 32, 32, 3))
#'
#' model <- keras_model_sequential()
#' model %>%
#'   layer_conv_2d(
#'     input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
#'     activation = "softplus", padding = "valid"
#'   ) %>%
#'   layer_conv_2d(
#'     kernel_size = 8, filters = 4, activation = "tanh",
#'     padding = "same"
#'   ) %>%
#'   layer_conv_2d(
#'     kernel_size = 4, filters = 2, activation = "relu",
#'     padding = "valid"
#'   ) %>%
#'   layer_flatten() %>%
#'   layer_dense(units = 64, activation = "relu") %>%
#'   layer_dense(units = 16, activation = "relu") %>%
#'   layer_dense(units = 2, activation = "softmax")
#'
#' # Convert the model
#' converter <- Converter$new(model)
#'
#' # Apply the DeepLift method with reveal-cancel rule
#' deeplift_revcancel <- DeepLift$new(converter, data,
#'   channels_first = FALSE,
#'   rule_name = "reveal_cancel"
#' )
#'
#' # Plot the result for the first image and both classes
#' plot(deeplift_revcancel, class_id = 1:2)
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#' @export
#'

DeepLift <- R6::R6Class(
  classname = "DeepLift",
  inherit = InterpretingMethod,
  public = list(

    #' @field x_ref The reference input of size (1, dim_in) for the
    #' interpretation.
    #' @field rule_name Name of the applied rule to calculate the contributions
    #' for the non-linear part of a Neural Network layer. Either
    #' \code{"rescale"} or \code{"reveal_cancel"}.
    #'
    rule_name = NULL,
    x_ref = NULL,


    #' @description
    #' Create a new instance of the DeepLift method.
    #'
    #' @param converter An instance of the R6 class \code{\link{Converter}}.
    #' @param data The data for which the contribution scores are to be
    #' calculated. It has to be an array or array-like format of size
    #' (batch_size, dim_in).
    #' @param channels_first Set the data format of the given data. Internally
    #' the format `channels_first` is used, therefore the format of the given
    #' data is required. Also use the default value `TRUE` if no convolutional
    #' layers are used.
    #' @param ignore_last_act Set this boolean value to include the last
    #' activation, or not (default: `TRUE`). In some cases, the last activation
    #' leads to a saturation problem.
    #' @param dtype The data type for the calculations. Use either `'float'`
    #' (default) or `'double'`.
    #' @param x_ref The reference input of size (1, dim_in) for the
    #' interpretation. With the default value \code{NULL} you use an input
    #' of zeros.
    #' @param rule_name Name of the applied rule to calculate the
    #' contributions. Use one of `'rescale'` and `'reveal_cancel'`.
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          dtype = "float",
                          ignore_last_act = TRUE,
                          rule_name = "rescale",
                          x_ref = NULL) {
      super$initialize(converter, data, channels_first, dtype, ignore_last_act)

      checkmate::assertChoice(rule_name, c("rescale", "reveal_cancel"))
      self$rule_name <- rule_name

      if (is.null(x_ref)) {
        x_ref <- array(0, dim = c(1, dim(data)[-1]))
      }
      self$x_ref <- private$test_data(x_ref, name = "x_ref")

      self$converter$model$forward(self$data,
        channels_first = self$channels_first
      )
      self$converter$model$update_ref(self$x_ref,
        channels_first = self$channels_first
      )


      self$result <- private$run()
    }
  ),
  private = list(
    run = function() {
      rev_layers <- rev(self$converter$model$modules_list)
      last_layer <- rev_layers[[1]]
      rev_layers <- rev_layers[-1]

      mul <- torch::torch_diag_embed(torch::torch_ones_like(last_layer$output))

      if (self$ignore_last_act &&
        !("Flatten_Layer" %in% last_layer$".classes")) {
        mul <- last_layer$get_input_multiplier(mul,
          rule_name = "ignore_last_act"
        )
      } else {
        mul <- last_layer$get_input_multiplier(mul, self$rule_name)
      }

      # other layers
      for (layer in rev_layers) {
        if ("Flatten_Layer" %in% layer$".classes") {
          mul <- layer$reshape_to_input(mul)
        } else {
          mul <- layer$get_input_multiplier(mul, self$rule_name)
        }
      }
      if (!self$channels_first) {
        mul <- torch::torch_movedim(mul, 2, length(dim(mul)) - 1)
      }
      x_diff <- (self$data - self$x_ref)$unsqueeze(-1)

      mul * x_diff
    }
  )
)
