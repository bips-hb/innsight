

################################################################################
#                         Lime (from the package lime)
################################################################################

#' Local interpretable model-agnostic explanations (LIME)
#'
#' @description
#' The R6 class `LIME` calculates the feature weights of a linear surrogate of
#' the prediction model for a instance to be explained, namely the
#' *local interpretable model-agnostic explanations (LIME)*. It is a
#' model-agnostic method that can be applied to any predictive model.
#' This means, in particular, that
#' `LIME` can be applied not only to objects of the [`Converter`] class but
#' also to any other model. The only requirement is the argument `pred_fun`,
#' which generates predictions with the model for given data. However, this
#' function is pre-implemented for models created with
#' \code{\link[torch]{nn_sequential}}, \code{\link[keras]{keras_model}},
#' \code{\link[neuralnet]{neuralnet}} or [`Converter`]. Internally, the
#' suggested package `lime` is utilized and applied to `data.frame`.
#'
#' **Note:** Even signal and image data are initially transformed into a
#' `data.frame` using `as.data.frame()` and then [`lime::lime`] and
#' [`lime::explain`] are
#' applied. In other words, a custom `pred_fun` may need to convert the
#' `data.frame` back into an `array` as necessary.
#'
#' @template param-output_idx
#' @template param-output_label
#' @template param-channels_first
#' @template param-model-agnostic
#' @template param-data-agnostic
#' @template param-x-agnostic
#' @template param-output_type-agnostic
#' @template param-pred_fun-agnostic
#' @template param-input_dim-agnostic
#' @template param-input_names-agnostic
#' @template param-output_names-agnostic
#' @template examples-LIME
#'
#' @family methods
#' @export
LIME <- R6Class(
  classname = "LIME",
  inherit = AgnosticWrapper,
  public = list(

    #' @description
    #' Create a new instance of the `LIME` R6 class. When initialized,
    #' the method *LIME* is applied to the given data and the results are
    #' stored in the field `result`.
    #'
    #' @param ... other arguments forwarded to [`lime::explain`].
    initialize = function(model, data, x,
                          output_type = NULL,
                          pred_fun = NULL,
                          output_idx = NULL,
                          output_label = NULL,
                          channels_first = TRUE,
                          input_dim = NULL,
                          input_names = NULL,
                          output_names = NULL, ...) {

      # Check if data or x is a torch_tensor
      if (inherits(data, "torch_tensor")) data <- as.array(data)
      if (inherits(x, "torch_tensor")) x <- as.array(x)

      super$initialize(model, data, x, output_type, pred_fun, output_idx,
                       output_label, channels_first, input_dim, input_names,
                       output_names)

      # Get the pre-processed x
      x <- self$x

      # We use the lime package for the explanation
      if (!requireNamespace("lime", quietly = TRUE)) {
        stopf("Package {.pkg lime} must be installed to use this function!")
      }

      # Create the explainer of the lime package
      explainer <- lime::lime(data.frame(data), self$converter)

      # Apply lime
      if (self$converter$output_type == "classification") {
        res <- lime::explain(data.frame(x), explainer,
                        labels = self$converter$output_names[[1]][[1]][self$output_idx[[1]]],
                        n_features = prod(self$converter$input_dim[[1]]),
                        input_dim = self$converter$input_dim[[1]], ...)
        res_dim <- c(dim(x)[-1], length(self$output_idx[[1]]), nrow(x))
        result <- torch_tensor(array(res$feature_weight, dim = res_dim))
        result <- result$movedim(-1, 1)
      } else {
        apply_lime <- function(idx) {
          tmp_res <- lime::explain(data.frame(x), explainer,
                        n_features = prod(self$converter$input_dim[[1]]),
                        input_dim = self$converter$input_dim[[1]],
                        idx = idx, ...)
          res_dim <- c(dim(x)[-1], nrow(x))
          tmp_res <- torch_tensor(array(tmp_res$feature_weight, dim = res_dim))
          tmp_res <- tmp_res$movedim(-1, 1)
        }
        res <- lapply(self$output_idx[[1]], apply_lime)
        result <- torch_stack(res, dim = -1)
      }

      self$result <- list(list(result))
    }
  )
)

# Add functions predict_model and model_type for the objects of class
# innsight_agnostic_wrapper

#' @exportS3Method lime::predict_model
predict_model.innsight_agnostic_wrapper <- function(x, newdata, type, idx, ...) {
  pred <- x$pred_fun(newdata = newdata, ...)
  if (type == "raw") {
    as.data.frame(pred[, idx, drop = FALSE])
  } else {
    if (!inherits(pred, c("data.frame", "matrix", "array"))) {
      pred <- as.array(pred)
    }
    colnames(pred) <- x$output_names[[1]][[1]]
    as.data.frame(pred, check.names = FALSE)
  }
}

#' @exportS3Method lime::model_type
model_type.innsight_agnostic_wrapper <- function(x, ...) {
  x$output_type
}

################################################################################
#                         SHAP (from the package fastshap)
################################################################################

#' Shapley values
#'
#' @description
#' The R6 class `SHAP` calculates the famous Shapley values based on game
#' theory for an instance to be explained. It is a model-agnostic method
#' that can be applied to any predictive model. This means, in particular, that
#' `SHAP` can be applied not only to objects of the [`Converter`] class but
#' also to any other model. The only requirement is the argument `pred_fun`,
#' which generates predictions with the model for given data. However, this
#' function is pre-implemented for models created with
#' \code{\link[torch]{nn_sequential}}, \code{\link[keras]{keras_model}},
#' \code{\link[neuralnet]{neuralnet}} or [`Converter`]. Internally, the
#' suggested package `fastshap` is utilized and applied to `data.frame`.
#'
#' **Note:** Even signal and image data are initially transformed into a
#' `data.frame` using `as.data.frame()` and then [`fastshap::explain`] is
#' applied. In other words, a custom `pred_fun` may need to convert the
#' `data.frame` back into an `array` as necessary.
#'
#' @template param-output_idx
#' @template param-output_label
#' @template param-channels_first
#' @template param-model-agnostic
#' @template param-data-agnostic
#' @template param-x-agnostic
#' @template param-pred_fun-agnostic
#' @template param-input_dim-agnostic
#' @template param-input_names-agnostic
#' @template param-output_names-agnostic
#' @template examples-SHAP
#'
#' @family methods
#' @export
SHAP <- R6Class(
  classname = "SHAP",
  inherit = AgnosticWrapper,
  public = list(

    #' @description
    #' Create a new instance of the `SHAP` R6 class. When initialized,
    #' the method *SHAP* is applied to the given data and the results are
    #' stored in the field `result`.
    #'
    #' @param ... other arguments forwarded to [`fastshap::explain`].
    initialize = function(model, data, x,
                          pred_fun = NULL,
                          output_idx = NULL,
                          output_label = NULL,
                          channels_first = TRUE,
                          input_dim = NULL,
                          input_names = NULL,
                          output_names = NULL, ...) {

      # output_type is not necessary for fastshap
      output_type <- "regression"

      # Check if data or x is a torch_tensor
      if (inherits(data, "torch_tensor")) data <- as.array(data)
      if (inherits(x, "torch_tensor")) x <- as.array(x)

      super$initialize(model, data, x, output_type, pred_fun, output_idx,
                       output_label, channels_first, input_dim, input_names,
                       output_names)

      # We use the fastshap package for the explanation
      if (!requireNamespace("fastshap", quietly = TRUE)) {
        stopf("Package {.pkg fastshap} must be installed to use this function!")
      }

      # Function for calculating Shapley values for a specific output
      apply_shap <- function(idx, input_dim) {
        pred_wrapper <- function(object, newdata, ...) {
          self$converter$pred_fun(newdata = newdata, input_dim = input_dim, ...)[, idx]
        }

        res <- fastshap::explain(
          self$converter,
          X = as.data.frame(data),
          newdata = as.data.frame(self$x),
          pred_wrapper = pred_wrapper, ...)
        dim(res) <- dim(self$x)
        res
      }

      # Calculate Shapley values for all outputs
      result <- lapply(self$output_idx[[1]], apply_shap,
                       input_dim = self$converter$input_dim[[1]])

      # Reshape the result to (batch_size, input_dim, output_idx)
      result <- torch_stack(result, dim = -1)

      # Save result
      self$result <- list(list(result))
    }
  )
)

