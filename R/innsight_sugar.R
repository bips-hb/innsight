#' @title Syntactic sugar for object construction
#'
#' @name innsight_sugar
#' @description
#'
#' Since all methods and the preceding conversion step in the `innsight`
#' package were implemented using R6 classes and these always require a call
#' to `classname$new()` for initialization, the following functions are
#' defined to shorten the construction of the corresponding R6 objects:
#'
#' * `convert()` for [`Converter`]
#' * `run_grad()` for [`Gradient`]
#' * `run_smoothgrad()` for [`SmoothGrad`]
#' * `run_intgrad()` for [`IntegratedGradient`]
#' * `run_expgrad()` for [`ExpectedGradient`]
#' * `run_lrp()` for [`LRP`]
#' * `run_deeplift()` for [`DeepLift`]
#' * `run_deepshap` for [`DeepSHAP`]
#' * `run_cw` for [`ConnectionWeights`]
#' * `run_lime` for [`LIME`]
#' * `run_shap` for [`SHAP`]
#'
#' @template param-converter
#' @template param-data_ref-agnostic
#'
#' @param model (\code{\link[torch]{nn_sequential}}, \code{\link[keras]{keras_model}},
#' \code{\link[neuralnet]{neuralnet}} or `list`)\cr
#' A trained neural network for classification or regression
#' tasks to be interpreted. Only models from the following types or
#' packages are allowed: \code{\link[torch]{nn_sequential}},
#' \code{\link[keras]{keras_model}},
#' \code{\link[keras]{keras_model_sequential}},
#' \code{\link[neuralnet]{neuralnet}} or a named list (see details).\cr
#' **Note:** For the model-agnostic methods, an arbitrary fitted model for a
#' classification or regression task can be passed. A [`Converter`] object can
#' also be passed. In order for the package to know how to make predictions
#' with the given model, a prediction function must also be passed with
#' the argument `pred_fun`. However, for models created by
#' \code{\link[torch]{nn_sequential}}, \code{\link[keras]{keras_model}},
#' \code{\link[neuralnet]{neuralnet}} or [`Converter`],
#' these have already been pre-implemented and do not need to be
#' specified.\cr
#' @param data ([`array`], [`data.frame`], \code{\link[torch]{torch_tensor}} or `list`)\cr
#' The data to which the method is to be applied. These must
#' have the same format as the input data of the passed model to the
#' converter object. This means either
#' \itemize{
#'   \item an `array`, `data.frame`, `torch_tensor` or array-like format of
#'   size *(batch_size, dim_in)*, if e.g., the model has only one input layer, or
#'   \item a `list` with the corresponding input data (according to the
#'   upper point) for each of the input layers.
#' }
#' **Note:** For the model-agnostic methods, only models with a single
#' input and output layer is allowed!\cr
#' @param ... Other arguments passed to the individual constructor functions
#' of the methods R6 classes.
#'
#' @return [R6::R6Class] object of the respective type.
#'
NULL


#' @rdname innsight_sugar
#'
#' @usage
#' # Create a new `Converter` object of the given `model`
#' convert(model, ...)
#'
#' @export
convert <- function(model, ...) {
  Converter$new(model, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `Gradient` method to the passed `data` to be explained
#' run_grad(converter, data, ...)
#'
#' @export
run_grad <- function(converter, data, ...) {
  Gradient$new(converter, data, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `SmoothGrad` method to the passed `data` to be explained
#' run_smoothgrad(converter, data, ...)
#'
#' @export
run_smoothgrad <- function(converter, data, ...) {
  SmoothGrad$new(converter, data, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `IntegratedGradient` method to the passed `data` to be explained
#' run_intgrad(converter, data, ...)
#'
#' @export
run_intgrad <- function(converter, data, ...) {
  IntegratedGradient$new(converter, data, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `ExpectedGradient` method to the passed `data` to be explained
#' run_expgrad(converter, data, ...)
#'
#' @export
run_expgrad <- function(converter, data, ...) {
  ExpectedGradient$new(converter, data, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `LRP` method to the passed `data` to be explained
#' run_lrp(converter, data, ...)
#'
#' @export
run_lrp <- function(converter, data, ...) {
  LRP$new(converter, data, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `DeepLift` method to the passed `data` to be explained
#' run_deeplift(converter, data, ...)
#'
#' @export
run_deeplift <- function(converter, data, ...) {
  DeepLift$new(converter, data, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `DeepSHAP` method to the passed `data` to be explained
#' run_deepshap(converter, data, ...)
#'
#' @export
run_deepshap <- function(converter, data, ...) {
  DeepSHAP$new(converter, data, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `ConnectionWeights` method (argument `data` is not always required)
#' run_cw(converter, ...)
#'
#' @export
run_cw <- function(converter, ...) {
  ConnectionWeights$new(converter, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `LIME` method to explain `data` by using the dataset `data_ref`
#' run_lime(model, data, data_ref, ...)
#'
#' @export
run_lime <- function(model, data, data_ref, ...) {
  LIME$new(model, data, data_ref, ...)
}

#' @rdname innsight_sugar
#'
#' @usage
#' # Apply the `SHAP` method to explain `data` by using the dataset `data_ref`
#' run_shap(model, data, data_ref, ...)
#'
#' @export
run_shap <- function(model, data, data_ref, ...) {
  SHAP$new(model, data, data_ref, ...)
}
