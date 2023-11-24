#' @param model \cr
#' A fitted model for a classification or regression task that
#' is intended to be interpreted. A [`Converter`] object can also be
#' passed. In order for the package to know how to make predictions
#' with the given model, a prediction function must also be passed with
#' the argument `pred_fun`. However, for models created by
#' \code{\link[torch]{nn_sequential}}, \code{\link[keras]{keras_model}},
#' \code{\link[neuralnet]{neuralnet}} or [`Converter`],
#' these have already been pre-implemented and do not need to be
#' specified.\cr
