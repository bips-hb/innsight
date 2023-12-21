#' @param pred_fun (`function`)\cr
#' Prediction function for the model. This argument is only
#' needed if `model` is not a model created by
#' \code{\link[torch]{nn_sequential}}, \code{\link[keras]{keras_model}},
#' \code{\link[neuralnet]{neuralnet}} or [`Converter`]. The first argument of
#' `pred_fun` has to be `newdata`, e.g.,
#' ```
#' function(newdata, ...) model(newdata)
#' ```
