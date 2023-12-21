#' @param data (`array`, `data.frame` or `torch_tensor`)\cr
#' The individual instances to be explained by the method.
#' These must have the same format as the input data of the passed model
#' and has to be either [`matrix`], an [`array`], a [`data.frame`] or a
#' [`torch_tensor`]. If no value is specified, all instances in the
#' dataset `data` will be explained.\cr
#' **Note:** For the model-agnostic methods, only models with a single
#' input and output layer is allowed!\cr
