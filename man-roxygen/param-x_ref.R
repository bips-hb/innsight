#' @param x_ref ([`array`], [`data.frame`], [`torch_tensor`] or `list`)\cr
#' The reference input for the DeepLift method. This value
#' must have the same format as the input data of the passed model to the
#' converter object. This means either
#' - an `array`, `data.frame`, `torch_tensor` or array-like format of
#' size *(1, dim_in)*, if e.g., the model has only one input layer, or
#' - a `list` with the corresponding input data (according to the upper point)
#' for each of the input layers.
#' - It is also possible to use the default value `NULL` to take only
#' zeros as reference input.\cr
