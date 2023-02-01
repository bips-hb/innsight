#' @param data ([`array`], [`data.frame`], [`torch_tensor`] or `list`)\cr
#' The data to which the method is to be applied. These must
#' have the same format as the input data of the passed model to the
#' converter object. This means either
#' \itemize{
#'   \item an `array`, `data.frame`, `torch_tensor` or array-like format of
#'   size *(batch_size, dim_in)*, if e.g.the model has only one input layer, or
#'   \item a `list` with the corresponding input data (according to the
#'   upper point) for each of the input layers.\cr
#' }
