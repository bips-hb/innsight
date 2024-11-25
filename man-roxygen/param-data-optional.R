#' @param data ([`array`], [`data.frame`], \code{\link[torch]{torch_tensor}} or `list`)\cr
#' The data to which the method is to be applied. These must
#' have the same format as the input data of the passed model to the
#' converter object. This means either
#' \itemize{
#'   \item an `array`, `data.frame`, `torch_tensor` or array-like format of
#'   size *(batch_size, dim_in)*, if e.g.the model has only one input layer, or
#'   \item a `list` with the corresponding input data (according to the
#'   upper point) for each of the input layers.
#' }
#' This argument is only relevant if
#' `times_input` is `TRUE`, otherwise it will be ignored because it is a
#' locale (i.e. explanation for each data point individually) method only
#' in this case.\cr
