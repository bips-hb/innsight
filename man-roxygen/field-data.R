#' @field data (`list`)\cr
#' The passed data as a `list` of `torch_tensors` in the selected
#' data format (field `dtype`) matching the corresponding shapes of the
#' individual input layers. Besides, the channel axis is moved to the
#' second position after the batch size because internally only the
#' format *channels first* is used.\cr
