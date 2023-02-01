#' @param channels_first (`logical(1)`)\cr
#' The channel position of the given data (argument
#' `data`). If `TRUE`, the channel axis is placed at the second position
#' between the batch size and the rest of the input axes, e.g.
#' `c(10,3,32,32)` for a batch of ten images with three channels and a
#' height and width of 32 pixels. Otherwise (`FALSE`), the channel axis
#' is at the last position, i.e. `c(10,32,32,3)`. If the data
#' has no channel axis, use the default value `TRUE`.\cr
