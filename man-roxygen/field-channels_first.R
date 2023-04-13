#' @field channels_first  (`logical(1)`)\cr
#' The channel position of the given data. If `TRUE`, the
#' channel axis is placed at the second position between the batch size and
#' the rest of the input axes, e.g., `c(10,3,32,32)` for a batch of ten images
#' with three channels and a height and width of 32 pixels. Otherwise (`FALSE`),
#' the channel axis is at the last position, i.e., `c(10,32,32,3)`. This is
#' especially important for layers like flatten, where the order is crucial
#' and therefore the channels have to be moved from the internal
#' format "channels first" back to the original format before the layer
#' is calculated.\cr
