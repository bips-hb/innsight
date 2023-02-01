#' @field result (`list`)\cr
#' The results of the method on the passed data. A unified
#' list structure is used regardless of the complexity of the model: The outer
#' list contains the individual output layers and the inner list the input
#' layers. The results for the respective output and input layer are then
#' stored there as torch tensors in the given data format (field `dtype`).
#' In addition, the channel axis is moved to its original place and the last
#' axis contains the selected output nodes for the individual output layers
#' (see `output_idx`).\cr
#' For example, the structure of the result for two output
#' layers (output node 1 for the first and 2 and 4 for the second) and two
#' input layers with `channels_first = FALSE` looks like this:
#' ```
#' List of 2 # both output layers
#'   $ :List of 2 # both input layers
#'     ..$ : torch_tensor [batch_size, dim_in_1, channel_axis, 1]
#'     ..$ : torch_tensor [batch_size, dim_in_2, channel_axis, 1]
#'  $ :List of 2 # both input layers
#'     ..$ : torch_tensor [batch_size, dim_in_1, channel_axis, 2]
#'     ..$ : torch_tensor [batch_size, dim_in_2, channel_axis, 2]
#' ```
#'
