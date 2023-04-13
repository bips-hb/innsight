#' @param output_idx (`integer`, `list` or `NULL`)\cr
#' These indices specify the output nodes for which
#' the method is to be applied. In order to allow models with multiple
#' output layers, there are the following possibilities to select
#' the indices of the output nodes in the individual output layers:
#' \itemize{
#'   \item An `integer` vector of indices: If the model has only one output
#'   layer, the values correspond to the indices of the output nodes, e.g.,
#'   `c(1,3,4)` for the first, third and fourth output node. If there are
#'   multiple output layers, the indices of the output nodes from the first
#'   output layer are considered.
#'   \item A `list` of `integer` vectors of indices: If the method is to be
#'   applied to output nodes from different layers, a list can be passed
#'   that specifies the desired indices of the output nodes for each
#'   output layer. Unwanted output layers have the entry `NULL` instead of
#'   a vector of indices, e.g., `list(NULL, c(1,3))` for the first and
#'   third output node in the second output layer.
#'   \item `NULL` (default): The method is applied to all output nodes in
#'   the first output layer but is limited to the first ten as the
#'   calculations become more computationally expensive for more output
#'   nodes.\cr
#' }
