#' @param output_label (`character`, `factor`, `list` or `NULL`)\cr
#' These values specify the output nodes for which
#' the method is to be applied. Only values that were previously passed with
#' the argument `output_names` in the `converter` can be used. In order to
#' allow models with multiple
#' output layers, there are the following possibilities to select
#' the names of the output nodes in the individual output layers:
#' \itemize{
#'   \item A `character` vector or `factor` of labels: If the model has only one output
#'   layer, the values correspond to the labels of the output nodes named in the
#'   passed `Converter` object, e.g.,
#'   `c("a", "c", "d")` for the first, third and fourth output node if the
#'   output names are `c("a", "b", "c", "d")`. If there are
#'   multiple output layers, the names of the output nodes from the first
#'   output layer are considered.
#'   \item A `list` of `charactor`/`factor` vectors of labels: If the method is to be
#'   applied to output nodes from different layers, a list can be passed
#'   that specifies the desired labels of the output nodes for each
#'   output layer. Unwanted output layers have the entry `NULL` instead of
#'   a vector of labels, e.g., `list(NULL, c("a", "c"))` for the first and
#'   third output node in the second output layer.
#'   \item `NULL` (default): The method is applied to all output nodes in
#'   the first output layer but is limited to the first ten as the
#'   calculations become more computationally expensive for more output
#'   nodes.\cr
#' }
