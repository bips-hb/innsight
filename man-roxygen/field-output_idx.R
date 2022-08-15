#' @field output_idx This list of indices specifies the output nodes to which
#' the method is to be applied. In the order of the output layers, the list
#' contains the respective output nodes indices and unwanted output layers
#' have the entry `NULL` instead of a vector of indices,
#' e.g. `list(NULL, c(1,3))` for the first and third output node in the
#' second output layer.
