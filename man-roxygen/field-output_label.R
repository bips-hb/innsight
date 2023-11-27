#' @field output_label (`list`)\cr
#' This list of `factors` specifies the output nodes to which
#' the method is to be applied. In the order of the output layers, the list
#' contains the respective output nodes labels and unwanted output layers
#' have the entry `NULL` instead of a vector of labels,
#' e.g., `list(NULL, c("a", "c"))` for the first and third output node in the
#' second output layer.\cr
