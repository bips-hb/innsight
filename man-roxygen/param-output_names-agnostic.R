#' @param output_names (`character`, `factor` )\cr
#' A character vector with the names for the output dimensions
#' excluding the batch dimension, e.g., for a model with 3 output nodes use
#' `c("Y1", "Y2", "Y3")`. Instead of a character
#' vector you can also use a factor to set an order for the plots.\cr
#' *Note:* This argument is optional and otherwise the names are
#' generated automatically. But if this argument is set, all found
#' output names in the passed model will be disregarded.\cr
