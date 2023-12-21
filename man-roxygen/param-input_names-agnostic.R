#' @param input_names (`character`, `factor` or `list`)\cr
#' The input names of the model excluding the batch dimension. For a model
#' with a single input layer and input axis (e.g., for tabular data), the
#' input names can be specified as a character vector or factor, e.g.,
#' for a dense layer with 3 input features use `c("X1", "X2", "X3")`. If
#' the model input consists of multiple axes (e.g., for signal and
#' image data), use a list of character vectors or factors for each axis
#' in the format "channels first", e.g., use
#' `list(c("C1", "C2"), c("L1","L2","L3","L4","L5"))` for a 1D
#' convolutional input layer with signal length 4 and 2 channels.\cr
#' *Note:* This argument is optional and otherwise the names are
#' generated automatically. But if this argument is set, all found
#' input names in the passed model will be disregarded.\cr
