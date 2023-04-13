#' @param ref_data_idx (`integer(1)` or `NULL`)\cr
#' This integer number determines the index for the
#' reference data point. In addition to the boxplots, it is displayed in
#' red color and is used to compare an individual result with the summary
#' statistics provided by the boxplot. With the default value (`NULL`),
#' no individual data point is plotted. This index can be chosen with
#' respect to all available data, even if only a subset is selected with
#' argument `data_idx`.\cr
#' *Note:* Because of the complexity of 2D inputs, this argument is used
#' only for tabular and 1D inputs and disregarded for 2D inputs.\cr
