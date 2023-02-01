#' @param individual_data_idx (`integer` or `NULL`)\cr
#' Only relevant for a `plotly` plot with tabular
#' or 1D inputs! This integer vector of data indices determines
#' the available data points in a dropdown menu, which are drawn in
#' individually analogous to `ref_data_idx` only for more data points.
#' With the default value `NULL` the first `individual_max` data points
#' are used.\cr
#' *Note:* If `ref_data_idx` is specified, this data point will be
#' added to those from `individual_data_idx` in the dropdown menu.\cr
