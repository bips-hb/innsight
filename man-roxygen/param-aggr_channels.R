#' @param aggr_channels Pass one of `'norm'`, `'sum'`, `'mean'` or a
#' custom function to aggregate the channels, e.g. the maximum
#' ([base::max]) or minimum ([base::min]) over the channels or only
#' individual channels with `function(x) x[1]`. By default (`'sum'`),
#' the sum of all channels is used.\cr
#' *Note:* This argument is used only for 2D and 3D input data.
