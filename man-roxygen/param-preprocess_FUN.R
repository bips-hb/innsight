#' @param preprocess_FUN (`function`)\cr
#' This function is applied to the method's result
#' before calculating the boxplots or medians. Since positive and negative values
#' often cancel each other out, the absolute value (`abs`) is used by
#' default. But you can also use the raw results (`identity`) to see the
#' results' orientation, the squared data (`function(x) x^2`) to weight
#' the outliers higher or any other function.\cr
