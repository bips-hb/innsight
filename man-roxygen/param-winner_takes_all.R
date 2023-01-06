#' @param winner_takes_all This logical argument is only relevant for
#' models with a MaxPooling layer. Since many zeros are produced during
#' the backward pass due to the selection of the maximum value in the
#' pooling kernel, another variant is implemented, which treats a
#' MaxPooling as an AveragePooling layer in the backward pass to overcome
#' the problem of too many zero relevances. With the default value `TRUE`,
#' the whole upper-layer relevance is passed to the maximum value in each
#' pooling window. Otherwise, if `FALSE`, the relevance is distributed equally
#' among all nodes in a pooling window.
