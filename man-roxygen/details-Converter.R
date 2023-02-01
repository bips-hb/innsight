#' @details
#'
#' In order to better understand and analyze the prediction of a neural
#' network, the preactivation or other information of the individual layers,
#' which are not stored in an ordinary forward pass, are often required. For
#' this reason, a given neural network is converted into a torch-based neural
#' network, which provides all the necessary information for an interpretation.
#' The converted torch model is stored in the field `model` and is an instance
#' of \code{\link[innsight:ConvertedModel]{ConvertedModel}}.
#' However, before the torch model is created, all relevant details of the
#' passed model are extracted into a named list. This list can be saved
#' in complete form in the `model_as_list` field with the argument
#' `save_model_as_list`, but this may consume a lot of memory for large
#' networks and is not done by default. Also, this named list can again be
#' used as a passed model for the class `Converter`, which will be described
#' in more detail in the section 'Implemented Libraries'.
#'
#' ## Implemented Methods
#'
#' An object of the Converter class can be applied to the
#' following methods:
#'   * *Layerwise Relevance Propagation* ([LRP]), Bach et al. (2015)
#'   * *Deep Learning Important Features* ([DeepLift]), Shrikumar et al. (2017)
#'   * *[SmoothGrad]* including *SmoothGrad x Input*, Smilkov et al. (2017)
#'   * *Vanilla [Gradient]* including *Gradient x Input*
#'   * *[ConnectionWeights]*, Olden et al. (2004)
#'
#'
#' ## Implemented Libraries
#' The converter is implemented for models from the libraries
#' \code{\link[torch]{nn_sequential}},
#' \code{\link[neuralnet]{neuralnet}} and \code{\link[keras]{keras}}. But you
#' can also write a wrapper for other libraries because a model can be passed
#' as a named list which is described in detail in the vignette "In-depth
#' Explanation"
#' (see \code{vignette("detailed_overview", package = "innsight")} or the
#' [website](https://bips-hb.github.io/innsight/vignette/detailed_overview.html#model-as-named-list)).
#'
