#' Get the insight of your neural network
#'
#' `innsight` is an R package that interprets the behavior and explains
#' individual predictions of modern neural networks. Many methods for
#' explaining individual predictions already exist, but hardly any of them
#' are implemented or available in R. Most of these so-called
#' *feature attribution* methods are only implemented in Python and,
#' thus, difficult to access or use for the R community. In this sense,
#' the package `innsight` provides a common interface for various methods
#' for the interpretability of neural networks and can therefore be considered
#' as an R analogue to 'iNNvestigate' or 'Captum' for Python.
#'
#' This package implements several model-specific interpretability
#' (feature attribution) methods based on neural networks in R, e.g.,
#'
#' * *Layer-wise relevance propagation ([LRP])*
#'   * Including propagation rules: \eqn{\epsilon}-rule and
#'   \eqn{\alpha}-\eqn{\beta}-rule
#' * *Deep learning important features ([DeepLift])*
#'   * Including propagation rules for non-linearities: *Rescale* rule and
#'  *RevealCancel* rule
#'  * [DeepSHAP]
#' * Gradient-based methods:
#'   * *Vanilla [Gradient]*, including *Gradient\eqn{\times}Input*
#'   * Smoothed gradients *([SmoothGrad])*, including *SmoothGrad\eqn{\times}Input*
#'   * *Integrated gradients* ([IntegratedGradient])
#'   * *Expected gradients* ([ExpectedGradient])
#' * *[ConnectionWeights]*
#' * Model-agnostic methods:
#'   * *Local interpretable model-agnostic explanation ([LIME])*
#'   * *Shapley values* ([SHAP])
#'
#' The package `innsight` aims to be as flexible as possible and independent
#' of a specific deep learning package in which the passed network has been
#' learned. Basically, a neural network of the libraries
#' [`torch::nn_sequential`], [`keras::keras_model_sequential`],
#' [`keras::keras_model`] and [`neuralnet::neuralnet`] can be passed to the
#' main building block [`Converter`],
#' which converts and stores the passed model as a torch model
#' ([`ConvertedModel`]) with special insights needed for interpretation.
#' It is also possible to pass an arbitrary net in form of a named list
#' (see details in [`Converter`]).
#'
#' The scientific background and implementation details of `innsight` are
#' described in the paper "Interpreting Deep Neural Networks with the Package
#' innsight" by Koenen & Wright (2024), published in the
#' *Journal of Statistical Software*. For a detailed explanation of the
#' methods and use cases, please refer to the publication
#' (doi: [10.18637/jss.v111.i08](https://doi.org/10.18637/jss.v111.i08)).
#'
#' @references
#' Koenen, N., & Wright, M. N. (2024). Interpreting Deep Neural Networks with
#' the Package innsight. Journal of Statistical Software, 111(8), 1-52. doi:
#' [10.18637/jss.v111.i08](https://doi.org/10.18637/jss.v111.i08)
#'
"_PACKAGE"

#' @import R6
#' @import torch
#' @import ggplot2
#' @import checkmate
#' @importFrom cli cli_h1 cli_h2 cli_text cli_ul cli_li cli_end col_grey
#' @importFrom cli cli_dl symbol cli_ol cli_div cli_bullets col_cyan
#' @importFrom cli cli_progress_bar cli_progress_update cli_progress_done
#' @importFrom stats predict
NULL
