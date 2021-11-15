#' Get the Insight of your Neural Network
#'
#' `innsight` is an R package that interprets the behavior and explains
#' individual predictions of modern neural networks. Many methods for
#' explaining individual predictions already exist, but hardly any of them
#' are implemented or available in R. Most of these so-called
#' *'Feature Attribution'* methods are only implemented in Python and
#' thus difficult to access or use for the R community. In this sense,
#' the package `innsight` provides a common interface for various methods
#' for the interpretability of neural networks and can therefore be considered
#' as an R analogue to 'iNNvestigate' for Python.
#'
#' This package implements several model-specific interpretability
#' (Feature Attribution) methods based on neural networks in R, e.g.,
#'
#' * Layer-wise Relevance Propagation ([LRP])
#'   * Including propagation rules: \eqn{\epsilon}-rule and
#'   \eqn{\alpha}-\eqn{\beta}-rule
#' * Deep Learning Important Features ([DeepLift])
#'   * Including propagation rules for non-linearities: rescale rule and
#'  reveal-cancel rule
#' * Gradient-based methods:
#'   * Vanilla [Gradient], including 'Gradient x Input'
#'   * Smoothed gradients ([SmoothGrad]), including 'SmoothGrad x Input'
#' * [ConnectionWeights]
#'
#' The package `innsight` aims to be as flexible as possible and independent of a
#' specific deep learning package in which the passed network has been learned.
#' Basically, a Neural Network of the libraries [torch::nn_sequential],
#' [keras::keras_model_sequential], [keras::keras_model] and
#' [neuralnet::neuralnet] can be passed to the main building block [Converter],
#' which converts and stores the passed model as a torch model
#' ([ConvertedModel]) with special insights needed for interpretation.
#' It is also possible to pass an arbitrary net in form of a named list
#' (see details in [Converter]).
#'
#'
"_PACKAGE"

#' @import R6
#' @import torch
#' @import ggplot2
#' @import checkmate
NULL
