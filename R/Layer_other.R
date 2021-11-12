
#' Torch implementation of a Flatten Layer
#'
#' This \code{torch::nn_module} implements a flatten layer. It takes an
#' input of dimension \emph{(batch_size, in_channels, dim_1, ..., dim_n)} and
#' flattens it to an output of dimensions \emph{(batch_size, in_channels *
#' dim_in * ... * features_n)}. Note that in this package all data is
#' transformed into the data format `channels_first`, so before flattening the
#' transformation must be undone.
#'
#' @param dim_in The input dimensions of the flatten layer
#' @param dim_out The output dimensions of the flatten layer
#'
#' @section Attributes:
#' \describe{
#'   \item{`self$input_dim`}{Dimension of the input without batch dimension}
#'   \item{`self$input`}{The last recorded input for this layer}
#'   \item{`self$input_ref`}{The last recorded reference input for this layer}
#'   \item{`self$output_dim`}{The dimension of the flattened input}
#'   \item{`self$output`}{The last recorded output of this layer, i.e. the
#'     flattened input}
#'   \item{`self$output_ref`}{The last recored reference output of this layer}
#'   \item{`self$channels_first`}{Boolean that determines whether to unroll
#'     values beginning at the last layer with \code{TRUE} or at the first
#'     layer with \code{FALSE}}
#' }
#'
#' @noRd
#'
flatten_layer <- nn_module(
  classname = "Flatten_Layer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
  channels_first = NULL,
  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$channels_first <- TRUE
  },

  #' @section `self$forward()`:
  #' This function takes the input and forwards it through the layer,
  #' updating the layer's output
  #'
  #' ## Usage
  #' `self(x, channels_first = TRUE)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x`}{The input of dimension \emph{(batch_size, in_channels,
  #'     dim_1, ..., dim_n)}}
  #'   \item{`channels_first`}{Boolean that determines whether the data
  #'     format is `channels_first` or `channels_last` (default: `TRUE`)}
  #' }
  #'
  #' ## Return
  #' Returns the output of the forward pass, of dimensions \emph{(batch_size,
  #' in_channels * dim_1 * ... * dim_n)}
  #'
  forward = function(x, channels_first = TRUE, save_input = TRUE,
                     save_output = TRUE) {
    if (save_input) {
      self$input <- x
    }
    if (channels_first == FALSE) {
      x <- torch_movedim(x, 2, -1)
      self$channels_first <- FALSE
    } else {
      self$channels_first <- TRUE
    }
    output <- torch_flatten(x, start_dim = 2)
    if (save_output) {
      self$output <- output
    }

    output
  },

  #' @section `self$update_ref()`:
  #' This function updates the reference input and forwards it through the
  #' layer, updating the output.
  #'
  #' ## Usage
  #' `self$update_ref(x_ref, channels_first = TRUE)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x`}{The reference input to be used of shape \emph{(1,
  #'     in_channels, dim_1, ..., dim_n)}}
  #'   \item{`channels_first`}{Boolean that determines whether the data
  #'     format is `channels_first` or `channels_last` (default: `TRUE`)}
  #' }
  #'
  #' ## Return
  #' Returns the output of the forward pass, of dimensions \emph{(1,
  #' in_channels * dim_1 * ... * dim_n)}
  #'
  update_ref = function(x_ref, channels_first = TRUE, save_input = TRUE,
                        save_output = TRUE) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    if (channels_first == FALSE) {
      x_ref <- torch_movedim(x_ref, 2, -1)
    }
    output_ref <- torch_flatten(x_ref, start_dim = 2)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  # Arguments:
  #   output  : relevance score from the upper layer to the output, torchTensor
  #           : of size [batch_size, dim_out,  model_out]
  #
  #   input   : torch Tensor of size [batch_size, in_channels, * , model_out]
  #

  #' @section `self$reshape_to_input()`:
  #' Reshape the output of the flatten layer to the input dimensions with an
  #' additional dimension at the end.
  #'
  #' ## Usage
  #' `self$reshape_to_input(output)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`output`}{Torch tensor of size \emph{(batch_size, in_channels
  #'     * dim_1 * ... * dim_n, model_out)}}
  #' }
  #'
  #' ## Return
  #' Reshapes the torch tensor `output` to the input dimension with the
  #' `model_out` axis \emph{(batch_size, in_channels, dim_1, ..., dim_n,
  #' model_out)}
  #'
  reshape_to_input = function(output) {
    batch_size <- dim(output)[1]
    model_out <- rev(dim(output))[1]

    if (self$channels_first == FALSE) {
      in_channels <- self$input_dim[1]
      in_dim <- c(self$input_dim[-1], in_channels)
      input <- output$reshape(c(batch_size, in_dim, model_out))
      input <- torch_movedim(input, length(self$input_dim) + 1, 2)
    } else {
      input <- output$reshape(c(batch_size, self$input_dim, model_out))
    }
    input
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
  }
)
