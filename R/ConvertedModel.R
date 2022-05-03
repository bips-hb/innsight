
###############################################################################
#                           Converted Torch Model
###############################################################################


#' Converted torch-based model
#'
#' This class stores all layers converted to torch in a module which can be
#' used like the original model (but torch-based). In addition, it provides
#' other functions that are useful for interpreting individual predictions or
#' explaining the entire model. This model is part of the class [Converter]
#' and is the core for all the necessary calculations in the methods provided
#' in this package.
#'
#' @param modules_list A list of all accepted layers created by the 'Converter'
#' class during initialization.
#' @param dtype The data type for all the calculations and defined tensors. Use
#' either `'float'` for [torch::torch_float] or `'double'` for
#' [torch::torch_double].
#'
#' @section Public fields:
#' \describe{
#'   \item{`modules_list`}{A list of all accepted layers created by the
#'   'Converter' class during initialization.}
#'   \item{`dtype`}{The datatype for all the calculations and defined
#'   tensors. Either `'float'` for [torch::torch_float] or `'double'` for
#'   [torch::torch_double]}.
#' }
#'
ConvertedModel <- nn_module(
  classname = "ConvertedModel",
  modules_list = NULL,
  dtype = NULL,
  initialize = function(modules_list, dtype = "float") {
    self$modules_list <- modules_list
    self$dtype <- dtype
  },

  ### -------------------------forward and update------------------------------
  #'
  #' @section Method `forward()`:
  #'
  #' The forward method of the whole model, i.e. it calculates the output
  #' \eqn{y=f(x)} of a given input \eqn{x}. In doing so, all intermediate
  #' values are stored in the individual torch modules from `modules_list`.
  #'
  #' ## Usage
  #' `self(x, channels_first = TRUE)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x`}{The input torch tensor of dimensions
  #'   \emph{(batch_size, dim_in)}.}
  #'   \item{`channels_first`}{If the input tensor `x` is given in the format
  #'   'channels first' use `TRUE`. Otherwise, if the channels are last,
  #'   use `FALSE` and the input will be transformed into the format 'channels
  #'   first'. Default: `TRUE`.}
  #' }
  #'
  #' ## Return
  #' Returns the output of the model with respect to the given inputs with
  #' dimensions \emph{(batch_size, dim_out)}.
  #'
  forward = function(x, channels_first = TRUE, save_input = FALSE,
                     save_preactivation = FALSE, save_output = FALSE,
                     save_last_layer = FALSE) {
    if (channels_first == FALSE) {
      x <- torch_movedim(x, -1, 2)
    }

    num_modules <- length(self$modules_list)
    i <- 1

    for (module in self$modules_list) {
      if (save_last_layer & i == num_modules) {
        save_preactivation <- TRUE
        save_output <- TRUE
      }
      if ("Flatten_Layer" %in% module$.classes) {
        x <- module(x, channels_first, save_input, save_output)
      } else {
        x <- module(x, save_input, save_preactivation, save_output)
      }

      i <- i + 1
    }
    x
  },

  #'
  #' @section Method `update_ref()`:
  #'
  #' This method updates the stored intermediate values in each module from the
  #' list `modules_list` when the reference input `x_ref` has changed.
  #'
  #' ## Usage
  #' `self$update_ref(x_ref, channels_first = TRUE)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x_ref`}{Reference input of the model of dimensions
  #'   \emph{(1, dim_in)}.}
  #'   \item{`channels_first`}{If the reference input tensor `x` is given in
  #'   the format 'channels first' use `TRUE`. Otherwise, if the channels are
  #'   last, use `FALSE` and the input will be transformed into the format
  #'   'channels first'. Default: `TRUE`.}
  #' }
  #'
  #' ## Return
  #' Returns the output of the reference input with dimension
  #' \emph{(1, dim_out)} after passing through the model.
  #'
  update_ref = function(x_ref, channels_first = TRUE, save_input = FALSE,
                        save_preactivation = FALSE, save_output = FALSE,
                        save_last_preactivation = FALSE) {
    if (channels_first == FALSE) {
      x_ref <- torch_movedim(x_ref, -1, 2)
    }

    num_modules <- length(self$modules_list)
    i <- 1

    for (module in self$modules_list) {
      if (save_last_preactivation & i == num_modules) {
        save_preactivation <- TRUE
      }
      if ("Flatten_Layer" %in% module$.classes) {
        x_ref <- module(x_ref, channels_first, save_input, save_output)
      } else {
        x_ref <- module$update_ref(x_ref, save_input, save_preactivation,
                                   save_output)
      }
      i <- i + 1
    }
    x_ref
  },

  #'
  #' @section Method `set_dtype()`:
  #'
  #' This method changes the data type for all the layers in `modules_list`.
  #' Use either `'float'` for [torch::torch_float] or `'double'` for
  #' [torch::torch_double].
  #'
  #' ## Usage
  #' `self$set_dtype(dtype)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`dtype`}{The data type for all the calculations and defined
  #'   tensors.}
  #' }
  #'
  set_dtype = function(dtype) {
    for (module in self$modules_list) {
      if (!("Flatten_Layer" %in% module$.classes)) {
        module$set_dtype(dtype)
      }
    }
    self$dtype <- dtype
  },

  reset = function() {
    for (module in self$modules_list) {
      module$reset()
    }
  }
)
