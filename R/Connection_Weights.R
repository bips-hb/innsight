

#' Connection Weights method
#'
#' @description
#' This function implements the \emph{Connection Weight} method investigated by
#' Olden et al. (2004) which results in a feature relevance score for each input
#' variable. The basic idea is to multiply up all path weights for each
#' possible connection between an input feature and the output and then
#' calculate the sum over them. Besides, it is a global interpretition method and
#' independent of the input data. For a neural network with \eqn{3} hidden layers with weight
#' matrices \eqn{W_1}, \eqn{W_2} and \eqn{W_3} this method results in a simple
#' matrix multiplication
#' \deqn{W_1 * W_2 * W_3. }
#'
#'
#' @field analyzer The analyzer with the stored and torch-converted model.
#' @field channels_first The data format of the result, i.e. channels on
#' last dimension (`FALSE`) or on the first dimension (`TRUE`). If the
#' data has no channels, use the default value `TRUE`.
#' @field dtype The type of the data and parameters (either `'float'` or `'double'`).
#' @field result The methods result as a torch tensor of size (model_in, model_out).
#'
#' @export
Connection_Weights <- R6::R6Class(
  classname = 'Connection_Weights',
  public = list(

    analyzer = NULL,
    channels_first = NULL,
    dtype = NULL,

    result = NULL,

    #' @param analyzer The analyzer with the stored and torch-converted model.
    #' @param channels_first The data format of the result, i.e. channels on
    #' last dimension (`FALSE`) or on the first dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data and parameters (either `'float'` or `'double'`).
    #'
    initialize = function(analyzer,
                          channels_first = TRUE,
                          dtype = 'float') {

      checkmate::assertClass(analyzer, "Analyzer")
      self$analyzer <- analyzer

      checkmate::assert_logical(channels_first)
      self$channels_first <- channels_first

      checkmate::assertChoice(dtype, c('float', 'double'))
      self$dtype <- dtype
      self$analyzer$model$set_dtype(dtype)

      self$result <- private$run()
    },

    #'
    #' @description
    #' This function returns the result of the Connection Weights method either as an
    #' array (`as_torch = FALSE`) or a torch tensor (`as_torch = TRUE`) of
    #' size (model_in, model_out).
    #'
    #' @param as_torch Boolean value whether the output is an array or a torch
    #' tensor.
    #'
    #' @return The result of this method.
    #'

    get_result = function(as_torch = FALSE) {
      result <- self$result
      if (!as_torch) {
        result <- as.array(result)
      }

      result
    }
  ),

  private = list(
    run = function(){
      if (self$dtype == "double") {
        grad <-
          torch::torch_tensor(diag(self$analyzer$output_dim),
                              dtype = torch::torch_double())$unsqueeze(1)
      }
      else {
        grad <-
          torch::torch_tensor(diag(self$analyzer$output_dim),
                              dtype = torch::torch_float())$unsqueeze(1)
      }

      for (layer in rev(self$analyzer$model$modules_list)) {
        if("Flatten_Layer" %in% layer$'.classes') {
          grad <- layer$reshape_to_input(grad)
        }
        else{
          grad <- layer$get_gradient(grad, layer$W)
        }
      }
      if (!self$channels_first) {
        grad <- torch::torch_movedim(grad, 2,length(dim(grad)) - 1)
      }

      grad$squeeze(1)
    }
  )
)
