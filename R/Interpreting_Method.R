
#' @title Superclass for interpreting methods
#' @description This is a superclass for all data-based interpreting methods.
#' Implemented are the following methods:
#'
#' - Deep Learning Important Features ([DeepLift])
#' - Layer-wise Relevance Propagation ([LRP])
#' - Gradient-based methods:
#'    - Normal gradients ([Gradient])
#'    - Smoothed Gradients ([SmoothGrad])
#'
#'
#' @field data The given data as a torch tensor to be interpreted with the
#' selected method.
#' @field analyzer The analyzer with the stored and torch-converted model.
#' @field dtype The type of the data (either `'float'` or `'double'`).
#' @field channels_first The format of the given date, i.e. channels on
#' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
#' data has no channels, use the default value `TRUE`.
#' @field ignore_last_act A boolean value to include the last
#' activation into all the calculations, or not. In some cases, the last activation
#' leads to a saturation problem.
#' @field result The methods result of the given data as a
#' torch tensor of size (batch_size, model_in, model_out).

Interpreting_Method <- R6::R6Class(
  classname = "Interpreting_Method",
  public = list(

    data = NULL,
    analyzer = NULL,
    channels_first = NULL,
    dtype = NULL,

    ignore_last_act = NULL,

    result = NULL,

    #' @description
    #' Create a new instance of this class.
    #'
    #' @param analyzer The analyzer with the stored and torch-converted model.
    #' @param data The given data as a torch tensor to be interpreted with the
    #' selected method.
    #' @param channels_first The format of the given data, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`). If the
    #' data has no channels, use the default value `TRUE`.
    #' @param dtype The type of the data (either `'float'` or `'double'`).
    #' @param ignore_last_act A boolean value to include the last
    #' activation into all the calculations, or not. In some cases, the last activation
    #' leads to a saturation problem.

    initialize = function(analyzer, data,
                          channels_first = TRUE,
                          dtype = 'float',
                          ignore_last_act = TRUE) {

      checkmate::assertClass(analyzer, "Analyzer")
      self$analyzer <- analyzer

      checkmate::assert_logical(channels_first)
      self$channels_first <- channels_first

      checkmate::assert_logical(ignore_last_act)
      self$ignore_last_act <- ignore_last_act

      checkmate::assertChoice(dtype, c('float', 'double'))
      self$dtype <- dtype
      self$analyzer$model$set_dtype(dtype)

      self$data <- private$test_data(data)
    },

    #'
    #' @description
    #' This function returns the result of this method for the given data either as an
    #' array (`as_torch = FALSE`) or a torch tensor (`as_torch = TRUE`) of
    #' size (batch_size, model_in, model_out).
    #'
    #' @param as_torch Boolean value whether the output is an array or a torch
    #' tensor.
    #'
    #' @return The result of this method for the given data.
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
    test_data = function(data, name = 'data'){
      data <- tryCatch({
        if (is.data.frame(data)) {
          data <- as.matrix(data)
        }
        as.array(data)
      },
      error=function(e) stop(sprintf("Failed to convert the argument '%s' to an array using the function 'base::as.array'. The class of your '%s': %s", name, name, class(data))))

      ordered_dim <- self$analyzer$input_dim
      if (!self$channels_first) {
        channels <- ordered_dim[1]
        ordered_dim <- c(ordered_dim[-1], channels)
      }

      if (length(dim(data)[-1]) != length(ordered_dim) || !all(dim(data)[-1] == ordered_dim)) {
        stop(sprintf("Unmatch in model dimension (*,%s) and dimension of argument '%s' (%s). Try to change the argument 'channels_first', if only the channels are wrong.",
                     paste0(ordered_dim, sep = '', collapse = ','),
                     name,
                     paste0(dim(data), sep = '', collapse = ',')))
      }


      if (self$dtype == "float") {
        data <- torch::torch_tensor(data,
                                    dtype = torch::torch_float())
      }
      else {
        data <- torch::torch_tensor(data,
                                    dtype = torch::torch_double())
      }

      data
    }
  )
)
