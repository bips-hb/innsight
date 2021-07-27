
#' @title Superclass for interpreting methods
#' @description This is a superclass for all data-based interpreting methods.
#'

Interpreting_Method <- R6::R6Class(
  classname = "Interpreting_Method",
  public = list(

    data = NULL,
    analyzer = NULL,
    channels_first = NULL,
    dtype = NULL,

    result = NULL,

    initialize = function(analyzer, data,
                          channels_first = TRUE,
                          dtype = 'float') {

      checkmate::assertClass(analyzer, "Analyzer")
      self$analyzer <- analyzer

      checkmate::assert_logical(channels_first)
      self$channels_first <- channels_first

      checkmate::assertChoice(dtype, c('float', 'double'))
      self$dtype <- dtype
      self$analyzer$model$set_dtype(dtype)

      data <- tryCatch({
        if (is.data.frame(data)) {
          data <- as.matrix(data)
        }
        as.array(data)
      },
      error=function(e) stop(sprintf("Failed to convert the argument 'data' to an array using the function 'base::as.array'. The class of your 'data': %s", class(data))))

      ordered_dim <- self$analyzer$input_dim
      if (!self$channels_first) {
        channels <- ordered_dim[1]
        ordered_dim <- c(ordered_dim[-1], channels)
      }

      if (length(dim(data)[-1]) != length(ordered_dim) || !all(dim(data)[-1] == ordered_dim)) {
        stop(sprintf("Unmatch in model dimension (*,%s) and dimension of argument 'data' (%s). Try to change the argument 'channels_first', if only the channels are wrong.",
                     paste0(ordered_dim, sep = '', collapse = ','),
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

      self$data <- data
    }
  )
)
