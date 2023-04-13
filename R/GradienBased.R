###############################################################################
#                         Super class: GradientBased
###############################################################################

#'
#' @title Super class for gradient-based interpretation methods
#' @description Super class for gradient-based interpretation methods. This
#' class inherits from [`InterpretingMethod`]. It summarizes all implemented
#' gradient-based methods and provides a private function to calculate the
#' gradients w.r.t. to the input for given data. Implemented are:
#'
#' - *Vanilla Gradients* and *Gradient\eqn{\times}Input* ([`Gradient`])
#' - *SmoothGrad* and *SmoothGrad\eqn{\times}Input* ([`SmoothGrad`])
#'
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-dtype
#' @template param-ignore_last_act
#' @template param-output_idx
#' @template param-verbose
#'
GradientBased <- R6Class(
  classname = "GradientBased",
  inherit = InterpretingMethod,
  public = list(

    #' @field times_input (`logical(1`))\cr
    #' This logical value indicates whether the results
    #' were multiplied by the provided input data or not.\cr
    times_input = NULL,

    #' @description
    #' Create a new instance of this class. When initialized,
    #' the method is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param times_input (`logical(1`)\cr
    #' Multiplies the gradients with the input features.
    #' This method is called *Gradient\eqn{\times}Input*.\cr
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = TRUE,
                          verbose = interactive(),
                          dtype = "float"
                          ) {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, TRUE, verbose, dtype)

      cli_check(checkLogical(times_input), "times_input")
      self$times_input <- times_input
    }
  ),
  private = list(
    calculate_gradients = function(input, method_name = "Gradient") {
      # Set 'requires_grad' for the input tensors
      lapply(input, function(i) i$requires_grad <- TRUE)

      # Run input through the model
      out <- self$converter$model(input,
                                  channels_first = self$channels_first,
                                  save_input = FALSE,
                                  save_preactivation = FALSE,
                                  save_output = FALSE,
                                  save_last_layer = TRUE)


      if (self$ignore_last_act) {
        out <- lapply(self$converter$model$output_nodes,
                      function(x) self$converter$model$modules_list[[x]]$preactivation)
      }

      # Add up the output over the batch dimension
      out_sum <- lapply(out, torch_sum, dim = 1)

      if (self$verbose) {
        # Define Progressbar
        pb <- cli_progress_bar(name = paste0("Backward pass '", method_name, "'"),
                         total = length(unlist(self$output_idx)),
                         type = "iterator",
                         clear = FALSE)
      }

      # Definition of some temporary functions --------------------------------
      # Define function for calculating the gradients of one output
      calc_gradient_for_one_output <- function(idx, list_idx) {
        if (self$verbose) {
          cli_progress_update(id = pb, inc = 1, force = TRUE)
        }

        autograd_grad(out_sum[[list_idx]][idx], input, retain_graph = TRUE,
                      allow_unused = TRUE)
      }

      # Define function for stacking the results
      stack_outputs <- function(input_idx, results) {
        batch <- lapply(seq_along(results), function(i) {
          result <- results[[i]][[input_idx]]
          if (is_undefined_tensor(result)) result <- NULL

          result
        })

        if (is.null(batch[[1]])) {
          result <- NULL
        } else {
          result <- torch_stack(batch, dim = -1)
        }

        result
      }

      # Define function for calculating gradients for multiple outputs
      calc_gradient_for_list_idx <- function(list_idx) {
        # Loop over every entry for the model output of index 'list_idx'
        res <- lapply(self$output_idx[[list_idx]],
                      calc_gradient_for_one_output,
                      list_idx = list_idx)

        lapply(seq_along(res[[1]]), stack_outputs, results = res)
      }
      # End of definitions ----------------------------------------------------

      output_idx <-
        seq_along(self$output_idx)[!unlist(lapply(self$output_idx, is.null))]
      grads <- lapply(output_idx, calc_gradient_for_list_idx)

      lapply(input, function(i) i$requires_grad <- FALSE)
      if (self$verbose) cli_progress_done(id = pb)

      grads
    }
  )
)


#'
#' @importFrom graphics boxplot
#' @exportS3Method
#'
boxplot.GradientBased <- function(x, ...) {
  x$boxplot(...)
}


###############################################################################
#                               Vanilla Gradient
###############################################################################

#' @title Vanilla Gradient and Gradient\eqn{\times}Input
#' @name Gradient
#'
#' @description
#' This method computes the gradients (also known as *Vanilla Gradients*) of
#' the outputs with respect to the input variables, i.e., for all input
#' variable \eqn{i} and output class \eqn{j}
#' \deqn{d f(x)_j / d x_i.}
#' If the argument `times_input` is `TRUE`, the gradients are multiplied by
#' the respective input value (*Gradient\eqn{\times}Input*), i.e.,
#' \deqn{x_i * d f(x)_j / d x_i.}
#'
#' @template examples-Gradient
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-dtype
#' @template param-output_idx
#' @template param-ignore_last_act
#' @template param-verbose
#'
#' @family methods
#' @export
Gradient <- R6Class(
  classname = "Gradient",
  inherit = GradientBased,
  public = list(

    #' @description
    #' Create a new instance of the `Gradient` R6 class. When initialized,
    #' the method *Gradient* or *Gradient\eqn{\times}Input* is applied to the
    #' given data and the results are stored in the field `result`.
    #'
    #' @param times_input (`logical(1`))\cr
    #' Multiplies the gradients with the input features.
    #' This method is called *Gradient\eqn{\times}Input*.\cr
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = FALSE,
                          verbose = interactive(),
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, times_input, verbose, dtype)

      self$result <- private$run()
      self$converter$model$reset()
    }
  ),
  private = list(
    run = function() {
      gradients <- private$calculate_gradients(self$data, "Gradient")

      if (self$times_input) {
        gradients <- calc_times_input(gradients, self$data)
      }

      gradients
    },

    print_method_specific = function() {
      i <- cli_ul()
      if (self$times_input) {
        cli_li(paste0("{.field times_input}:  TRUE (",
                      symbol$arrow_right,
                      " {.emph Gradient x Input} method)"))
      } else {
        cli_li(paste0("{.field times_input}:  FALSE (",
                      symbol$arrow_right,
                      " {.emph Gradient} method)"))
      }
      cli_end(id = i)
    }
  )
)


###############################################################################
#                                 SmoothGrad
###############################################################################

#' @title SmoothGrad and SmoothGrad\eqn{\times}Input
#'
#' @description
#' *SmoothGrad* was introduced by D. Smilkov et al. (2017) and is an extension
#' to the classical *Vanilla [Gradient]* method. It takes the mean of the
#' gradients for \code{n} perturbations of each data point, i.e., with
#' \eqn{\epsilon \sim N(0,\sigma)}
#' \deqn{1/n \sum_n d f(x+ \epsilon)_j / d x_j.}
#' Analogous to the *Gradient\eqn{\times}Input* method, you can also use the argument
#' `times_input` to multiply the gradients by the inputs before taking the
#' average (*SmoothGrad\eqn{\times}Input*).
#'
#' @template examples-SmoothGrad
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-dtype
#' @template param-output_idx
#' @template param-ignore_last_act
#' @template param-verbose
#'
#' @references
#' D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @family methods
#' @export
SmoothGrad <- R6Class(
  classname = "SmoothGrad",
  inherit = GradientBased,
  public = list(

    #' @field n (`integer(1)`)\cr
    #' Number of perturbations of the input data (default: \eqn{50}).\cr
    #' @field noise_level (`numeric(1)`)\cr
    #' The standard deviation of the Gaussian
    #' perturbation, i.e., \eqn{\sigma = (max(x) - min(x)) *} `noise_level`.\cr
    #'
    n = NULL,
    noise_level = NULL,

    #' @description
    #' Create a new instance of the `SmoothGrad` R6 class. When initialized,
    #' the method *SmoothGrad* or *SmoothGrad\eqn{\times}Input* is applied to
    #' the given data and the results are stored in the field `result`.
    #'
    #' @param times_input (`logical(1`)\cr
    #' Multiplies the gradients with the input features.
    #' This method is called *SmoothGrad\eqn{\times}Input*.\cr
    #' @param n (`integer(1)`)\cr
    #' Number of perturbations of the input data (default: \eqn{50}).\cr
    #' @param noise_level (`numeric(1)`)\cr
    #' Determines the standard deviation of the Gaussian
    #' perturbation, i.e., \eqn{\sigma = (max(x) - min(x)) *} `noise_level`.\cr
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = FALSE,
                          n = 50,
                          noise_level = 0.1,
                          verbose = interactive(),
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, times_input, verbose, dtype)

      cli_check(checkInt(n, lower = 1), "n")
      cli_check(checkNumber(noise_level, lower = 0), "noise_level")
      self$n <- n
      self$noise_level <- noise_level

      self$result <- private$run()
      self$converter$model$reset()
    }
  ),
  private = list(
    run = function() {
      tmp_fun <- function(input) {
        torch_repeat_interleave(
          input,
          repeats = torch_tensor(self$n, dtype = torch_long()),
          dim = 1)
        }
      data <- lapply(self$data, tmp_fun)

      data <- lapply(data, function(input) {
        noise_scale <- self$noise_level * (max(input) - min(input))
        if (noise_scale$item() == 0) {
          noise_scale <- self$noise_level
        }
        noise <- torch_randn_like(input) * noise_scale

        input + noise
      })

      gradients <- private$calculate_gradients(data, "SmoothGrad")

      if (self$times_input) {
        gradients <- calc_times_input(gradients, data)
      }

      smoothgrads <- lapply(
        gradients,
        function(grad_output) {
          lapply(
            grad_output,
            function(grad_input) {
              if (is.null(grad_input)) {
                res <- NULL
              } else {
                res <- torch_stack(
                  lapply(grad_input$chunk(dim(self$data[[1]])[1]),
                         FUN = torch_mean, dim = 1
                  ),
                  dim = 1
                )
              }
            })
        })

      smoothgrads
    },

    print_method_specific = function() {
      i <- cli_ul()
      if (self$times_input) {
        cli_li(paste0("{.field times_input}:  TRUE (",
                      symbol$arrow_right,
                      " {.emph SmoothGrad x Input} method)"))
      } else {
        cli_li(paste0("{.field times_input}:  FALSE (",
                      symbol$arrow_right,
                      " {.emph SmoothGrad} method)"))
      }
      cli_li(paste0("{.field n}: ", self$n))
      cli_li(paste0("{.field noise_level}: ", self$noise_level))
      cli_end(id = i)
    }
  )
)


###############################################################################
#                                     Utils
###############################################################################

calc_times_input <- function(gradients, input) {
  for (i in seq_along(gradients)) {
    for (k in seq_along(gradients[[i]])) {
      grad <- gradients[[i]][[k]]
      if (is.null(grad)) {
        gradients[[i]][k] <- list(NULL)
      } else {
        gradients[[i]][[k]] <- grad * input[[k]]$unsqueeze(-1)
      }
    }
  }

  gradients
}
