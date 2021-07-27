
#'
#' @title Superclass for gradient-based interpretation methods
#' @description Superclass for gradient-based interpretation methods. This class
#' inherits from [Interpreting_Method].
#'
Gradient_Based <- R6::R6Class(
  classname = "Gradient_Based",
  inherit = Interpreting_Method,
  public = list(

    times_input = NULL,
    ignore_last_act = NULL,

    initialize = function(analyzer, data,
                          channels_first = TRUE,
                          times_input = TRUE,
                          ignore_last_act = TRUE,
                          dtype = 'float') {

      super$initialize(analyzer, data, channels_first, dtype)

      checkmate::assert_logical(times_input)
      self$times_input <- times_input

      checkmate::assert_logical(ignore_last_act)
      self$ignore_last_act <- ignore_last_act

    }
  ),

  private = list(
    calculate_gradients = function(input, ignore_last_act = FALSE) {

      input$requires_grad <- TRUE

      out <- self$analyzer$model(input, channels_first = self$channels_first)


      if (ignore_last_act) {
        output <- rev(self$analyzer$model$modules_list)[[1]]$preactivation
      }
      else {
        output <- out
      }

      # Implemented is only the case where the output is one-dimensional
      checkmate::assertTRUE(length(dim(output)) == 2)

      res <- vector(mode = 'list', length = dim(output)[2])
      out_sum <- sum(output, dim = 1)

      for (i in 1:dim(output)[2]) {
        res[[i]] <- autograd_grad(out_sum[i], input, retain_graph = TRUE)[[1]]
      }

      torch::torch_stack(res, dim = length(dim(input)) + 1)
    }
  )
)

#' @title Calculate the Gradients
#' @name Gradient
#'
#' @description
#' This method computes the gradients of the outputs with respect to the input
#' variables, i.e. for all input variable \eqn{i} and output class \eqn{j}
#' \deqn{d f(x)_j / d x_i.}
#'
#'
#' @export
#'

Gradient <- R6::R6Class(
  classname = "Gradient",
  inherit = Gradient_Based,
  public = list(

    initialize = function(analyzer, data,
                          channels_first = TRUE,
                          times_input = TRUE,
                          ignore_last_act = TRUE,
                          dtype = "float") {

      super$initialize(analyzer, data, channels_first, times_input, ignore_last_act, dtype)

      self$result <- private$run()

      gc() # we have to call gc otherwise R tensors are not disposed.

    }
  ),

  private = list(
    run = function() {

      gradients <- private$calculate_gradients(self$data,
                                               ignore_last_act = self$ignore_last_act)

      if (self$times_input) {
        gradients <- gradients * self$data$unsqueeze(-1)
      }

      gradients
    }
  )
)


#' @title SmoothGrad method
#' @name SmoothGrad
#'
#' @description
#' SmoothGrad was introduced by D. Smilkov et al. (2017) and is an extension to
#' the classical [Gradient] method. It takes the mean of the gradients for \code{n}
#' perturbations of each data point, i.e. with \eqn{\epsilon ~ N(0,\sigma)}
#' \deqn{1/n \sum_n d f(x+ \epsilon)_j / d x_j.}
#'
#' @references
#' D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @export
#'
SmoothGrad <- R6::R6Class(
  classname = "SmoothGrad",
  inherit = Gradient_Based,
  public = list(

    n = NULL,
    noise_level = NULL,

    initialize = function(analyzer, data,
                          n = 50,
                          noise_level = 0.1,
                          channels_first = TRUE,
                          dtype = "float",
                          times_input = TRUE,
                          ignore_last_act = TRUE) {

      super$initialize(analyzer, data, channels_first, times_input, ignore_last_act, dtype)

      checkmate::assertInt(n, lower = 1)
      checkmate::assertNumber(noise_level, lower = 0)
      self$n <- n
      self$noise_level <- noise_level

      self$result <- private$run()

      gc() # we have to call gc otherwise R tensors are not disposed.

    }
  ),

  private = list(
    run = function() {

      data <-
        torch::torch_repeat_interleave(
          self$data,
          repeats = torch::torch_tensor(self$n, dtype = torch::torch_long()),
          dim = 1)

      noise_scale <- self$noise_level * (max(data) - min(data))

      noise <- torch::torch_randn_like(data) * noise_scale

      gradients <- private$calculate_gradients(data + noise,
                                               ignore_last_act = self$ignore_last_act)

      smoothgrads <-
        torch::torch_stack(lapply(gradients$chunk(dim(self$data)[1]),
                                  torch::torch_mean,
                                  dim = 1),
                           dim = 1)

      if (self$times_input) {
        smoothgrads <- smoothgrads * self$data$unsqueeze(-1)
      }

      smoothgrads

    }
  )
)
