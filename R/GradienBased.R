###############################################################################
#                         Super class: GradientBased
###############################################################################

#'
#' @title Super class for Gradient-based Interpretation Methods
#' @description Super class for gradient-based interpretation methods. This
#' class inherits from [InterpretingMethod]. It summarizes all implemented
#' gradient-based methods and provides a private function to calculate the
#' gradients w.r.t. to the input for given data. Implemented are:
#'
#' - 'Vanilla Gradients' and 'Gradient x Input' ([Gradient])
#' - 'SmoothGrad' and 'SmoothGrad x Input' ([SmoothGrad])
#'
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-dtype
#' @template param-ignore_last_act
#' @template param-aggr_channels
#' @template param-as_plotly
#' @template param-ref_data_idx
#' @template param-individual_data_idx
#' @template param-individual_max
#' @template param-preprocess_FUN
#'
GradientBased <- R6Class(
  classname = "GradientBased",
  inherit = InterpretingMethod,
  public = list(

    #' @field times_input This logical value indicates whether the results
    #' were multiplied by the provided input data or not. If `TRUE`, the
    #' method is called *Gradient x Input*.
    times_input = NULL,

    #' @description
    #' Create a new instance of this class. When initialized,
    #' the method is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param times_input Multiplies the gradients with the input features.
    #' This method is called 'Gradient x Input'.
    #' @param output_idx These indices specify the output nodes for which
    #' the method is to be applied. In order to allow models with multiple
    #' output layers, there are the following possibilities to select
    #' the indices of the output nodes in the individual output layers:
    #' \itemize{
    #'   \item A `vector` of indices: If the model has only one output layer,
    #'   the values correspond to the indices of the output nodes, e.g.
    #'   `c(1,3,4)` for the first, third and fourth output node. If there are
    #'   multiple output layers, the indices of the output nodes from the first
    #'   output layer are considered.
    #'   \item A `list` of `vectors` of indices: If the method is to be
    #'   applied to output nodes from different layers, a list can be passed
    #'   that specifies the desired indices of the output nodes for each
    #'   output layer. Unwanted output layers have the entry `NULL` instead of
    #'   a vector of indices, e.g. `list(NULL, c(1,3))` for the first and
    #'   third output node in the second output layer.
    #'   \item `NULL` (default): The method is applied to all output nodes in
    #'   the first output layer but is limited to the first ten as the
    #'   calculations become more computationally expensive for more output
    #'   nodes.
    #' }
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = TRUE,
                          dtype = "float"
                          ) {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, dtype)

      assert_logical(times_input)
      self$times_input <- times_input
    },

    #' @description
    #' This method visualizes the result of individual data points of the
    #' selected method and enables a visual in-depth investigation with the help
    #' of the S4 classes [`innsight_ggplot2`] and [`innsight_plotly`].\cr
    #' You can use the argument `data_idx` to select the data points in the given
    #' data for the plot. In addition, the individual output nodes for the plot
    #' can be selected with the argument `output_idx`. The different results for
    #' the selected data points and outputs are visualized using the ggplot2-based
    #' S4 class `innsight_ggplot2`. You can also use the `as_plotly` argument to
    #' generate an interactive plot with `innsight_plotly` based on the
    #' plot function [plotly::plot_ly]. For more information and the whole bunch
    #' of possibilities, see [`innsight_ggplot2`] and [`innsight_plotly`].\cr
    #' \cr
    #' **Note:**
    #' 1. For the interactive plotly-based plots, the suggested package `plotly`
    #' is required.
    #' 2. The ggplot2-based plots for models with multiple input layers are a bit
    #' more complex, therefore the suggested packages `'grid'`, `'gridExtra'`
    #' and `'gtable'` must be installed in your R session.
    #'
    #' @param data_idx An integer vector containing the numbers of the data
    #' points whose result is to be plotted, e.g. `c(1,3)` for the first
    #' and third data point in the given data. Default: `1`.
    #' @param output_idx The indices of the output nodes for which the results is
    #' to be plotted. This can be either a `vector` of indices or a `list` of
    #' vectors of indices but must be a subset of the indices for which the
    #' results were calculated, i.e. a subset of `output_idx` from the
    #' initialization `new()` (see argument `output_idx` in method `new()` of this
    #' R6 class for details). By default (`NULL`), the smallest index of all
    #' calculated output nodes and output layers is used.
    #'
    #' @return
    #' Returns either an [`innsight_ggplot2`] (`as_plotly = FALSE`) or an
    #' [`innsight_plotly`] (`as_plotly = TRUE`) object with the plotted
    #' individual results.
    plot = function(data_idx = 1,
                    output_idx = NULL,
                    aggr_channels = 'sum',
                    as_plotly = FALSE) {

      private$plot(data_idx, output_idx, aggr_channels,
                   as_plotly, "Gradient")
    },

    #' @description
    #' This method visualizes the results of the selected method summarized as
    #' boxplots and enables a visual in-depth investigation of the global
    #' behavior with the help of the S4 classes [`innsight_ggplot2`] and
    #' [`innsight_plotly`].\cr
    #' You can use the argument `output_idx` to select the individual output
    #' nodes for the plot. For tabular and 1D data, boxplots are created in
    #' which a reference value can be selected from the data using the
    #' `ref_data_idx` argument. For images, only the pixel-wise median is
    #' visualized due to the complexity. The plot is generated using the
    #' ggplot2-based S4 class `innsight_ggplot2`. You can also use the
    #' `as_plotly` argument to generate an interactive plot with
    #' `innsight_plotly` based on the plot function [plotly::plot_ly]. For
    #' more information and the whole bunch of possibilities, see
    #' [`innsight_ggplot2`] and [`innsight_plotly`].\cr \cr
    #' **Note:**
    #' 1. For the interactive plotly-based plots, the suggested package `plotly`
    #' is required.
    #' 2. The ggplot2-based plots for models with multiple input layers are a bit
    #' more complex, therefore the suggested packages `'grid'`, `'gridExtra'`
    #' and `'gtable'` must be installed in your R session.
    #'
    #' @param output_idx The indices of the output nodes for which the results is
    #' to be plotted. This can be either a `vector` of indices or a `list` of
    #' vectors of indices but must be a subset of the indices for which the
    #' results were calculated, i.e. a subset of `output_idx` from the
    #' initialization `new()` (see argument `output_idx` in method `new()` of this
    #' R6 class for details). By default (`NULL`), the smallest index of all
    #' calculated output nodes and output layers is used.
    #' @param data_idx By default ("all"), all available data points are used to
    #' calculate the boxplot information. However, this parameter can be used
    #' to select a subset of them by passing the indices. E.g. with
    #' `c(1:10, 25, 26)` only the first 10 data points and
    #' the 25th and 26th are used to calculate the boxplots.
    #'
    #' @return
    #' Returns either an [`innsight_ggplot2`] (`as_plotly = FALSE`) or an
    #' [`innsight_plotly`] (`as_plotly = TRUE`) object with the plotted
    #' summarized results.
    boxplot = function(output_idx = NULL,
                       data_idx = "all",
                       ref_data_idx = NULL,
                       aggr_channels = 'norm',
                       preprocess_FUN = abs,
                       as_plotly = FALSE,
                       individual_data_idx = NULL,
                       individual_max = 20) {
      private$boxplot(output_idx, data_idx, ref_data_idx, aggr_channels,
                      preprocess_FUN, as_plotly, individual_data_idx,
                      individual_max, "Gradient")
    }
  ),
  private = list(
    calculate_gradients = function(input) {
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

      # Define Progressbar
      pb <- txtProgressBar(min = 0, max = length(unlist(self$output_idx)), style = 3)
      n <- 1

      # Definition of some temporary functions --------------------------------
      # Define function for calculating the gradients of one output
      calc_gradient_for_one_output <- function(idx, list_idx) {
        setTxtProgressBar(pb, n)
        n <<- n + 1

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
        res <- lapply(self$output_idx[[list_idx]], calc_gradient_for_one_output,
                      list_idx = list_idx)

        lapply(seq_along(res[[1]]), stack_outputs, results = res)
      }
      # End of definitions ----------------------------------------------------

      output_idx <-
        seq_along(self$output_idx)[!unlist(lapply(self$output_idx, is.null))]
      grads <- lapply(output_idx, calc_gradient_for_list_idx)

      lapply(input, function(i) i$requires_grad <- FALSE)
      close(pb)

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

#' @title Vanilla Gradient and Gradient x Input
#' @name Gradient
#'
#' @description
#' This method computes the gradients (also known as 'Vanilla Gradients') of
#' the outputs with respect to the input variables, i.e. for all input
#' variable \eqn{i} and output class \eqn{j}
#' \deqn{d f(x)_j / d x_i.}
#' If the argument `times_input` is `TRUE`, the gradients are multiplied by
#' the respective input value ('Gradient x Input'), i.e.
#' \deqn{x_i * d f(x)_j / d x_i.}
#'
#' @template examples-Gradient
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-dtype
#' @template param-ignore_last_act
#'
#' @family methods
#' @export
Gradient <- R6Class(
  classname = "Gradient",
  inherit = GradientBased,
  public = list(

    #' @description
    #' Create a new instance of the Vanilla Gradient method. When initialized,
    #' the method is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param times_input Multiplies the gradients with the input features.
    #' This method is called 'Gradient x Input'.
    #' @param output_idx These indices specify the output nodes for which
    #' the method is to be applied. In order to allow models with multiple
    #' output layers, there are the following possibilities to select
    #' the indices of the output nodes in the individual output layers:
    #' \itemize{
    #'   \item A `vector` of indices: If the model has only one output layer,
    #'   the values correspond to the indices of the output nodes, e.g.
    #'   `c(1,3,4)` for the first, third and fourth output node. If there are
    #'   multiple output layers, the indices of the output nodes from the first
    #'   output layer are considered.
    #'   \item A `list` of `vectors` of indices: If the method is to be
    #'   applied to output nodes from different layers, a list can be passed
    #'   that specifies the desired indices of the output nodes for each
    #'   output layer. Unwanted output layers have the entry `NULL` instead of
    #'   a vector of indices, e.g. `list(NULL, c(1,3))` for the first and
    #'   third output node in the second output layer.
    #'   \item `NULL` (default): The method is applied to all output nodes in
    #'   the first output layer but is limited to the first ten as the
    #'   calculations become more computationally expensive for more output
    #'   nodes.
    #' }
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = FALSE,
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, times_input, dtype)

      self$result <- private$run()
      self$converter$model$reset()
    }
  ),
  private = list(
    run = function() {
      message("Backward pass 'Gradient':")
      gradients <- private$calculate_gradients(self$data)

      if (self$times_input) {
        gradients <- calc_times_input(gradients, self$data)
      }

      gradients
    }
  )
)


###############################################################################
#                                 SmoothGrad
###############################################################################

#' @title SmoothGrad and SmoothGrad x Input
#'
#' @description
#' 'SmoothGrad' was introduced by D. Smilkov et al. (2017) and is an extension
#' to the classical Vanilla [Gradient] method. It takes the mean of the
#' gradients for \code{n} perturbations of each data point, i.e. with
#' \eqn{\epsilon \sim N(0,\sigma)}
#' \deqn{1/n \sum_n d f(x+ \epsilon)_j / d x_j.}
#' Analogous to the *Gradient x Input* method, you can also use the argument
#' *times_input* multiply the gradients by the inputs before taking the
#' average (*SmoothGrad x Input*).
#'
#' @template examples-SmoothGrad
#' @template param-converter
#' @template param-data
#' @template param-channels_first
#' @template param-dtype
#' @template param-ignore_last_act
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

    #' @field n Number of perturbations of the input data (default: \eqn{50}).
    #' @field noise_level The standard deviation of the Gaussian
    #' perturbation, i.e. \eqn{\sigma = (max(x) - min(x)) *} `noise_level`.
    #'
    n = NULL,
    noise_level = NULL,

    #' @description
    #' Create a new instance of the *SmoothGrad* method. When initialized,
    #' the method is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param times_input Multiplies the gradients with the input features.
    #' This method is called 'Gradient x Input'.
    #' @param output_idx These indices specify the output nodes for which
    #' the method is to be applied. In order to allow models with multiple
    #' output layers, there are the following possibilities to select
    #' the indices of the output nodes in the individual output layers:
    #' \itemize{
    #'   \item A `vector` of indices: If the model has only one output layer,
    #'   the values correspond to the indices of the output nodes, e.g.
    #'   `c(1,3,4)` for the first, third and fourth output node. If there are
    #'   multiple output layers, the indices of the output nodes from the first
    #'   output layer are considered.
    #'   \item A `list` of `vectors` of indices: If the method is to be
    #'   applied to output nodes from different layers, a list can be passed
    #'   that specifies the desired indices of the output nodes for each
    #'   output layer. Unwanted output layers have the entry `NULL` instead of
    #'   a vector of indices, e.g. `list(NULL, c(1,3))` for the first and
    #'   third output node in the second output layer.
    #'   \item `NULL` (default): The method is applied to all output nodes in
    #'   the first output layer but is limited to the first ten as the
    #'   calculations become more computationally expensive for more output
    #'   nodes.
    #' }
    #' @param n Number of perturbations of the input data (default: \eqn{50}).
    #' @param noise_level Determines the standard deviation of the gaussian
    #' perturbation, i.e. \eqn{\sigma = (max(x) - min(x)) *} `noise_level`.
    #'
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          ignore_last_act = TRUE,
                          times_input = FALSE,
                          n = 50,
                          noise_level = 0.1,
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx,
                       ignore_last_act, times_input, dtype)

      assertInt(n, lower = 1)
      assertNumber(noise_level, lower = 0)
      self$n <- n
      self$noise_level <- noise_level

      self$result <- private$run()
      self$converter$model$reset()
    }
  ),
  private = list(
    run = function() {
      data <- lapply(self$data, function(input)
        torch_repeat_interleave(
          input,
          repeats = torch_tensor(self$n, dtype = torch_long()),
          dim = 1
        ))

      data <- lapply(data, function(input) {
        noise_scale <- self$noise_level * (max(input) - min(input))
        noise <- torch_randn_like(input) * noise_scale

        input + noise
      })

      message("Backward pass 'SmoothGrad':")
      gradients <- private$calculate_gradients(data)

      if (self$times_input) {
        gradients <- calc_times_input(gradients, data)
      }

      smoothgrads <- lapply(gradients, function(grad_output)
        lapply(grad_output, function(grad_input) {
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
        }))

      smoothgrads
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

