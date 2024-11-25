###############################################################################
#                                 DeepLift
###############################################################################

#' @title Deep learning important features (DeepLift)
#'
#' @description
#' This is an implementation of the \emph{deep learning important features
#' (DeepLift)} algorithm introduced by Shrikumar et al. (2017). It's a local
#' method for interpreting a single element \eqn{x} of the dataset concerning
#' a reference value \eqn{x'} and returns the contribution of each input
#' feature from the difference of the output (\eqn{y=f(x)}) and reference
#' output (\eqn{y'=f(x')}) prediction. The basic idea of this method is to
#' decompose the difference-from-reference prediction with respect to the
#' input features, i.e.,
#' \deqn{\Delta y = y - y'  = \sum_i C(x_i).}
#' Compared to \emph{Layer-wise relevance propagation} (see [LRP]), the
#' DeepLift method is an exact decomposition and not an approximation, so we
#' get real contributions of the input features to the
#' difference-from-reference prediction. There are two ways to handle
#' activation functions: the *Rescale* rule (`'rescale'`) and
#' *RevealCancel* rule (`'reveal_cancel'`).
#'
#' The R6 class can also be initialized using the [`run_deeplift`] function
#' as a helper function so that no prior knowledge of R6 classes is required.
#'
#' @template param-converter
#' @template param-data
#' @template param-output_idx
#' @template param-output_label
#' @template param-channels_first
#' @template param-ignore_last_act
#' @template param-x_ref
#' @template param-dtype
#' @template param-verbose
#' @template field-x_ref
#' @template examples-DeepLift
#'
#' @references
#' A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#'
#' @family methods
#' @export
DeepLift <- R6Class(
  classname = "DeepLift",
  inherit = InterpretingMethod,
  public = list(

    #' @field rule_name (`character(1)`)\cr
    #' Name of the applied rule to calculate the contributions.
    #' Either `'rescale'` or `'reveal_cancel'`.\cr
    rule_name = NULL,
    x_ref = NULL,

    #' @description
    #' Create a new instance of the `DeepLift` R6 class. When initialized,
    #' the method *DeepLift* is applied to the given data and the results are stored in
    #' the field `result`.
    #'
    #' @param rule_name (`character(1)`)\cr
    #' Name of the applied rule to calculate the
    #' contributions. Use either `'rescale'` or `'reveal_cancel'`. \cr
    #' @param winner_takes_all (`logical(1)`)\cr
    #' This logical argument is only relevant for MaxPooling
    #' layers and is otherwise ignored. With this layer type, it is possible that
    #' the position of the maximum values in the pooling kernel of the normal input
    #' \eqn{x} and the reference input \eqn{x'} may not match, which leads to a
    #' violation of the summation-to-delta property. To overcome this problem,
    #' another variant is implemented, which treats a MaxPooling layer as an
    #' AveragePooling layer in the backward pass only, leading to an uniform
    #' distribution of the upper-layer contribution to the lower layer.\cr
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          output_label = NULL,
                          ignore_last_act = TRUE,
                          rule_name = "rescale",
                          x_ref = NULL,
                          winner_takes_all = TRUE,
                          verbose = interactive(),
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx, output_label,
                       ignore_last_act, winner_takes_all, verbose, dtype)

      cli_check(checkChoice(rule_name, c("rescale", "reveal_cancel")),
                "rule_name")
      self$rule_name <- rule_name

      if (is.null(x_ref)) {
          x_ref <- lapply(lapply(self$data, dim),
                          function(x) array(0, dim = c(1, x[-1])))
      }
      self$x_ref <- private$test_data(x_ref, name = "x_ref")

      # Check if x_ref is only a single instance
      num_instances <- unlist(lapply(self$x_ref, function(x) dim(x)[1]))
      if (any(num_instances != 1)) {
        stopf("For the method {.code DeepLift}, you have to pass ",
              "only a single instance for the argument {.arg x_ref}. ",
              "You passed (for at least one input layer) '",
              max(num_instances), "' data instances!")
      }

      self$converter$model$forward(self$data,
        channels_first = self$channels_first,
        save_input = TRUE,
        save_preactivation = TRUE,
        save_output = TRUE
      )
      self$converter$model$update_ref(self$x_ref,
        channels_first = self$channels_first,
        save_input = TRUE,
        save_preactivation = TRUE,
        save_output = TRUE
      )


      self$result <- private$run("DeepLift")
    }
  ),

  private = list(
    print_method_specific = function() {
      i <- cli_ul()
      cli_li(paste0("{.field rule_name}: '", self$rule_name, "'"))
      cli_li(paste0("{.field winner_takes_all}: ", self$winner_takes_all))
      all_zeros <- all(unlist(lapply(self$x_ref,
                                     function(x) all(as_array(x) == 0))))
      if (all_zeros) {
        s <- "zeros"
      } else {
        values <- unlist(lapply(self$x_ref, as_array))
        s <- paste0("mean: ", mean(values), " (q1: ", quantile(values, 0.25),
                    ", q3: ", quantile(values, 0.75), ")")
      }
      cli_li(paste0("{.field x_ref}: ", s))
      cli_end(id = i)
    }
  )
)


###############################################################################
#                                 DeepSHAP
###############################################################################

#' Deep Shapley additive explanations (DeepSHAP)
#'
#' @description
#' The *DeepSHAP* method extends the [`DeepLift`] technique by not only
#' considering a single reference value but by calculating the average
#' from several, ideally representative reference values at each layer. The
#' obtained feature-wise results are approximate Shapley values for the
#' chosen output, where the conditional expectation is computed using these
#' different reference values, i.e., the *DeepSHAP* method decompose the
#' difference from the prediction and the mean prediction \eqn{f(x) - E[f(\tilde{x})]}
#' in feature-wise effects. The reference values can be passed by the argument
#' `data_ref`.
#'
#' The R6 class can also be initialized using the [`run_deepshap`] function
#' as a helper function so that no prior knowledge of R6 classes is required.
#'
#' @template param-converter
#' @template param-data
#' @template param-output_idx
#' @template param-output_label
#' @template param-channels_first
#' @template param-ignore_last_act
#' @template param-dtype
#' @template param-verbose
#' @template examples-DeepSHAP
#'
#' @references
#' S. Lundberg & S. Lee (2017) \emph{A unified approach to interpreting model
#' predictions.}  NIPS 2017, p. 4768–4777
#'
#' @family methods
#' @export
DeepSHAP <- R6Class(
  classname = "DeepSHAP",
  inherit = InterpretingMethod,
  public = list(

    #' @field rule_name (`character(1)`)\cr
    #' Name of the applied rule to calculate the contributions.
    #' Either `'rescale'` or `'reveal_cancel'`.\cr
    #' @field data_ref (`list`)\cr
    #' The passed reference dataset for estimating the conditional expectation
    #' as a `list` of `torch_tensors` in the selected
    #' data format (field `dtype`) matching the corresponding shapes of the
    #' individual input layers. Besides, the channel axis is moved to the
    #' second position after the batch size because internally only the
    #' format *channels first* is used.\cr
    rule_name = NULL,
    data_ref = NULL,

    #' @description
    #' Create a new instance of the `DeepSHAP` R6 class. When initialized,
    #' the method *DeepSHAP* is applied to the given data and the results are
    #' stored in the field `result`.
    #'
    #' @param rule_name (`character(1)`)\cr
    #' Name of the applied rule to calculate the
    #' contributions. Use either `'rescale'` or `'reveal_cancel'`. \cr
    #' @param winner_takes_all (`logical(1)`)\cr
    #' This logical argument is only relevant for MaxPooling
    #' layers and is otherwise ignored. With this layer type, it is possible that
    #' the position of the maximum values in the pooling kernel of the normal input
    #' \eqn{x} and the reference input \eqn{x'} may not match, which leads to a
    #' violation of the summation-to-delta property. To overcome this problem,
    #' another variant is implemented, which treats a MaxPooling layer as an
    #' AveragePooling layer in the backward pass only, leading to an uniform
    #' distribution of the upper-layer contribution to the lower layer.\cr
    #' @param data_ref ([`array`], [`data.frame`],
    #' \code{\link[torch]{torch_tensor}} or `list`)\cr
    #' The reference data which is used to estimate the conditional expectation.
    #' These must have the same format as the input data of the passed model to
    #' the converter object. This means either
    #' \itemize{
    #'   \item an `array`, `data.frame`, `torch_tensor` or array-like format of
    #'   size *(batch_size, dim_in)*, if e.g., the model has only one input layer, or
    #'   \item a `list` with the corresponding input data (according to the
    #'   upper point) for each of the input layers.
    #'   \item or `NULL` (default) to use only a zero baseline for the estimation.\cr
    #' }
    #' @param limit_ref (`integer(1)`)\cr
    #' This argument limits the number of instances taken from the reference
    #' dataset `data_ref` so that only random `limit_ref` elements and not
    #' the entire dataset are used to estimate the conditional expectation.
    #' A too-large number can significantly increase the computation time.\cr
    initialize = function(converter, data,
                          channels_first = TRUE,
                          output_idx = NULL,
                          output_label = NULL,
                          ignore_last_act = TRUE,
                          rule_name = "rescale",
                          data_ref = NULL,
                          limit_ref = 100,
                          winner_takes_all = TRUE,
                          verbose = interactive(),
                          dtype = "float") {
      super$initialize(converter, data, channels_first, output_idx, output_label,
                       ignore_last_act, winner_takes_all, verbose, dtype)

      cli_check(checkChoice(rule_name, c("rescale", "reveal_cancel")), "rule_name")
      self$rule_name <- rule_name
      cli_check(checkInt(limit_ref), "limit_ref")

      # For default values of data_ref, DeepLift with zero baseline is applied
      if (is.null(data_ref)) {
        data_ref <- lapply(lapply(self$data, dim),
                           function(x) array(0, dim = c(1, x[-1])))
      }
      self$data_ref <- private$test_data(data_ref, name = "data_ref")

      # For computational reasons, the number of instances in the reference
      # dataset 'data_ref' is limited by the value `limit_ref`
      num_samples <- dim(self$data_ref[[1]])[1]
      ids <- sample.int(num_samples, min(num_samples, limit_ref))
      self$data_ref <- lapply(self$data_ref, function(x) x[ids, drop = FALSE])

      # Repeat values, s.t. `data` and `data_ref` have the same number of
      # instances
      num_samples <- dim(self$data[[1]])[1]
      num_samples_ref <- dim(self$data_ref[[1]])[1]
      data <- lapply(self$data, torch_repeat_interleave,
                     repeats = as.integer(num_samples_ref),
                     dim = 1) # now of shape (batch_size * num_samples_ref, input_dim)
      repeat_input <- function(x) {
        torch_cat(lapply(seq_len(num_samples), function(i) x))
      }
      data_ref <- lapply(self$data_ref, repeat_input) # now of shape (batch_size * num_samples_ref, input_dim)

      # Forward for normal input
      self$converter$model$forward(data,
                                   channels_first = self$channels_first,
                                   save_input = TRUE,
                                   save_preactivation = TRUE,
                                   save_output = TRUE
      )

      self$converter$model$update_ref(data_ref,
                                      channels_first = self$channels_first,
                                      save_input = TRUE,
                                      save_preactivation = TRUE,
                                      save_output = TRUE
      )

      result <- private$run("DeepSHAP")

      # For the DeepSHAP method, we only get the multiplier.
      # Hence, we have to multiply this by the differences of inputs
      fun <- function(result, out_idx, in_idx, x, x_ref, n) {
        res <- result[[out_idx]][[in_idx]]
        if (is.null(res)) {
          res <- NULL
        } else {
          res <- res * (x[[in_idx]] - x_ref[[in_idx]])$unsqueeze(-1)
          res <- torch_stack(res$chunk(n), dim = 1)$mean(2)
        }
      }
      result <- apply_results(result, fun, x = data, x_ref = data_ref,
                              n = num_samples)

      self$result <- result
    }
  ),

  private = list(
    print_method_specific = function() {
      i <- cli_ul()
      cli_li(paste0("{.field rule_name}: '", self$rule_name, "'"))
      cli_li(paste0("{.field winner_takes_all}: ", self$winner_takes_all))
      all_zeros <- all(unlist(lapply(self$data_ref,
                                     function(x) all(as_array(x) == 0))))
      if (all_zeros) {
        s <- "zeros"
      } else {
        values <- unlist(lapply(self$data_ref, as_array))
        s <- paste0("mean: ", mean(values), " (q1: ", quantile(values, 0.25),
                    ", q3: ", quantile(values, 0.75), ")")
      }
      cli_li(paste0("{.field data_ref}: ", s))
      cli_end(id = i)
    }
  )
)
