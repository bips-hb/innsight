
#' @title Layer-wise Relevance Propagation (LRP) Method
#' @name LRP
#'
#' @description
#' This is an implementation of the \emph{Layer-wise Relevance Propagation
#' (LRP)} algorithm introduced by Bach et al. (2015). It's a local method for
#' interpreting a single element of the dataset and calculates the relevance
#' scores for each input feature. The basic idea of this method is to decompose
#' the prediction score of the model with respect to the input features, i.e.
#' \deqn{f(x) = \sum_i R(x_i).}
#' Because of the bias vector, this decomposition is generally an approximation.
#' There exist several propagation rules to determine the relevance scores.
#' In this package are implemented: simple rule ("simple"), epsilon rule
#' ("epsilon") and alpha-beta rule ("alpha_beta").
#'
#' @examples
#' library(neuralnet)
#' data(iris)
#' nn <- neuralnet(Species ~ .,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(10, 8), act.fct = "tanh", rep = 1, threshold = 0.5
#' )
#' # create an converter for this model
#' converter <- Converter$new(nn)
#'
#' # create new instance of 'LRP'
#' lrp <- LRP$new(converter, iris[, -5], rule_name = "simple")
#'
#' # get the result as an array
#' lrp$get_result()
#'
#' # get the result as a torch tensor
#' lrp$get_result(type = "torch.tensor")
#'
#' # use the alpha-beta rule with alpha = 2
#' lrp <- LRP$new(converter, iris[, -5],
#'   rule_name = "alpha_beta",
#'   rule_param = 2
#' )
#'
#' # include the last activation into the calculation
#' lrp <- LRP$new(converter, iris[, -5],
#'   rule_name = "alpha_beta",
#'   rule_param = 2,
#'   ignore_last_act = FALSE
#' )
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10,
#' p. 1-46
#'
#' @export

LRP <- R6::R6Class(
  classname = "LRP",
  inherit = InterpretingMethod,
  public = list(
    #' @field rule_name The name of the rule, with which the relevance scores
    #' are calculated. Implemented are \code{"simple"}, \code{"epsilon"},
    #' \code{"alpha_beta"} (default: \code{"simple"}).
    #' @field rule_param The parameter of the selected rule.
    #'
    rule_name = NULL,
    rule_param = NULL,


    #' @description
    #' Create a new instance of the LRP-Method.
    #'
    #' @param converter An instance of the R6 class \code{\link{Converter}}.
    #' @param data The data for which the relevance scores are to be
    #' calculated. It has to be an array or array-like format of size
    #' (batch_size, dim_in).
    #' @param rule_name The name of the rule, with which the relevance scores
    #' are calculated. Implemented are \code{"simple"}, \code{"epsilon"},
    #' \code{"alpha_beta"} (default: \code{"simple"}).
    #' @param rule_param The parameter of the selected rule. Note: Only the
    #' rules \code{"epsilon"} and \code{"alpha_beta"} take use of the
    #' parameter. Use the default value \code{NULL} for the default parameters
    #' ("epsilon" : \eqn{0.01}, "alpha_beta" : \eqn{0.5}).
    #' @param channels_first Set the data format of the given data. Internally
    #' the format `channels_first` is used, therefore the format of the given
    #' data is required. Also use the default value `TRUE` if no convolutional
    #' layers are used.
    #' @param ignore_last_act Set this boolean value to include the last
    #' activation, or not (default: `TRUE`). In some cases, the last activation
    #' leads to a saturation problem.
    #' @param dtype The data type for the calculations. Use either `'float'` or
    #' `'double'`.
    #'
    #' @return A new instance of the R6 class `'LRP'`.
    #'
    initialize = function(converter, data,
                          channels_first = TRUE,
                          rule_name = "simple",
                          rule_param = NULL,
                          ignore_last_act = TRUE,
                          dtype = "float") {
      super$initialize(converter, data, channels_first, dtype, ignore_last_act)

      checkmate::assertChoice(rule_name, c("simple", "epsilon", "alpha_beta"))
      self$rule_name <- rule_name

      checkmate::assertNumber(rule_param, null.ok = TRUE)
      self$rule_param <- rule_param

      self$converter$model$forward(self$data,
        channels_first = self$channels_first
      )

      self$result <- private$run()
    }
  ),
  private = list(
    run = function() {
      rev_layers <- rev(self$converter$model$modules_list)
      last_layer <- rev_layers[[1]]

      if (self$ignore_last_act) {
        rel <- torch::torch_diag_embed(last_layer$preactivation)
      } else {
        rel <- torch::torch_diag_embed(last_layer$output)

        # For probabilistic output we need to subtract 0.5, such that
        # 0 means no relevance
        if (last_layer$activation_name %in%
          c("softmax", "sigmoid", "logistic")) {
          rel <- rel - 0.5
        }
      }

      # other layers
      for (layer in rev_layers) {
        if ("Flatten_Layer" %in% layer$".classes") {
          rel <- layer$reshape_to_input(rel)
        } else {
          rel <- layer$get_input_relevances(
            rel,
            self$rule_name,
            self$rule_param
          )
        }
      }
      if (!self$channels_first) {
        rel <- torch::torch_movedim(rel, 2, length(dim(rel)) - 1)
      }


      rel
    }
  )
)
