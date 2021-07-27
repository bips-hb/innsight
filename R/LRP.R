#' @include Interpreting_Method.R
#'
NULL


#' @title Layer-wise Relevance Propagation (LRP) method
#' @name LRP
#'
#' @description
#' This is an implementation of the \emph{Layer-wise Relevance Propagation (LRP)}
#' algorithm introduced by Bach et al. (2015). It's a local method for
#' interpreting a single element of the dataset and calculates the relevance scores for
#' each input feature. The basic idea of this method is to decompose the
#' prediction score of the model with respect to the input features, i.e.
#' \deqn{f(x) = \sum_i R(x_i).}
#' Because of the bias vector, this decomposition is generally an approximation.
#' There exist several propagation rules to determine the relevance scores. In this
#' package are implemented: simple rule ("simple"), epsilon rule ("epsilon") and
#' alpha-beta rule ("alpha_beta").
#'
#' @examples
#' library(neuralnet)
#' data(iris)
#' nn <- neuralnet( Species ~ .,
#'                  iris, linear.output = FALSE,
#'                  hidden = c(10,8), act.fct = "tanh", rep = 1, threshold = 0.5)
#' # create an analyzer for this model
#' analyzer = Analyzer$new(nn)
#'
#' # create new instance of 'LRP'
#' lrp <- LRP$new(analyzer, iris[,-5], rule_name = "simple")
#'
#' # get the result as an array
#' lrp$get_result()
#'
#' # get the result as a torch tensor
#' lrp$get_result(as_torch = TRUE)
#'
#' # use the alpha-beta rule with alpha = 2
#' lrp <- LRP$new(analyzer, iris[,-5], rule_name = "alpha_beta", rule_param = 2)
#'
#'
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46
#'
#' @export

LRP <- R6::R6Class(
  classname = "LRP",
  inherit = Interpreting_Method,

  public = list(

    #' @field data The given data as a torch tensor for which the relevance
    #' scores are to be calculated.
    #' @field analyzer The analyzer with the stored to torch converted model.
    #' @field dtype The type of the date (either `'float'` or `'double'`).
    #' @field channels_first The format of the given date, i.e. channels on
    #' last dimension (`FALSE`) or after the batch dimension (`TRUE`).
    #' @field result The calculated relevance scores of the given data as a
    #' torch tensor of size (batch_size, model_in, model_out).
    #' @field rule_name The name of the rule, with which the relevance scores are
    #' calculated. Implemented are \code{"simple"}, \code{"epsilon"}, \code{"alpha_beta"},
    #' (default: \code{"simple"}).
    #' @field rule_param The parameter of the selected rule.
    #'

    rule_name = NULL,
    rule_param = NULL,


    #' @description
    #' Create a new instance of the LRP-Method.
    #'
    #' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
    #' @param data The data for which the relevance scores are to be calculated.
    #' It has to be an array or array-like format of size (batch_size, model_in).
    #' @param rule_name The name of the rule, with which the relevance scores are
    #' calculated. Implemented are \code{"simple"}, \code{"epsilon"}, \code{"alpha_beta"},
    #' (default: \code{"simple"}).
    #' @param rule_param The parameter of the selected rule. Note: Only the rules
    #' \code{"epsilon"} and \code{"alpha_beta"} take use of the parameter. Use the default
    #' value \code{NULL} for the default parameters ("epsilon" : \eqn{0.01}, "alpha_beta" : \eqn{0.5}).
    #' @param channels_first Set the data format of the given data. Internally the format
    #' `channels_first` is used, therefore the format of the given data is required.
    #' Also use the default value `TRUE` if no convolutional layers are used.
    #' @param dtype The data type for the calculations. Use either `'float'` or
    #' `'double'`.
    #'
    #' @return A new instance of the R6 class `'LRP'`.
    initialize = function(analyzer, data,
                          channels_first = TRUE,
                          rule_name = "simple",
                          rule_param = NULL,
                          dtype = "float") {

      super$initialize(analyzer, data, channels_first, dtype)

      checkmate::assertChoice(rule_name, c('simple', 'epsilon', 'alpha_beta'))
      self$rule_name <- rule_name

      checkmate::assertNumber(rule_param, null.ok = TRUE)
      self$rule_param <- rule_param

      self$analyzer$model$forward(self$data, channels_first = self$channels_first)

      self$result <- private$run()

    },

    #'
    #' @description
    #' This function returns the relevance scores for the given data either as an
    #' array (`as_torch = FALSE`) or a torch tensor (`as_torch = TRUE`) of
    #' size (batch_size, model_in, model_out).
    #'
    #' @param as_torch Boolean value whether the output is an array or a torch
    #' tensor.
    #'
    #' @return The relevance scores of the given data.
    #'

    get_result = function(as_torch = FALSE) {
      result <- self$result
      if (!as_torch) {
        result <- as.array(result)
      }

      result
    },

    plot_relevances = function(i = NULL,j = NULL, rank = FALSE, scale = FALSE, ...){

      output_dim <- self$analyzer$output_dim
      batch_size <- dim(self$data)[1]
      rel <- torch_squeeze(self$result)
      rel_array <- as.array(rel)


      if (self$rule_name %in% c("epsilon", "alpha_beta")) {
        subtitle = sprintf("%s-Rule (%s)", rule_name, rule_param)
      } else {
        subtitle = sprintf("%s-Rule", self$rule_name)
      }

      if(!is.null(j)){
        aperm_array <- as.array(torch_tensor(aperm(rel_array,length(dim(rel_array)):1))[j,])
        rel_array <- aperm(aperm_array,length(dim(aperm_array)):1)

        rel <- torch_tensor(rel_array)
        output_dim <- 1
        names_out <- "Y"
      }

      print("before i")
      if(!is.null(i)){
        rel <- rel[i,]
        rel_array <- as.array(rel)

        batch_size <- 1
      }

      if("Dense_Layer" %in% self$analyzer$model$modules_list[[1]]$'.classes'){
        print("in dense")
        dim_in <- dim(self$data)[2]

        if(is.null(i)){names_in <- paste0("X",1:dim(rel_array)[[2]])}else{names_in <- paste0("X",1:length(rel_array))}
        if(is.null(j)){names_out <- paste0("Y",1:dim(rel_array)[[3]])}

        x <- rel_array

          features = rep(names_in, output_dim*batch_size)

          Class = rep(names_out, each = dim_in, times = batch_size)
          if (rank) {
            x[] <- apply(x, 3, function(z) apply(z,2, rank))
            y_min <- 1
            y_max <- dim(x)[1]
          } else if (scale) {
            y_min <- stats::quantile(x, 0.05)
            y_max <- stats::quantile(x, 0.95)
          } else {
            y_min <- min(x)
            y_max <- max(x)
          }
          Relevance = as.vector(x)
          Features <- factor(features, levels = names_in)
          ggplot2::ggplot(data.frame(Features, Class, Relevance),
                          mapping = ggplot2::aes(x = Features, y = Relevance, fill = Class), ...) +
            ggplot2::geom_boxplot(alpha = 0.6) +
            ggplot2::scale_fill_viridis_d() +
            ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
            ggplot2::ggtitle("Feature Importance with Layerwise Relevance Propagation", subtitle = subtitle)

      }else if("Conv2D_Layer" %in% self$analyzer$model$modules_list[[1]]$".classes"){
        print("in conv2d")
        if((!is.null(i) | batch_size == 1) && (!is.null(j) | output_dim == 1)){
          if(length(dim(rel_array)) == 3){

            rel <- torch_sum(rel,1)
            rel_array <- as.matrix(rel)
            print("before image")
            print(dim(as.data.frame(rel_array)))
            image(rel_array,col=grey(seq(0, 1, length = 256)))
            x <- 1:32
            y <- x
            #plot(rel_array)

          }
        }
      }
    }),

  private = list(

    run = function() {
      rev_layers <- rev(self$analyzer$model$modules_list)
      last_layer <- rev_layers[[1]]

      rel <- torch::torch_diag_embed(last_layer$preactivation)

      # other layers
      for (layer in rev_layers) {
        if("Flatten_Layer" %in% layer$'.classes') {
          rel <- layer$reshape_to_input(rel)
        }
        else{
          rel <- layer$get_input_relevances(rel,self$rule_name, self$rule_param)
        }
      }
      if (!self$channels_first) {
        rel <- torch::torch_movedim(rel, 2,length(dim(rel)) - 1)
      }

      rel
    }
  )
)
