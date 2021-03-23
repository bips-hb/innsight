library(R6)

#' Analyzer of an artificial Neural Network
#'
#' @description
#' This class analyzes a passed neural network and then provides various ways
#' to better understand the predictions and the overall model. Implemented are
#' methods like \emph{Layer-wise Relevance Propagation (LRP)}, \emph{DeepLift},
#' \emph{Connection Weights}, \emph{SmoothGrad}, \emph{Gradient},
#' \emph{Input times Gradient}.
#'
#' @export
#'

Analyzer <- R6Class("Analyzer",
public = list(

    #' @field model The given neural network. Can be a \code{keras}, \code{torch}, or
    #' \code{neuralnet} model.
    #' @field layers List of layers in the model. It contains only layers that
    #' are relevant for the evaluation. For example a list of
    #' \code{\link{Dense_Layer}}.
    #' @field num_layers Number of all layers in \code{layers}.
    #' @field last_input Last recorded input for the forward pass
    #' (default: \code{NULL}).
    #' @field last_input_ref Last recorded reference input for the forward pass
    #' (default: \code{NULL}).

    model = NULL,
    layers = NULL,
    num_layers = NULL,
    last_input = NULL,
    last_input_ref = NULL,

###-----------------------------Initialize--------------------------------------
    #' @description
    #' Create a new analyzer for a given neural network.
    #'
    #' @param model A trained neural network for classification or regression
    #' tasks to be interpreted. Only models from the following types or packages
    #' are allowed: \code{\link[keras]{keras_model}},
    #' \code{\link[keras]{keras_model_sequential}} or
    #' \code{\link[neuralnet]{neuralnet}}.
    #'
    #' @return A new instance of the R6 class \code{'Analyzer'}.
    #'

    initialize = function(model) {
        self$model = model
        self$layers <- analyze_model(model)
        self$num_layers <- length(self$layers)
    },

###-------------------------forward and update----------------------------------
    #' @description
    #'
    #' The forward method of the whole model, i.e. it calculates the output
    #' \eqn{y=f(x)} of a given input \eqn{x} respectively an reference input \eqn{x'}.
    #' In doing so all intermediate values are stored in the individual layers.
    #'
    #' @param x Input vector of the model.
    #' @param x_ref Reference input vector of the model. If this value is not needed, it
    #' can be set to the default value \code{NULL}.
    #'
    #' @return A list with two vectors. The first one is the output for input
    #' \code{x} and the second entry is the output for the
    #' reference input \code{x_ref}.
    #'
    forward = function(x, x_ref = NULL) {
        self$last_input <- x
        self$last_input_ref <- x_ref

        y = list(x, x_ref)
        for (layer in self$layers) {
            y <- layer$forward(y[[1]], y[[2]])
        }
        y
    },

    #' @description
    #'
    #' This method updates the stored intermediate values in each layer from the
    #' list \code{layers} when the input \code{x} or reference input \code{x_ref}
    #' has changed.
    #'
    #' @param x Input vector of the model.
    #' @param x_ref Reference input vector of the model. If this value is not needed, it
    #' can be set to the default value \code{NULL}.
    #'
    #' @return Returns the instance itself.
    #'

    update = function(x, x_ref = NULL) {
        if ( !identical(x, self$last_input) || !identical(x_ref, self$last_input_ref) ) {
            self$forward(x, x_ref)
            self$last_input = x
            self$last_input_ref = x_ref
        }
        invisible(self)
    },

###----------------------Connection Weights-------------------------------------
    #' @description
    #' This is an implementation of the \emph{Connection Weights} algorithm
    #' investigated by Olden et al. (2004). It's a global method for
    #' interpreting a model and therefore returns the feature importance for
    #' each input variable as a vector. The main calculation is outsourced to the
    #' method \code{\link{func_connection_weights}}.
    #'
    #' @param out_class If the given model is a classification model, this
    #' parameter can be used to determine which class the importance should be
    #' calculated for. Use the default value \code{NULL} to return the importance
    #' for all classes.
    #'
    #' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
    #' which contains the importance scores for each input variable to the
    #' output predictions. Otherwise returns a vector of the importance scores
    #' for each input variable for the given output class.
    #'
    #' @examples
    #' # neuralnet example
    #' library(neuralnet)
    #'
    #' # train a neural network
    #' nn <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length
    #'                + Petal.Width, iris, linear.output = FALSE,
    #'                hidden = c(5,4), act.fct = "tanh", rep = 2)
    #'
    #' # create an analyzer for this model
    #' analyzer = Analyzer$new(nn)
    #'
    #' # calculate importance score for all three classes with the
    #' # connection weight method
    #' analyzer$Connection_Weights()
    #'
    #' # calculate importance only for class 1
    #' analyzer$Connection_Weights(out_class = 1)
    #'
    #' @seealso
    #' \code{\link{func_connection_weights}}
    #'
    #' @references
    #' J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
    #' quantifying variable importance in artificial neural networks using
    #' simulated data.} Ecological Modelling 178, p. 389–397
    #'

    Connection_Weights = function(out_class = NULL) {
        importance = func_connection_weights(self$layers, out_class)
        importance
    },

###--------------------Layer-wise Relevance Propagation--------------------------
    #' @description
    #' This is an implementation of the \emph{Layer-wise Relevance Propagation (LRP)}
    #' algorithm introduced by Bach et al. (2015). It's a local method for
    #' interpreting a single element of the dataset and returns the relevance for
    #' each input feature. The main calculation is outsourced to the
    #' method \code{\link{func_lrp}}.
    #'
    #'
    #' @param x The input vector of the model to be interpreted.
    #' @param out_class If the given model is a classification model, this
    #' parameter can be used to determine which class the relevance should be
    #' calculated for. Use the default value \code{NULL} to return the relevance
    #' for all classes.
    #' @param rule_name The name of the rule, with which the relevance scores are
    #' calculated. Implemented are \code{"simple"}, \code{"eps"}, \code{"ab"},
    #' \code{"ww"} (default: \code{"simple"}).
    #' @param rule_param The parameter of the selected rule. Note: Only the rules
    #' \code{"eps"} and \code{"ab"} take use of the parameter. Use the default
    #' value \code{NULL} for the default parameters ("eps" : \eqn{0.01}, "ab" : \eqn{0.5}).
    #'
    #'
    #' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
    #' which contains the relevance scores for each input variable to the
    #' output predictions. Otherwise returns a vector of the relevance scores
    #' for each input variable for the given output class.
    #'
    #' @examples
    #' library(neuralnet)
    #'
    #' # train a model
    #' nn <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length
    #'                 + Petal.Width, iris, linear.output = FALSE,
    #'                 hidden = c(5,4), rep = 2)
    #'
    #' # create an analyzer for this model
    #' analyzer = Analyzer$new(nn)
    #'
    #' # get one example from the dataset as vector
    #' input <- as.vector(t(iris[1,-5]))
    #'
    #' # calculate relevance scores for class 2
    #' analyzer$LRP(input, out_class = 2)
    #'
    #' # calculate relevance scores for all classes
    #' analyzer$LRP(input)
    #'
    #' # calculate relevance scores for all classes with eps-rule
    #' analyzer$LRP(input, rule_name = "eps")
    #'
    #' @seealso
    #' \code{\link{func_lrp}}, \code{\link{linear_simple_rule}},
    #' \code{\link{linear_eps_rule}}, \code{\link{linear_ab_rule}},
    #' \code{\link{linear_ww_rule}}
    #'
    #' @references
    #' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
    #' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46

    LRP = function(x, out_class = NULL, rule_name = "simple", rule_param = NULL ){
        self$update(x)

        relevance = func_lrp(self$layers, out_class, rule_name, rule_param)

        relevance

    },

###---------------------------DeepLift------------------------------------------

    #' @description
    #' This is an implementation of the \emph{Deep Learning Important FeaTures (DeepLIFT)}
    #' algorithm introduced by Shrikumar et al. (2017). It's a local method for
    #' interpreting a single element concerning a reference value and
    #' returns the contribution for each input feature to the difference-from-reference
    #' output. The main calculation is outsourced to the
    #' method \code{\link{func_deeplift}}.
    #'
    #' @param x The input vector of the model to be interpreted.
    #' @param x_ref The reference input vector for the interpretation.
    #' @param rule_name Name of the applied rule to calculate the contributions. Use one
    #' of \code{"rescale"} and \code{"revealcancel"} (default: \code{"rescale"}).
    #' @param out_class If the given model is a classification model, this
    #' parameter can be used to determine which class the contributions should be
    #' calculated for. Use the default value \code{NULL} to return the contribution
    #' for all classes.
    #'
    #' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
    #' which contains the contribution values for each input variable to the
    #' output predictions. Otherwise returns a vector of the contribution values
    #' for each input variable for the given output class.
    #'
    #' @examples
    #' library(neuralnet)
    #'
    #' # train a model
    #' nn <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length
    #'                + Petal.Width, iris, linear.output = FALSE,
    #'                hidden = c(5,4), rep = 2)
    #'
    #' # create an analyzer for this model
    #' analyzer = Analyzer$new(nn)
    #'
    #' # get one example from the dataset as vector
    #' input <- as.vector(t(iris[1,-5]))
    #' input_ref <- rnorm(4)
    #'
    #' # calculate contribution scores for class 2
    #' analyzer$DeepLift(input, input_ref, out_class = 2)
    #'
    #' # calculate contribution scores for all classes
    #' analyzer$DeepLift(input, input_ref)
    #'
    #' # calculate contribution scores for all classes with reveal-cancel-rule
    #' analyzer$DeepLift(input, input_ref, rule_name = "revealcancel")
    #'
    #' @seealso
    #' \code{\link{func_deeplift}}, \code{\link{rescale_rule}},
    #' \code{\link{reveal_cancel_rule}}
    #'
    #' @references
    #' A. Shrikumar et al. (2017) \emph{Learning important features through
    #' propagating activation differences.}  ICML 2017, p. 4844-4866
    #'

    DeepLift = function(x, x_ref = NULL, rule_name = "rescale", out_class = NULL) {

        if ( is.null(x_ref) ) {
            x_ref <- x * 0
        }
        self$update(x, x_ref)

        contrib <- func_deeplift(self$layers, rule_name, out_class)
        contrib
    },

###------------------------Gradient based methods ------------------------------

    #' @description
    #' This method computes the gradients of the outputs with respect to the input
    #' variables, i.e. for all input variable \eqn{i} and output class \eqn{j}
    #' \deqn{\frac{\partial f(x)_j}{\partial x_i}.}
    #' The main calculation is outsourced to the
    #' method \code{\link{func_gradient}}.
    #'
    #' @param x The input vector of the model to be interpreted.
    #' @param out_class If the given model is a classification model, this
    #' parameter can be used to determine which class the gradients should be
    #' calculated for. Use the default value \code{NULL} to return the gradients
    #' for all classes.
    #'
    #' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
    #' which contains the gradients for each input variable to the
    #' output predictions. Otherwise returns a vector of the gradient
    #' for each input variable for the given output class.
    #'
    #' @examples
    #' library(neuralnet)
    #'
    #' # train a model
    #' nn <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length
    #'                 + Petal.Width, iris, linear.output = FALSE,
    #'                 hidden = c(5,4), rep = 2)
    #'
    #' # create an analyzer for this model
    #' analyzer = Analyzer$new(nn)
    #' # get one example from the dataset as vector
    #' input <- as.vector(t(iris[1,-5]))
    #' input
    #'
    #' # calculate the gradients for class 2
    #' analyzer$Gradients(input, out_class = 2)
    #'
    #' # calculate the gradients for all classes
    #' analyzer$Gradients(input)
    #'
    #' @seealso
    #' \code{\link{func_gradient}}
    #'
    Gradients = function(x, out_class = NULL) {
        self$update(x)

        gradients <- func_gradient(self$layers, out_class)
        gradients
    },

    #' @description
    #' This method computes the (smoothed) gradients of the outputs with respect
    #' to the input variables and multiplies these with the input vector.
    #'
    #' @param x The input vector of the model to be interpreted.
    #' @param grad_type Use \code{"normal"} for the normal calculated gradients
    #' and \code{"smooth"} for the smoothed gradients (default: \code{"normal"}).
    #' @param out_class If the given model is a classification model, this
    #' parameter can be used to determine which class the gradients times input should be
    #' calculated for. Use the default value \code{NULL} to return the gradients
    #' times input for all classes.
    #' @param n Number of perturbations of the input vector (default: \eqn{50}).
    #' This parameter is only for smoothed gradients required.
    #' @param noise_level Determines the standard deviation of the gaussian
    #' perturbation, i.e. \eqn{\sigma = (\max(x) - \min(x)) *} \code{noise_level}.
    #' This parameter is only for smoothed gradients required.
    #'
    #' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
    #' which contains the gradients times input for each input variable to the
    #' output predictions. Otherwise returns a vector of the gradient times input
    #' for each input variable for the given output class.
    #'
    #' @examples
    #' library(neuralnet)
    #'
    #' # train a model
    #' nn <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length
    #'                 + Petal.Width, iris, linear.output = FALSE,
    #'                 hidden = c(5,4), rep = 2)
    #'
    #' # create an analyzer for this model
    #' analyzer = Analyzer$new(nn)
    #' # get one example from the dataset as vector
    #' input <- as.vector(t(iris[1,-5]))
    #'
    #' # calculate the input times gradients for class 2
    #' analyzer$Inputs_times_Gradients(input, out_class = 2)
    #'
    #' # calculate the smoothed input times gradients for all classes
    #' analyzer$Inputs_times_Gradients(input, grad_type = "smooth")
    #'
    #' @seealso
    #' \code{\link{func_gradient}}
    #'

    Inputs_times_Gradients = function(x, grad_type = "normal", out_class = NULL, n = 50, noise_level = 0.3) {
        if (grad_type == "normal") {
          in_t_grad <- self$Gradients(x, out_class) * x
        } else if (grad_type == "smooth") {
          in_t_grad <- self$SmoothGrad(x, out_class, n, noise_level) * x
        } else {
          stop(sprintf("Unknown parameter \"%s\" for 'grad_type'. Use \"normal\" or \"smooth\".", grad_type))
        }
        in_t_grad
    },

    #' @description
    #' This is an implementation of the \emph{SmoothGrad} algorithm introduced
    #' by D. Smilkov et al. (2017).
    #' It computes smoothed gradients of the outputs with respect to the input
    #' variables by averaging over gradients of the randomly perturbed input,
    #' i.e. for all input variable \eqn{i} and output class \eqn{j}
    #' \deqn{\frac{1}{n} \sum_i^n \frac{\partial f(x + \varepsilon )_j}{\partial x_i}}
    #' with \eqn{\varepsilon \sim \mathcal{N}(0, \sigma^2)}.
    #'
    #' @param x The input vector of the model to be interpreted.
    #' @param out_class If the given model is a classification model, this
    #' parameter can be used to determine which class the gradients should be
    #' calculated for. Use the default value \code{NULL} to return the gradients
    #' for all classes.
    #' @param n Number of perturbations of the input vector (default: \eqn{50}).
    #' @param noise_level Determines the standard deviation of the gaussian
    #' perturbation, i.e. \eqn{\sigma = (\max(x) - \min(x)) *} \code{noise_level}.
    #'
    #' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
    #' which contains the gradients for each input variable to the
    #' output predictions. Otherwise returns a vector of the gradient
    #' for each input variable for the given output class.
    #'
    #' @examples
    #' library(neuralnet)
    #'
    #' # train a model
    #' nn <- neuralnet(Species ~Sepal.Length+ Sepal.Width + Petal.Length
    #'                 + Petal.Width, iris, linear.output = FALSE,
    #'                 hidden = c(5,4), rep = 2)
    #'
    #' # create an analyzer for this model
    #' analyzer = Analyzer$new(nn)
    #' # get one example from the dataset as vector
    #' input <- as.vector(t(iris[1,-5]))
    #' input
    #'
    #' # calculate the smoothed gradients for class 2
    #' analyzer$SmoothGrad(input, out_class = 2)
    #'
    #' # calculate the smoothed gradients for all classes
    #' analyzer$SmoothGrad(input)
    #'
    #' @seealso
    #' \code{\link{func_gradient}}
    #'
    #' @references
    #' D. Smilkov et al. (2017) \emph{SmoothGrad: Removing noise by adding noise.}
    #' arXiv: 1706.03825
    #'
    SmoothGrad = function(x, out_class = NULL, n = 50, noise_level = 0.3 ) {
        sigma <- noise_level * (max(x) - min(x))
        smooth_grad <- 0
        for (i in 1:n) {
            input_pert <- x + rnorm(length(x), mean = 0, sd = sigma)
            self$update(input_pert)
            smooth_grad <- smooth_grad + func_gradient(self$layers, out_class = out_class)
        }
        smooth_grad / n
    }
  )
)




analyze_model <- function(model) {
  if (inherits(model, "nn_module")) {
    require(torch)
    if (torch::is_nn_module(model)) {
      analyze_torch_model(model)
    } else {
      stop("This model isn't a valid torch model!")
    }
  }
  else if (inherits(model, "nn")) {
    #library(neuralnet)
    analyze_neuralnet_model(model)
  }
  else if (inherits(model, c("keras.engine.sequential.Sequential", "keras.engine.functional.Functional"))) {
    #library(keras)
    analyze_keras_model(model)
  } else {
    stop(sprintf("Unknown model of class \"%s\".", paste0(class(model), collapse = "\", \"")))
  }
}

implemented_layers <- c("Dense", "Dropout", "InputLayer")

analyze_keras_model <- function(model) {
  layers_list <- list()
  tryCatch({
    for (layer in model$layers) {
      type <- layer$`__class__`$`__name__`

      if (type %in% implemented_layers) {
        if (type == "Dropout" || type == "InputLayer") {
          message(sprintf("Skipping %s-Layer...", type))
        } else {
          act_name <- layer$activation$`__name__`
          weights <- layer$weights[[1]]$numpy()
          bias <- as.vector(layer$weights[[2]]$numpy())
          activation <- get_activation(act_name)
          layers_list <- c(layers_list, Dense_Layer$new(#type = type,
                                                  weights = weights,
                                                  bias = bias,
                                                  activation = activation,
                                                  activation_name = act_name))
        }
      } else {
        stop(sprintf("Layer of type \"%s\" is not implemented yet. Supported layers are: \"%s\"", type,
                     paste0(implemented_layers, collapse = "\", \"")))
      }
    }
    layers_list
  },
  error=function(cond) {
    warning(cond)
  })
}

analyze_neuralnet_model <- function(model) {
  tryCatch({
    layers_list = list()

    if (!("result.matrix" %in% names(model))) {
      warning("The model hasn't been fitted yet!")
    }

    # Get number of best repition
    if (ncol(model$result.matrix) == 1 ) {
      best_rep = 1
    } else {
      best_rep = which.min(model$result.matrix["error",])
    }

    weights <- model$weights[[best_rep]]
    act_name <- attributes(model$act.fct)$type
    act <- get_activation(act_name)

    for (i in 1:length(weights)) {

      b <- as.vector(weights[[i]][1,])
      w <- matrix(weights[[i]][-1,], ncol = length(b))

      if (i == length(weights) && model$linear.output == TRUE) {
        layers_list[[i]] = Dense_Layer$new(weights = w,
                                     bias = b,
                                     activation = get_activation("linear"),
                                     activation_name = "linear"
        )
      } else {
        layers_list[[i]] = Dense_Layer$new(weights = w,
                                     bias = b,
                                     activation = act,
                                     activation_name = act_name
        )
      }
    }
    layers_list
  },
  error=function(cond) {
    warning(cond)
  })
}

analyze_torch_model <- function(model) {
  # todo
  layers_list <- list()
  cat("Not implemented yet\n")
}




#r <- analyze_model(model_keras_seq)
#r
#r <- analyze_model(model_keras)
#r <- analyze_model(model_neuralnet)
#r
#r <- analyze_model(model_torch_seq)
#r <- analyze_model(model_torch)
#r <- analyze_model(torch_m)



