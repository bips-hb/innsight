
#' Analyzer of an artificial Neural Network
#'
#' @description
#' This class analyzes a passed neural network and stores its internal structures
#' and layers independently of the actual class of the network. With the help
#' of this object, various methods of interpretable machine learning can be applied
#' to it for a better understanding of individual predictions or of the whole model.
#' You can use models from the following libraries:
#' * \code{\link[keras]{keras}},
#' * \code{\link[neuralnet]{neuralnet}}
#'
#' ## Implemented methods
#' Methods that aim to explain an individual prediction of a neural network are
#' called **local**. In contrast, **global** methods provide an explanation for the entire
#' model. An object of the Analyzer class can be applied to the following local
#' and global methods:
#' * Global:
#'     * [Connection_Weights], Olden et al. (2004)
#' * Local:
#'     * Layerwise Relevance Propagation ([LRP]), Bach et al. (2015)
#'     * Deep Learning Important Feartures ([DeepLift]), Shrikumar et al. (2017)
#'     * [SmoothGrad], Smilkov et al. (2017)
#'     * [Gradient]
#'
#'
#' @examples
#' #------------------------- neuralnet model ----------------------------------
#' library(neuralnet)
#' nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#'                 iris, linear.output = FALSE,
#' hidden = c(3,2), act.fct = "tanh", rep = 1)
#' analyzer = Analyzer$new(nn)
#'
#' #-------------------------- keras model -------------------------------------
#' library(keras)
#'
#' iris[,5] <- as.numeric(iris[,5]) -1
#' # Turn `iris` into a matrix
#' iris <- as.matrix(iris)
#' # Set iris `dimnames` to `NULL`
#' dimnames(iris) <- NULL
#' iris.training <- iris[, 1:4]
#' iris.trainingtarget <- iris[, 5]
#' # One hot encode training target values
#' iris.trainLabels <- to_categorical(iris.trainingtarget)
#'
#' # Define model
#' model <- keras_model_sequential()
#' model %>%
#'   layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>%
#'   layer_dropout(0.1) %>%
#'   layer_dense(units = 8, activation = 'relu') %>%
#'   layer_dropout(0.1) %>%
#'   layer_dense(units = 3, activation = 'softmax')
#'
#' # Compile the model
#' model %>% compile(
#'   loss = 'categorical_crossentropy',
#'   optimizer = 'adam',
#'   metrics = 'accuracy'
#' )
#'
#' # Train the model
#' history <- model %>% fit(
#'   iris.training,
#'   iris.trainLabels,
#'   epochs = 50,
#'   batch_size = 5,
#'   validation_split = 0.2, verbose = 0
#' )
#' analyzer = Analyzer$new(model)
#'
#' @references
#' * J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
#'  quantifying variable importance in artificial neural networks using
#'  simulated data.} Ecological Modelling 178, p. 389–397
#' * S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#'  classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46
#' * A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#' * D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @export
#'

Analyzer <- R6::R6Class("Analyzer",
public = list(

    #' @field model The given neural network.
    #' @field layers List of layers in the model. It contains only layers that
    #' are relevant for the evaluation. For example a list of
    #' \code{\link{Dense_Layer}}.
    #' @field num_layers Number of all layers in \code{layers}.
    #' @field last_input Last recorded input for the forward pass
    #' (default: \code{NULL}).
    #' @field last_input_ref Last recorded reference input for the forward pass
    #' (default: \code{NULL}).
    #' @field dim_in Dimension of the input features.
    #' @field dim_out Dimension of the models output, i.e. dimension of the
    #' response variables.
    #' @field feature_names A list of names for the input features.
    #' @field response_names A list of names for the response variables.
    #'

    model = NULL,
    layers = NULL,
    num_layers = NULL,
    last_input = NULL,
    last_input_ref = NULL,
    dim_in = NULL,
    dim_out = NULL,
    feature_names = NULL,
    response_names = NULL,

###-----------------------------Initialize--------------------------------------
    #' @description
    #' Create a new analyzer for a given neural network.
    #'
    #' @param model A trained neural network for classification or regression
    #' tasks to be interpreted. Only models from the following types or packages
    #' are allowed: \code{\link[keras]{keras_model}},
    #' \code{\link[keras]{keras_model_sequential}} or
    #' \code{\link[neuralnet]{neuralnet}}.
    #' @param feature_names A list of names for the input features. Use the
    #' default value \code{NULL} for the default names (X1, X2, ...).
    #' @param response_names A list of names for the response variables. Use the
    #' default value \code{NULL} for the default names (Y1, Y2, ...).
    #'
    #' @return A new instance of the R6 class \code{'Analyzer'}.
    #'

    initialize = function(model, feature_names = NULL, response_names = NULL) {
      checkmate::assertVector(feature_names, null.ok = TRUE)
      checkmate::assertVector(response_names, null.ok = TRUE)

      # Analyze the passed model and store its internal structure in a list of
      # layers

      # Torch model
      if (inherits(model, "nn_module")) {
        self$layers = analyze_torch_model(model)
      }
      # Neuralnet model
      else if (inherits(model, "nn")) {
        self$layers = analyze_neuralnet_model(model)
        self$feature_names = model$model.list$variables
        self$response_names = model$model.list$response
      }
      # Keras model
      else if (inherits(model, c("keras.engine.sequential.Sequential", "keras.engine.functional.Functional"))) {
        self$layers = analyze_keras_model(model)
      }
      else {
        stop(sprintf("Unknown model of class \"%s\".", paste0(class(model), collapse = "\", \"")))
      }

      self$dim_in <- self$layers[[1]]$dim[1]
      self$dim_out <- rev(self$layers)[[1]]$dim[2]

      # Set the names for the feature and response variables
      if (length(feature_names)  == self$dim_in) {
        self$feature_names <- feature_names
      }
      else if (!is.null(feature_names)) {
        warning("Wrong length of argument 'feature_names'. Use default values instead.")
      }
      if (length(self$feature_names) != self$dim_in) {
        self$feature_names = paste0(rep("X", self$dim_in), 1:self$dim_in)
      }

      if (length(response_names)  == self$dim_out) {
        self$response_names <- response_names
      }
      else if (!is.null(response_names)) {
        warning("Wrong length of argument 'response_names'. Use default values instead.")
      }
      if (length(self$response_names) != self$dim_out) {
        self$response_names = paste0(rep("Y", self$dim_out), 1:self$dim_out)
      }

      self$num_layers <- length(self$layers)
    },

###-------------------------forward and update----------------------------------
    #' @description
    #'
    #' The forward method of the whole model, i.e. it calculates the output
    #' \eqn{y=f(x)} of a given input \eqn{x} respectively an reference input \eqn{x'}.
    #' In doing so all intermediate values are stored in the individual layers.
    #' A batch-wise evaluation is performed, hence \eqn{x} must be a matrix of
    #' inputs.
    #'
    #' @param x Input matrix of the model with size \emph{(num_data, dim_in)}.
    #' @param x_ref Reference input vector of the model. If this value is not needed, it
    #' can be set to the default value \code{NULL}.
    #'
    #' @return A list with two vectors. The first one is the output for the inputs
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
    #' list \code{layers} when the inputs \code{x} or reference input \code{x_ref}
    #' has changed.
    #'
    #' @param x Input matrix of the model with size \emph{(num_data, dim_in)}.
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
    })
)


implemented_layers <- c("Dense", "Dropout", "InputLayer")

analyze_keras_model <- function(model) {
  if (!requireNamespace("keras")) {
    stop("Please install the 'keras' package.")
  }
  layers_list = list()
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
        layers_list <- c(layers_list, Dense_Layer$new(weights = weights,
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
}

analyze_neuralnet_model <- function(model) {
  if (!requireNamespace("neuralnet")) {
    stop("Please install the 'neuralnet' package.")
  }
  tryCatch({
    layers_list = list()

    if (!("result.matrix" %in% names(model))) {
      stop("The model hasn't been fitted yet!")
    }

    # Get number of best repition
    if (ncol(model$result.matrix) == 1 ) {
      best_rep = 1
    } else {
      best_rep = which.min(model$result.matrix["error",])
    }

    weights <- model$weights[[best_rep]]
    act_name <- attributes(model$act.fct)$type
    if (act_name == "function") {
      stop("You can't use custom activation functions for this package.")
    }
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
    stop(cond)
  })
}

analyze_torch_model <- function(model) {
  if (!requireNamespace("torch")) {
    stop("Please install the 'torch' package.")
  }
  if (torch::is_nn_module(model)) {
    analyze_torch_model(model)
  } else {
    stop("This model isn't a valid torch model!")
  }

  # todo
  layers_list <- list()
  cat("Not implemented yet\n")
}
