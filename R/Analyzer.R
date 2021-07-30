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
#'     * Vanilla [Gradient]
#'     * [SmoothGrad]
#'
#'
#' @field model The given neural network.
#' @field input_dim Dimension of the input features.
#' @field input_names Names of the input features
#' @field output_dim Dimension of the models output, i.e. dimension of the
#' response variables.
#' @field output_names A list of names for the response variables.
#'
#' @export
#'
Analyzer <- R6::R6Class("Analyzer",
  public = list(

    model = NULL,

    input_dim = NULL,
    input_names = NULL,

    output_dim = NULL,
    output_names = NULL,

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
    #' @param dtype The data type for the calculations. Use either `'float'` or `'double'`
    #'
    #'
    #' @return A new instance of the R6 class \code{'Analyzer'}.
    #'

    initialize = function(model, feature_names = NULL, response_names = NULL, dtype = 'float') {
      checkmate::assertArray(feature_names, null.ok = TRUE)
      checkmate::assertArray(response_names, null.ok = TRUE)
      checkmate::assertChoice(dtype, c('float', 'double'))

      # Analyze the passed model and store its internal structure in a list of
      # layers
      if (inherits(model, "nn")) {
        result <- analyze_neuralnet_model(model)
      }
      else if (inherits(model, c("keras.engine.sequential.Sequential", "keras.engine.functional.Functional"))) {
        result <- analyze_keras_model(model, dtype)
      }
      else if (inherits(model, "nn_module") && torch::is_nn_module(model)) {
        #
        # toDo
        #
      }
      else {
        stop(sprintf("Unknown model of class \"%s\".", paste0(class(model), collapse = "\", \"")))
      }

      self$model <- result$model
      self$input_dim <- result$input_dim
      self$output_dim <- result$output_dim
      self$input_names <-result$input_names
      self$output_names <- result$output_names
    }
  )
)



#' A \code{torch::nn_module} that stores the layers of the model to be analyzed
#' @description
#' This torch_module is how the different types of models to be analyzed (keras, neuralnet) are stored.
#'
#' @noRd
analyzed_model <- torch::nn_module(
  classname = "Analyzed_Model",

  #'@field modules_list The layers of the model in the form of a list of torch modules
  modules_list = NULL,
  dtype = NULL,

  initialize = function(modules_list, dtype = 'float') {
    self$modules_list <- modules_list
    self$dtype <- dtype
  },

  ###-------------------------forward and update----------------------------------
  #' @description
  #'
  #' The forward method of the whole model, i.e. it calculates the output
  #' \eqn{y=f(x)} of a given input \eqn{x}.
  #' In doing so all intermediate values are stored in the individual torch modules.
  #'
  #' @param x Input of the model with size \emph{(batch_size, dim_in)}.
  #'
  #' @return Returns the output for the inputs \code{x}.
  #'
  forward = function(x, channels_first = TRUE) {
    if (channels_first == FALSE) {
      x <- torch::torch_movedim(x, -1,2)
    }

    for (module in self$modules_list) {
      if ("Flatten_Layer" %in% module$.classes) {
        x <- module(x, channels_first)
      }
      else {
        x <- module(x)
      }
    }
    x
  },

  #' @description
  #'
  #' This method updates the stored intermediate values in each module from the
  #' list \code{modules_list} when the reference input \code{x_ref}
  #' has changed.
  #' @param x_ref Reference input of the model.
  #' @param channels_first If \code{TRUE}, any flatten layers will be flattened channels first, if \code{FALSE} they will be flattened
  #' channels last.
  #'
  #' @return Returns the instance itself.
  update_ref = function(x_ref, channels_first = TRUE) {

    if (channels_first == FALSE) {
      x_ref <- torch::torch_movedim(x_ref, -1,2)
    }
    for (module in self$modules_list) {
      if ("Flatten_Layer" %in% module$.classes) {
        x_ref <- module(x_ref, channels_first)
      }
      else {
        x_ref <- module$update_ref(x_ref)
      }
    }
    x_ref
  },

  set_dtype = function(dtype) {
    for (module in self$modules_list) {
      if (!('Flatten_Layer' %in% module$.classes)) {
        module$set_dtype(dtype)
      }
    }
    self$dtype <- dtype
  }

)

#'
#'@title Analyze a neuralnet model
#'@name analyze_neuralnet_model
#'@description
#'This function takes a neuralnet model as input and returns a torch analyzed_model module
#'with the same weights and biases of the original neuralnet model
#'@param model A neuralnet model
#'@return
#'This function returns a \code{result} obejct with attributes \code{result$model},
#' a torch nn_module with the same layers as the input neuralnet module. \code{result$input_dim}
#' is the input dimension of the model, \code{result$output_dim} the output dimensions of
#' the model, \code{result$input_names} is the names of the input variables, \code{result$output_names}
#' is the name of the output variables.
#'
#' @noRd
#'
analyze_neuralnet_model <- function(model, dtype = "float") {
  if (!requireNamespace("neuralnet")) {
    stop("Please install the 'neuralnet' package.")
  }

  # Test whether the model has been fitted yet
  if (!("result.matrix" %in% names(model))) {
    stop("The model hasn't been fitted yet!")
  }

  # Get number of best repition
  if (ncol(model$result.matrix) == 1 ) {
    best_rep <- 1
  } else {
    best_rep <- which.min(model$result.matrix["error",])
  }

  weights <- model$weights[[best_rep]]
  act_name <- attributes(model$act.fct)$type
  if (act_name == "function") {
    stop("You can't use custom activation functions for this package.")
  }

  modules_list <- list()

  for (i in 1:length(weights)) {
    name <- sprintf("Dense_Layer_%s", i)

    # the first row is the bias vector and the rest the weight matrix
    b <- as.vector(weights[[i]][1,])
    w <- t(matrix(weights[[i]][-1,], ncol = length(b)))

    if (i == length(weights) && model$linear.output == TRUE) {
      modules_list[[name]] <- dense_layer(weight = w,
                                          bias = b,
                                          activation_name = "linear",
                                          dtype = dtype)
    }
    else {
      modules_list[[name]] <- dense_layer(weight = w,
                                          bias = b,
                                          activation_name = act_name,
                                          dtype = dtype)
    }
  }

  result <- NULL

  result$model <- analyzed_model(modules_list, dtype)
  result$input_dim <- ncol(model$covariate)
  result$output_dim <- ncol(model$response)
  result$input_names <- model$model.list$variables
  result$output_names <- model$model.list$response

  result

}

implemented_layers <- c("Dense", "Dropout", "InputLayer", "Conv1D", "Conv2D", "Flatten")

#'
#'@title Analyze a keras model
#'@name analyze_neuralnet_model
#'@description
#'This function takes a keras model as input and returns a torch analyzed_model module
#'with the same weights and biases of the original keras model
#'@param model A keras model
#'@return
#'This function returns a \code{result} obejct with attributes \code{result$model},
#' a torch nn_module with the same layers as the input neuralnet module. \code{result$input_dim}
#' is the input dimension of the model, \code{result$output_dim} the output dimensions of
#' the model, \code{result$input_names} is the names of the input variables, \code{result$output_names}
#' is the name of the output variables.
#'
#' @noRd
#'
analyze_keras_model <- function(model, dtype = 'float') {
  if (!requireNamespace("keras")) {
    stop("Please install the 'keras' package.")
  }
  modules_list = list()
  data_format = NULL
  num = 1
  for (layer in model$layers) {
    type <- layer$`__class__`$`__name__`
    name <- paste(type, num, sep = "_")

    checkmate::assertChoice(type, implemented_layers)

    if (type == "Dropout" || type == "InputLayer") {
      message(sprintf("Skipping %s-Layer...", type))
    }
    else if (type == "Dense") {
      modules_list[[name]] <- add_keras_dense(layer, dtype)
      num <- num + 1
    }
    else if (type == "Conv1D") {
      # set the data_format
      if (is.null(data_format)) {
        data_format <- layer$data_format
      }
      modules_list[[name]] <- add_keras_conv1d(layer, dtype)
      num <- num + 1
    }
    else if (type == "Conv2D") {
      # set the data_format
      if (is.null(data_format)) {
        data_format <- layer$data_format
      }
      modules_list[[name]] <- add_keras_conv2d(layer, dtype)
      num <- num + 1
    }
    else if (type == "Flatten") {
      input_dim <- unlist(layer$input_shape)
      output_dim <- unlist(layer$output_shape)

      # in this package only 'channels_first'
      if (layer$data_format == "channels_last") {
        input_dim <- c(rev(input_dim)[1], input_dim[-length(input_dim)])
        output_dim <- c(rev(output_dim)[1], output_dim[-length(output_dim)])
      }

      modules_list[[name]] <- flatten_layer(input_dim, output_dim)
      num <- num + 1
    }
  }
  result <- NULL

  result$model <- analyzed_model(modules_list, dtype)
  input_dim <- unlist(model$input_shape)
  output_dim <- unlist(model$output_shape)
  # in this package only 'channels_first'
  if (is.character(data_format) && data_format == "channels_last") {
    in_channels <- rev(input_dim)[1]
    input_dim[length(input_dim)] <- input_dim[1]
    input_dim[1] <- in_channels

    out_channels <- rev(output_dim)[1]
    output_dim[length(output_dim)] <- output_dim[1]
    output_dim[1] <- out_channels
  }

  result$input_dim <- input_dim
  result$output_dim <- output_dim
  if (length(input_dim) == 1) {
    short_names <- c("X")
  }
  else if (length(input_dim) == 2) {
    short_names <- c("C", "L")
  }
  else {
    short_names <- c("C", "H", "W")
  }
  result$input_names <- mapply(function(x,y) paste0(rep(y, times = x), 1:x), input_dim, short_names, SIMPLIFY = FALSE)
  result$output_names <- lapply(result$output_dim, function(x) paste0(rep("Y", times = x), 1:x))

  result
}



add_keras_dense <- function(layer, dtype) {
  act_name <- layer$activation$`__name__`
  weights <- as.array(t(layer$get_weights()[[1]]))

  if (layer$use_bias) {
    bias <- as.vector(layer$get_weights()[[2]])
  }
  else {
    bias <- rep(0, times = dim(weights)[1])
  }

  dense_layer(weight = weights,
              bias = bias,
              activation_name = act_name,
              dtype = dtype)
}

add_keras_conv1d <- function(layer, dtype) {

  act_name <- layer$get_config()$activation
  filters <- as.numeric(layer$get_config()$filters)
  kernel_size <- as.numeric(unlist(layer$get_config()$kernel_size))
  stride <- as.numeric(unlist(layer$get_config()$strides))
  padding <- layer$get_config()$padding
  dilation <- unlist(layer$get_config()$dilation_rate)

  # input_shape:
  #     channels_first:  [batch_size, in_channels, in_length]
  #     channels_last:   [batch_size, in_length, in_channels]
  input_dim <- unlist(layer$input_shape)
  output_dim <- unlist(layer$output_shape)

  # in this package only 'channels_first'
  if (layer$data_format == "channels_last") {
    input_dim <- c(rev(input_dim)[1], input_dim[-length(input_dim)])
    output_dim <- c(rev(output_dim)[1], output_dim[-length(output_dim)])
  }

  # padding differs in keras and torch
  checkmate::assertChoice(padding, c('valid', 'same'))
  if (padding == "valid") {
    padding <- c(0,0)
  }
  else if (padding == "same") {
    in_length <- input_dim[2]
    out_length <- output_dim[2]
    filter_length <- (kernel_size - 1) * dilation + 1

    if ((in_length %% stride[1]) == 0) {
      pad = max(filter_length - stride[1], 0)
    }
    else {
      pad = max(filter_length - (in_length %% stride[1]), 0)
    }

    pad_left = pad %/% 2
    pad_right = pad - pad_left

    padding <- c(pad_left, pad_right)
  }

  weight <-  as.array(layer$get_weights()[[1]])

  if (layer$use_bias) {
    bias <- as.vector(layer$get_weights()[[2]])
  }
  else {
    bias <- rep(0, times = dim(weight)[3])
  }

  # keras weight format: [kernel_length, in_channels, out_channels]
  # torch weight format: [out_channels, in_channels, kernel_length]
  weight <- aperm(weight, c(3,2,1))

  conv1d_layer(weight = weight,
               bias = bias,
               dim_in = input_dim,
               dim_out = output_dim,
               stride = stride,
               padding = padding,
               dilation = dilation,
               activation_name = act_name,
               dtype = dtype)
}


add_keras_conv2d <- function(layer, dtype) {

  act_name <- layer$get_config()$activation
  filters <- as.numeric(layer$get_config()$filters)
  kernel_size <- as.numeric(unlist(layer$get_config()$kernel_size))
  stride <- as.numeric(unlist(layer$get_config()$strides))
  padding <- layer$get_config()$padding
  dilation <- unlist(layer$get_config()$dilation_rate)

  # input_shape:
  #     channels_first:  [batch_size, in_channels, in_height, in_width]
  #     channels_last:   [batch_size, in_height, in_width, in_channels]
  input_dim <- unlist(layer$input_shape)
  output_dim <- unlist(layer$output_shape)

  # in this package only 'channels_first'
  if (layer$data_format == "channels_last") {
    input_dim <- c(rev(input_dim)[1], input_dim[-length(input_dim)])
    output_dim <- c(rev(output_dim)[1], output_dim[-length(output_dim)])
  }

  # padding differs in keras and torch
  checkmate::assertChoice(padding, c('valid', 'same'))
  if (padding == "valid") {
    padding <- c(0,0,0,0)
  }
  else if (padding == "same") {
    in_height <- input_dim[2]
    in_width <- input_dim[3]
    out_height <- output_dim[2]
    out_width <- output_dim[3]
    filter_height <- (kernel_size[1] - 1 ) * dilation[1] + 1
    filter_width <- (kernel_size[2] - 1) * dilation[2] + 1

    if ((in_height %% stride[1]) == 0) {
      pad_along_height = max(filter_height - stride[1], 0)
    }
    else {
      pad_along_height = max(filter_height - (in_height %% stride[1]), 0)
    }
    if ((in_width %% stride[2]) == 0) {
      pad_along_width = max(filter_width - stride[2], 0)
    }
    else {
      pad_along_width = max(filter_width - (in_width %% stride[2]), 0)
    }

    pad_top = pad_along_height %/% 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width %/% 2
    pad_right = pad_along_width - pad_left

    padding <- c(pad_left, pad_right, pad_top, pad_bottom)
  }

  weight <-  as.array(layer$get_weights()[[1]])

  if (layer$use_bias) {
    bias <- as.vector(layer$get_weights()[[2]])
  }
  else {
    bias <- rep(0, times = dim(weight)[4])
  }
  # Conv2D
  # keras weight format: [kernel_height, kernel_width, in_channels, out_channels]
  # torch weight format: [out_channels, in_channels, kernel_height, kernel_width]
  weight <- aperm(weight, perm = c(4,3,1,2))

  conv2d_layer(weight = weight,
               bias = bias,
               dim_in = input_dim,
               dim_out = output_dim,
               stride = stride,
               padding = padding,
               dilation = dilation,
               activation_name = act_name,
               dtype = dtype)
}
