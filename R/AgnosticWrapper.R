###############################################################################
#                         Super class: AgnosticWrapper
###############################################################################

#'
#' @title Super class for model-agnostic interpretability methods
#' @description This is a super class for all implemented model-agnostic
#' interpretability methods and inherits from the [`InterpretingMethod`]
#' class. Instead of just an object of the [`Converter`] class, any model
#' can now be passed. In contrast to the other model-specific methods in this
#' package, only the prediction function of the model is required, and not
#' the internal details of the model. The following model-agnostic methods
#' are available (all are wrapped by other packages):
#'
#' - *Shapley values* ([`SHAP`]) based on [`fastshap::explain`]
#' - *Local interpretable model-agnostic explanations*  ([`LIME`]) based on
#' [`lime::lime`]
#'
#' @template param-output_idx
#' @template param-channels_first
#' @template param-model-agnostic
#' @template param-data_ref-agnostic
#' @template param-output_label
#' @template param-data-agnostic
#' @template param-output_type-agnostic
#' @template param-pred_fun-agnostic
#' @template param-input_dim-agnostic
#' @template param-input_names-agnostic
#' @template param-output_names-agnostic
#'
#' @export
AgnosticWrapper <- R6Class(
  classname = "AgnosticWrapper",
  inherit = InterpretingMethod,
  public = list(
    #' @field data_orig The individual instances to be explained by the method
    #' (unprocessed!).
    data_orig = NULL,

    #' @description
    #' Create a new instance of the `AgnosticWrapper` R6 class.
    #'
    initialize = function(model, data, data_ref,
                          output_type = NULL,
                          pred_fun = NULL,
                          output_idx = NULL,
                          output_label = NULL,
                          channels_first = TRUE,
                          input_dim = NULL,
                          input_names = NULL,
                          output_names = NULL) {

      # Check for required arguments
      if (missing(model)) stopf("Argument {.arg model} is missing, with no default!")
      if (missing(data)) stopf("Argument {.arg data} is missing, with no default!")

      # If data_ref is missing, we interpret all instances in the given data
      if (missing(data_ref)) data_ref <- data

      # Set the default input shape given by the data
      if (is.null(input_dim)) {
        input_dim <- dim(data_ref)[-1]
        # The input shape is always in the channels first format
        if (!channels_first) {
          input_dim <- c(input_dim[length(input_dim)], input_dim[-length(input_dim)])
        }
      }

      # Get names from data (if possible)
      if (is.null(input_names)) input_names <- names(data_ref)

      # Create converter object for agnostic IML methods
      conv_model <- get_converter(model, data, input_dim, input_names,
                                  output_names, output_type, pred_fun,
                                  channels_first)

      self$converter <- conv_model
      self$channels_first <- channels_first
      self$ignore_last_act <- FALSE
      self$dtype <- "float"

      # Check output indices and labels
      outputs <- check_output_idx(output_idx, self$converter$output_dim,
                                  output_label, self$converter$output_names)
      self$output_idx <- outputs[[1]]
      self$output_label <- outputs[[2]]

      # Save the original data to be explained
      self$data_orig <- data

      # Calculate predictions
      self$preds <- list(self$converter$pred_fun(as.data.frame(data),
                                            input_dim = self$converter$input_dim[[1]]))

      # Save the data
      if (is.data.frame(data)) data <- as.matrix(data)
      self$data <- list(torch_tensor(data))
    }
  )
)


################################################################################
#                     Converter for model-agnostic methods
################################################################################

get_converter <- function(model, data, input_dim = NULL, input_names = NULL,
                          output_names = NULL, output_type = NULL,
                          pred_fun = NULL, channels_first = NULL) {
  # We currently only support models with one input and one output layer
  if (is.list(data) && !is.data.frame(data)) {
    if (length(data) > 1) {
      stopf("The package supports only models with a single input layer for ",
            "the model-agnostic approaches!")
    }
  }

  # Look for pre-implemented methods
  if (inherits(model, "nn"))  {
    conv_model <- get_nn_model(model)
  } else if (is_keras_model(model)) {
    conv_model <- get_keras_model(model)
  } else if (inherits(model, "nn_module") && is_nn_module(model)) {
    conv_model <- get_torch_model(model)
  } else if (inherits(model, "Converter")) {
    conv_model <- list(
      model = model$model,
      input_dim = model$input_dim,
      input_names = model$input_names,
      output_names = model$output_names,
      output_type = get_output_type(model),
      pred_fun = get_pred_fun(model, channels_first)
    )
  } else {
    conv_model <- list(
      model = model,
      input_dim = input_dim,
      input_names = input_names,
      output_names = output_names,
      output_type = output_type,
      pred_fun = pred_fun
    )
  }

  # Overwrite defaults if arguments aren't NULL
  if (!is.null(output_type)) conv_model$output_type <- output_type
  if (!is.null(pred_fun)) conv_model$pred_fun <- pred_fun

  # Do some checks
  cli_check(checkChoice(conv_model$output_type, c("regression", "classification")),
            "output_type")
  cli_check(checkFunction(conv_model$pred_fun, args = "newdata"), "pred_fun")

  # Check input_dim
  if (!is.null(conv_model$input_dim)) {
    if (any(unlist(conv_model$input_dim) != unlist(input_dim))) {
      stopf("There is a missmatch in the calculated input shape ",
            shape_to_char(unlist(conv_model$input_dim)), " and the given ",
            "input shape {.arg input_dim} ",
            shape_to_char(unlist(input_dim)),
            " (or calculated by the given data {.arg data})! Remember ",
            "that {.arg input_dim} has to be in the channels first format. ",
            "If your data is not provided with the channels first, set the ",
            "argument {.arg channels_first} to {.code FALSE}.")
    }
  } else {
    conv_model$input_dim <- list(unlist(input_dim))
  }

  # Check output_dim
  out <- tryCatch(
    conv_model$pred_fun(as.data.frame(data),
                        input_dim = conv_model$input_dim[[1]]),
    error = function(e) {
      e$message <- c(
        paste0(
          "An error occurred when evaluating the {.arg data} using ",
          "{.arg pred_fun}! Remember that the data is converted into a ",
          "{.code data.frame} before it is fed into the model, i.e. the ",
          "{.arg pred_fun} may have to reverse this process. Also note that ",
          "you must specify with the {.arg channels_first} argument if your ",
          "data does not have the channels directly after the batch dimension."),
          "",
          "x" = "Original message:", col_grey(e$message)
      )
      stopf(e$message, use_paste = FALSE)
  })

  # We currently only support models with one input and one output layer
  if (is.list(out) && !is.data.frame(out)) {
    if (length(out) > 1) {
      stopf("The package supports only models with a single output layer for ",
            "the model-agnostic approaches!")
    }
  }

  calc_output_dim <- list(dim(out)[-1])
  if (!is.null(conv_model$output_dim)) {
    if (any(unlist(conv_model$output_dim) != unlist(calc_output_dim))) {
      stopf("There is a missmatch in the calculated output shape ",
            shape_to_char(unlist(conv_model$output_dim)), " and the given ",
            "output shape ", shape_to_char(unlist(calc_output_dim)),
            " extracted of the {.arg model}!")
    }
  } else {
    conv_model$output_dim <- list(unlist(calc_output_dim))
  }

  # Check input names
  if (is.null(input_names)) {
    if (is.null(conv_model$input_names)) {
      conv_model$input_names <-
        set_name_format(get_input_names(conv_model$input_dim))
    }
  } else {
    input_names <- set_name_format(input_names)
    input_names_lenght <- lapply(input_names,
                                 function(x) unlist(lapply(x, length)))
    if (!all_equal(input_names_lenght, conv_model$input_dim)) {
      given <- shape_to_char(input_names_lenght)
      calc <- shape_to_char(conv_model$input_dim)
      stopf(c(
        paste0("Missmatch between the calculated shape of input names and ",
               "given input names:"),
        "*" = paste0("Calculated: '", calc, "'"),
        "*" = paste0("Given: '", given, "'")),
        use_paste = FALSE)
    }
    conv_model$input_names <- input_names
  }

  # Check output names
  if (is.null(output_names)) {
    if (is.null(conv_model$output_names)) {
      conv_model$output_names <-
        set_name_format(get_output_names(conv_model$output_dim))
    }
  } else {
    output_names <- set_name_format(output_names)
    output_names_length <- lapply(output_names,
                                  function(x) unlist(lapply(x, length)))

    if (!all_equal(output_names_length, conv_model$output_dim)) {
      given <- shape_to_char(output_names_length)
      calc <- shape_to_char(conv_model$output_dim)
      stopf(c(
        paste0("Missmatch between the calculated shape of output names and ",
               "given output names:"),
        "*" = paste0("Calculated: '", calc, "'"),
        "*" = paste0("Given: '", given, "'")),
        use_paste = FALSE)
    }
    conv_model$output_names <- output_names
  }

  class(conv_model) <- c("innsight_agnostic_wrapper", class(conv_model))
  conv_model
}

################################################################################
#                             Utility functions
################################################################################

# Package neuralnet
get_nn_model <- function(model) {
  conv_model <- list(
    model = model,
    input_dim = list(ncol(model$covariate)),
    input_names = list(list(
      factor(colnames(model$covariate),
             levels = unique(colnames(model$covariate))))),
    output_dim = list(ncol(model$response)),
    output_names = list(list(
      factor(colnames(model$response),
             levels = unique(colnames(model$response))))),
    output_type =
      ifelse(model$linear.output, "regression", "classification"),
    pred_fun = function(newdata, ...) predict(model, newdata = newdata, ...)
  )
}

# Keras
get_keras_model <- function(model) {

  # Get model input shape
  input_dim <-  list(unlist(model$input_shape))

  # Get data formats
  fun <- function(x) if("data_format" %in% names(x)) x$data_format
  data_formats <- unlist(lapply(model$layers, fun))

  # Convert the shapes to the channels first format
  if (all(data_formats == "channels_last") && !is.null(data_formats)) {
    fun <- function(x) c(x[length(x)], x[-length(x)])
    input_dim <- lapply(input_dim, fun)
  }

  # Get output shape
  output_dim <- list(unlist(model$output_shape))

  # Get output type
  fun <- function(x) model$get_layer(x)$get_config()$activation
  out_types <- unlist(lapply(model$output_names, fun))
  if (all(out_types %in% c("softmax", "sigmoid"))) {
    out_type <- "classification"
  } else if (all(out_types == "linear")) {
    out_type <- "regression"
  } else {
    stopf("You cannot use regression and classification output layers for ",
          "model-agnostic methods!")
  }

  # Build Converter as a list
  conv_model <- list(
    model = model,
    input_dim = input_dim,
    output_dim = output_dim,
    output_type = out_type,
    pred_fun = function(newdata, ...) {
      newdata <- array(unlist(c(newdata)), dim = c(nrow(newdata), unlist(model$input_shape)))
      predict(model, x = newdata, ...)
    }
  )
}

# Torch model
get_torch_model <- function(model) {
  conv_model <- list(
    model = model,
    output_type =
      ifelse(inherits(rev(model$modules)[[1]], c("nn_sigmoid", "nn_softmax")),
             "classification", "regression"),
    pred_fun = function(newdata, input_dim, ...) {
      newdata <- array(unlist(c(newdata)), dim = c(nrow(newdata), input_dim))
      as.array(model$forward(torch_tensor(newdata)))
    }
  )
}

# Get output_type from Converter object
get_output_type <- function(converter) {
  out_nodes <- converter$model$output_nodes
  acts <- unlist(lapply(out_nodes,
                        function(i) converter$model$modules_list[[i]]$activation_name))
  if (all(acts %in% c("softmax", "sigmoid", "tanh", "logistic"))) {
    output_type <- "classification"
  } else {
    output_type <- "regression"
  }

  output_type
}

# Get pred_fun from Converter object
get_pred_fun <- function(converter, channels_first) {
  input_dim <- unlist(converter$input_dim)
  if (!channels_first) {
    input_dim <- c(input_dim[-1], input_dim[1])
  }
  function(newdata, ...) {
    newdata <- array(unlist(c(newdata)), dim = c(nrow(newdata), input_dim))
    res <- converter$model(torch_tensor(newdata), channels_first = channels_first)[[1]]
    as.array(res)
  }
}
