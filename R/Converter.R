#' Converter of an Artificial Neural Network
#'
#' @description
#' This class analyzes a passed neural network and stores its internal
#' structure and the individual layers by converting the entire network into an
#' \code{\link[torch]{nn_module}}. With the help of this converter, many
#' methods for interpreting the behavior of neural networks are provided, which
#' give a better understanding of the whole model or individual predictions.
#' You can use models from the following libraries:
#' * `torch` (\code{\link[torch]{nn_sequential}})
#' * \code{\link[keras]{keras}} (\code{\link[keras]{keras_model}},
#' \code{\link[keras]{keras_model_sequential}}),
#' * \code{\link[neuralnet]{neuralnet}}
#'
#' Furthermore, a model can be passed as a list (see details for more
#' information).
#'
#' @field model The converted neural network based on the torch module
#' [ConvertedModel].
#' @field input_dim A list of the input dimensions of each input layer. Since
#' internally the channels-first format is used for all calculations, the
#' input shapes are already in this format. In addition, the batch
#' dimension isn't included, e.g. for an input layer of shape `c(*,32,32,3)`
#' with channels in the last axis you get `list(c(3,32,32))`.
#' @field input_names A list with the names for each input
#' dimension of the shape as stored in the field `input_dim`.
#' @field output_dim  A list of the output dimensions of each output layer.
#' @field output_names A list with the names for each output dimension of
#' shape as stored in the field `output_dim`.
#' @field model_as_list The model stored in a named list (see details for more
#' information). By default, the entry `model_as_list$layers` is deleted
#' because it may require a lot of memory for large networks. However, with
#' the argument `save_model_as_list` this can be saved anyway.
#'
#' @template details-Converter
#' @template examples-Converter
#'
#' @references
#' * J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
#'  quantifying variable importance in artificial neural networks using
#'  simulated data.} Ecological Modelling 178, p. 389–397
#' * S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#'  classifier decisions by layer-wise relevance propagation.} PLoS ONE 10,
#'  p. 1-46
#' * A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#' * D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @export
Converter <- R6Class("Converter",
  public = list(
    model = NULL,
    input_dim = NULL,
    input_names = NULL,
    output_dim = NULL,
    output_names = NULL,
    model_as_list = NULL,

    ### -----------------------------Initialize--------------------------------
    #' @description
    #' Create a new Converter for a given neural network. When initialized,
    #' the model is inspected, converted as a list and then the a
    #' torch-converted model ([ConvertedModel]) is created and stored in
    #' the field `model`.
    #'
    #' @param model A trained neural network for classification or regression
    #' tasks to be interpreted. Only models from the following types or
    #' packages are allowed: \code{\link[torch]{nn_sequential}},
    #' \code{\link[keras]{keras_model}},
    #' \code{\link[keras]{keras_model_sequential}},
    #' \code{\link[neuralnet]{neuralnet}} or a named list (see details).
    #' @param input_dim The model input dimension excluding the batch
    #' dimension. If there is only one input layer it can be specified as
    #' a vector, otherwise use a list of the shapes of the
    #' individual input layers.\cr
    #' *Note:* This argument is only necessary for `torch::nn_sequential`,
    #' for all others it is automatically extracted from the passed model
    #' and used for internal checks. In addition, the input dimension
    #' `input_dim` has to be in the format channels first.
    #' @param input_names (Optional) A list with the names for each input
    #' dimension excluding the batch dimension, e.g. for a dense layer
    #' with `3` input features use `list(c("X1", "X2", "X3"))` or for a
    #' 1D convolutional layer with signal length `5` and `2` channels use
    #' `list(c("C1", "C2"), c("L1","L2","L3","L4","L5"))`.\cr
    #' *Note:* This argument is optional and otherwise the names are
    #' generated automatically. But if this argument is set, all found
    #' input names in the passed model will be disregarded.
    #' @param output_names (Optional) A list with the names for the output
    #' dimensions exluding the batch dimension,
    #' e.g. for a model with `3` output nodes use
    #' `list(c("Y1", "Y2", "Y3"))`.\cr
    #' *Note:* This argument is optional and otherwise the names are
    #' generated automatically. But if this argument is set, all found
    #' output names in the passed model will be disregarded.
    #' @param save_model_as_list This logical value specifies whether the
    #' passed model should be stored as a list. This list can take
    #' a lot of memory for large networks, so by default the model is not
    #' stored as a list (`FALSE`).
    #' @param dtype The data type for the calculations. Use
    #' either `'float'` for [torch::torch_float] or `'double'` for
    #' [torch::torch_double].
    #'
    #' @return A new instance of the R6 class \code{'Converter'}.
    #'
    initialize = function(model, input_dim = NULL, input_names = NULL,
                          output_names = NULL, dtype = "float",
                          save_model_as_list = FALSE) {
      assertChoice(dtype, c("float", "double"))
      assertLogical(save_model_as_list)

      # Analyze the passed model and store its internal structure in a list of
      # layers

      # Package: Neuralnet ----------------------------------------------------
      if (inherits(model, "nn")) {
        model_as_list <- convert_neuralnet_model(model)
      } else if (is_keras_model(model)) {
        # Package: Keras ------------------------------------------------------
        model_as_list <- convert_keras_model(model)
      } else if (is.list(model)) {
        # Model from list -----------------------------------------------------
        model_as_list <- model
      } else if (inherits(model, "nn_module") && is_nn_module(model)) {
        # Package: Torch ------------------------------------------------------
        model_as_list <- convert_torch(model, input_dim)
      } else {
        # Unknown model class -------------------------------------------------
        stop(sprintf(
          "Unknown module of classes: '%s'!",
          paste(class(model), collapse = "', '")
        ))
      }

      # Check input dimension (if specified)
      if (!is.null(input_dim)) {
        if (!is.list(input_dim)) {
          input_dim <- list(input_dim)
        }
        assertTRUE(length(input_dim) == length(model_as_list$input_dim))
        for (i in seq_along(input_dim)) {
          assertSetEqual(input_dim[[i]], model_as_list$input_dim[[i]],
                         ordered = TRUE)
        }
      }

      # Set input and output names
      if (!is.null(input_names)) {
        model_as_list$input_names <- input_names
      }
      if (!is.null(output_names)) {
        model_as_list$output_names <- output_names
      }

      # Create torch model
      private$create_model_from_list(model_as_list, dtype, save_model_as_list)
    }
  ),
  private = list(
    create_model_from_list = function(model_as_list, dtype = "float",
                                      save_model_as_list = FALSE) {

      #------------------- Do necessary checks --------------------------------
      # Necessary entries in the list:
      #   'layers', 'input_dim', 'input_nodes', 'output_nodes'
      assertSubset("layers", names(model_as_list))
      assertList(model_as_list$layers, min.len = 1)
      # Make sure that 'input_dim' is a list
      if (!is.list(model_as_list$input_dim)) {
        model_as_list$input_dim <- list(model_as_list$input_dim)
      }
      assertList(model_as_list$input_dim, types = "integerish")
      # Save input dimensions as integers
      model_as_list$input_dim <- lapply(model_as_list$input_dim, as.integer)

      # Check whether input and output nodes are given
      assertNumeric(model_as_list$input_nodes,
        lower = 1, null.ok = TRUE,
        upper = length(model_as_list$layers)
      )
      assertNumeric(model_as_list$output_nodes,
        lower = 1, null.ok = TRUE,
        upper = length(model_as_list$layers)
      )

      if (is.null(model_as_list$input_nodes)) {
        warning("Argument 'input_nodes' is unspecified! In the following, ",
                "it is assumed that the first entry in '$layers' is the ",
                "only input layer.", call. = FALSE)
        model_as_list$input_nodes <- 1L
      }
      if (is.null(model_as_list$output_nodes)) {
        warning("Argument 'output_nodes' is unspecified! In the following, ",
                "it is assumed that the last entry in '$layers' is the ",
                "only output layer.", call. = FALSE)
        model_as_list$output_nodes <- length(model_as_list$layers)
      }

      # Optional arguments
      # Make sure that 'output_dim' is a list or NULL
      out_dim <- model_as_list$output_dim
      if (!is.null(out_dim) & !is.list(out_dim)) {
        model_as_list$output_dim <- list(out_dim)
      }
      assertList(model_as_list$output_dim, types = "integerish",
                 null.ok = TRUE)

      # In- and output names are stored as a list of lists. The outer list
      # represents the different in- or output layers of the model and the
      # inner list contains the names for the layer
      if (!is.null(model_as_list$input_names)) {
        model_as_list$input_names <- set_name_format(model_as_list$input_names)
      }
      if (!is.null(model_as_list$output_names)) {
        model_as_list$output_names <-
          set_name_format(model_as_list$output_names)
      }

      #--------------------- Create torch modules -----------------------------

      # Create lists to save the layers (modules list) and the corresponding
      # input (input_layers) and output layers (output_layers). These are
      # necessary to reconstruct the computational graph
      modules_list <- list()
      input_layers <- list()
      output_layers <- list()

      # Create torch modules from 'model_as_list'
      for (i in seq_along(model_as_list$layers)) {
        layer_as_list <- model_as_list$layers[[i]]
        type <- layer_as_list$type
        assertString(type)
        assertChoice(type, c(
          "Flatten", "Skipping", "Dense", "Conv1D", "Conv2D",
          "MaxPooling1D", "MaxPooling2D", "AveragePooling1D",
          "AveragePooling2D", "Concatenate", "Add", "Activation", "Padding"
        ))

        # Get incoming and outgoing layers (as indices) of the current layer
        in_layers <- layer_as_list$input_layers
        out_layers <- layer_as_list$output_layers
        # Check for correct format of the layer indices
        assert(
          checkList(in_layers, types = "integerish"),
          checkIntegerish(in_layers, any.missing = FALSE, null.ok = TRUE)
        )
        assert(
          checkList(out_layers, types = "integerish"),
          checkIntegerish(out_layers, any.missing = FALSE, null.ok = TRUE)
        )
        # Set default for 'out_layers' and 'in_layers'
        if (is.null(in_layers)) {
          warning("Argument '$layers[[", i, "]]$in_layers' is not specified!",
                  "In the following, it is assumed that the ",
                  "layer of index '", max(i - 1, 0), "' is the input ",
                  "layer of the current one.", call. = FALSE)
          in_layers <- max(i - 1, 0)
        }
        if (is.null(out_layers)) {
          num_layers <- length(model_as_list$layers)
          out_layers <- ifelse(i == num_layers, -1, i + 1)
          warning("Argument '$layers[[", i, "]]$out_layers' is not specified!",
                  " In the following, it is assumed that the ",
                  "layer of index '", out_layers, "' is the subsequent ",
                  "layer of the current one.", call. = FALSE)
        }

        # Store the layer indices in the corresponding lists
        input_layers[[i]] <- in_layers
        output_layers[[i]] <- out_layers

        # Create the torch module for the current layer
        layer <- switch(type,
          Flatten = create_flatten_layer(layer_as_list),
          Skipping = create_skipping_layer(layer_as_list),
          Dense = create_dense_layer(layer_as_list, dtype),
          Conv1D = create_conv1d_layer(layer_as_list, dtype, i),
          Conv2D = create_conv2d_layer(layer_as_list, dtype, i),
          MaxPooling1D = create_pooling_layer(layer_as_list, type),
          MaxPooling2D = create_pooling_layer(layer_as_list, type),
          AveragePooling1D = create_pooling_layer(layer_as_list, type),
          AveragePooling2D = create_pooling_layer(layer_as_list, type),
          Concatenate = create_concatenate_layer(layer_as_list),
          Add = create_add_layer(layer_as_list),
          Padding = create_padding_layer(layer_as_list),
          Activation = create_activation_layer(layer_as_list)
        )

        # Set a name for the layer
        modules_list[[paste(type, i, sep = "_")]] <- layer
      }

      #-------- Create computational graph and ConvertedModel ------------------
      # Create graph
      graph <- create_structered_graph(
        input_layers, output_layers,
        model_as_list
      )

      # Check modules and graph and register input and output shapes for each
      # layer
      tmp <-
        check_and_register_shapes(modules_list, graph, model_as_list, dtype)

      # Create torch model of class 'ConvertedModel'
      model <- ConvertedModel(tmp$modules_list, graph,
                              model_as_list$input_nodes,
                              model_as_list$output_nodes,
                              dtype = dtype
      )

      #---------------- Check if the converted model is correct ---------------
      # Check output dimensions
      if (!is.null(model_as_list$output_dim) &&
        !all_equal(model_as_list$output_dim, tmp$calc_output_shapes)) {
        calc <- shape_to_char(tmp$calc_output_shapes)
        given <- shape_to_char(model_as_list$output_dim)
        stop(
          "Missmatch between the calculated and given model output shapes:\n",
          "Calculated: ", calc, "\n",
          "Given:      ", given
        )
      }
      model_as_list$output_dim <- tmp$calc_output_shapes

      # Check for classification output
      output_dims <- unlist(lapply(model_as_list$output_dim, length))
      if (any(output_dims != 1)) {
        stop(
          "This package only allows models with classification or regression ",
          "output, i.e. the model output dimension has to be one. ",
          "But your model has an output dimension of '",
          max(output_dims), "'!"
        )
      }

      # Check input names
      if (is.null(model_as_list$input_names)) {
        model_as_list$input_names <- get_input_names(model_as_list$input_dim)
      } else {
        input_names <- model_as_list$input_names
        input_names_lenght <- lapply(input_names,
                                     function(x) unlist(lapply(x, length)))

        if (!all_equal(input_names_lenght, model_as_list$input_dim)) {
          given <- shape_to_char(input_names_lenght)
          calc <- shape_to_char(model_as_list$input_dim)
          stop(
            "Missmatch between the calculated shape of input names and ",
            "given input names:\n",
            "Calculated: ", calc, "\n",
            "Given:      ", given
          )
        }
      }

      # Check output names
      if (is.null(model_as_list$output_names)) {
        model_as_list$output_names <-
          get_output_names(model_as_list$output_dim)
      } else {
        output_names <- model_as_list$output_names
        output_names_length <- lapply(output_names,
                                      function(x) unlist(lapply(x, length)))

        if (!all_equal(output_names_length, model_as_list$output_dim)) {
          given <- shape_to_char(output_names_length)
          calc <- shape_to_char(model_as_list$output_dim)
          stop(
            "Missmatch between the calculated shape of output names and ",
            "given output names:\n",
            "Calculated: ", calc, "\n",
            "Given:      ", given
          )
        }
      }

      #------------------------- Saving and clean up --------------------------

      self$model <- model
      self$model$reset()

      if (save_model_as_list) {
        self$model_as_list <- model_as_list
      }

      self$input_dim <- model_as_list$input_dim
      self$input_names <- model_as_list$input_names
      self$output_dim <- model_as_list$output_dim
      self$output_names <- model_as_list$output_names
    }
  )
)



###############################################################################
#                             Create Layers
###############################################################################

# Dense Layer -----------------------------------------------------------------
create_dense_layer <- function(layer_as_list, dtype) {
  # Check for required keys
  assertSubset(c("weight", "bias"), names(layer_as_list))

  # Get arguments
  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out
  weight <- layer_as_list$weight
  bias <- layer_as_list$bias

  # Check arguments
  assertIntegerish(dim_in, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)
  assertArray(weight, mode = "numeric", d = 2)
  assertVector(bias, len = dim_out)

  dense_layer(weight, bias, dim_in, dim_out, dtype = dtype)
}

# Conv1D Layer ----------------------------------------------------------------
create_conv1d_layer <- function(layer_as_list, dtype, i) {
  # Check for required keys
  assertSubset(c("weight", "bias"), names(layer_as_list))

  # Get arguments
  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out
  weight <- layer_as_list$weight
  bias <- layer_as_list$bias
  stride <- layer_as_list$stride
  dilation <- layer_as_list$dilation
  padding <- layer_as_list$padding

  # Check arguments
  assertIntegerish(dim_in, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)
  assertArray(weight, mode = "numeric", d = 3)
  assertVector(bias, len = dim_out[1])
  assertInt(stride, null.ok = TRUE)
  assertInt(dilation, null.ok = TRUE)
  assertNumeric(padding, null.ok = TRUE, lower = 0)

  # Set default arguments
  if (is.null(stride)) stride <- 1
  if (is.null(dilation)) dilation <- 1
  if (is.null(padding)) padding <- c(0, 0)

  if (length(padding) == 1) {
    padding <- rep(padding, 2)
  } else if (length(padding) != 2) {
    stop(paste0(
      "Expected a padding vector in 'model_as_list$layers[[", i, "]] ",
      "of length:\n", "   - 1: same padding for each side\n",
      "   - 2: first value: padding for left side; second value: ",
      "padding for right side\n But your length: ", length(padding)
    ))
  }

  conv1d_layer(weight, bias, dim_in, dim_out, stride, padding, dilation,
               dtype = dtype)
}

# Conv2D Layer ----------------------------------------------------------------
create_conv2d_layer <- function(layer_as_list, dtype, i) {
  assertSubset(c("weight", "bias"), names(layer_as_list))

  # Get arguments
  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out
  weight <- layer_as_list$weight
  bias <- layer_as_list$bias
  stride <- layer_as_list$stride
  dilation <- layer_as_list$dilation
  padding <- layer_as_list$padding

  # Check arguments
  assertIntegerish(dim_in, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)
  assertArray(weight, mode = "numeric", d = 4)
  assertVector(bias, len = dim_out[1])
  assertNumeric(stride, null.ok = TRUE, lower = 1)
  assertNumeric(dilation, null.ok = TRUE, lower = 1)
  assertNumeric(padding, null.ok = TRUE, lower = 0)

  # Set default arguments
  if (is.null(stride)) stride <- c(1, 1)
  if (is.null(dilation)) dilation <- c(1, 1)
  if (is.null(padding)) padding <- c(0, 0, 0, 0)

  if (length(padding) == 1) {
    padding <- rep(padding, 4)
  } else if (length(padding) == 2) {
    padding <- rep(padding, each = 2)
  } else if (length(padding) != 4) {
    stop(paste0(
      "Expected a padding vector in 'model_as_list$layers[[", i, "]] ",
      "of length:\n", "   - 1: same padding on each side\n",
      "   - 2: first value: pad_left and pad_right; second value: pad_top ",
      "and pad_bottom\n",
      "   - 4: (pad_left, pad_right, pad_top, pad_bottom)\n",
      "But your length: ", length(padding)
    ))
  }
  if (length(stride) == 1) {
    stride <- rep(stride, 2)
  } else if (length(stride) != 2) {
    stop(paste0(
      "Expected a stride vector in 'model_as_list$layers[[", i, "]] ",
      "of length:\n", "   - 1: same stride for image heigth and width\n",
      "   - 2: first value: strides for height; second value: strides ",
      "for width\n", "But your length: ", length(stride)
    ))
  }
  if (length(dilation) == 1) {
    dilation <- rep(dilation, 2)
  } else if (length(dilation) != 2) {
    stop(paste0(
      "Expected a dilation vector in 'model_as_list$layers[[", i, "]] ",
      "of length:\n", "   - 1: same dilation for image heigth and width\n",
      "   - 2: first value: dilation for height; second value: dilation ",
      "for width\n", "But your length: ", length(dilation)
    ))
  }

  conv2d_layer(weight, bias, dim_in, dim_out, stride, padding, dilation,
               dtype = dtype)
}

# Pooling Layers -------------------------------------------------------------
create_pooling_layer <- function(layer_as_list, type) {
  # Check for required keys
  assertSubset(c("kernel_size"), names(layer_as_list))


  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out
  kernel_size <- layer_as_list$kernel_size
  strides <- layer_as_list$strides

  assertVector(kernel_size, min.len = 1, max.len = 2)
  assertVector(strides, min.len = 1, max.len = 2, null.ok = TRUE)
  assertIntegerish(dim_in, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)

  if (is.null(strides)) strides <- kernel_size

  if (type == "MaxPooling1D") {
    assertSetEqual(length(kernel_size), 1)
    assertSetEqual(length(strides), 1)

    layer <- max_pool1d_layer(kernel_size, dim_in, dim_out, strides)
  } else if (type == "AveragePooling1D") {
    assertSetEqual(length(kernel_size), 1)
    assertSetEqual(length(strides), 1)

    layer <- avg_pool1d_layer(kernel_size, dim_in, dim_out, strides)
  } else if (type == "MaxPooling2D") {
    assertSetEqual(length(kernel_size), 2)
    assertSetEqual(length(strides), 2)

    layer <- max_pool2d_layer(kernel_size, dim_in, dim_out, strides)
  } else if (type == "AveragePooling2D") {
    assertSetEqual(length(kernel_size), 2)
    assertSetEqual(length(strides), 2)

    layer <- avg_pool2d_layer(kernel_size, dim_in, dim_out, strides)
  }

  layer
}

# Padding Layer ---------------------------------------------------------------
create_padding_layer <- function(layer_as_list) {
  # Check for required keys
  assertSubset(c("pad"), names(layer_as_list))


  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out
  pad <- layer_as_list$pad
  mode <- layer_as_list$mode
  fill <- layer_as_list$fill

  assertIntegerish(dim_in, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(pad)
  assertChoice(mode, c("constant", "reflect", "replicate", "circular"))
  assertNumber(fill)

  padding_layer(pad, dim_in, dim_out, mode, fill)
}

# Activation Layer ------------------------------------------------------------
create_activation_layer <- function(layer_as_list) {
  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out

  assertIntegerish(dim_in, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)

  activation_layer(layer_as_list$act_name, dim_in, dim_out)
}

# Flatten Layer ---------------------------------------------------------------
create_flatten_layer <- function(layer_as_list) {
  # Get arguments
  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out
  start_dim <- layer_as_list$start_dim
  end_dim <- layer_as_list$end_dim

  # Check arguments
  assertIntegerish(dim_in, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)
  assertInt(start_dim, null.ok = TRUE)
  assertInt(end_dim, null.ok = TRUE)

  # Set default arguments
  if (is.null(start_dim)) start_dim <- 2L
  if (is.null(end_dim)) end_dim <- -1L

  flatten_layer(dim_in, dim_out, start_dim, end_dim)
}

# Concatenate Layer -----------------------------------------------------------
create_concatenate_layer <- function(layer_as_list) {
  # Get arguments
  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out
  dim <- layer_as_list$axis

  # Check arguments
  assertList(dim_in, null.ok = TRUE, types = "integerish")
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)
  assertInt(dim)

  concatenate_layer(dim, dim_in, dim_out)
}

# Add Layer -------------------------------------------------------------------
create_add_layer <- function(layer_as_list) {
  # Get arguments
  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out

  # Check arguments
  assertList(dim_in, null.ok = TRUE, types = "integerish")
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)

  add_layer(dim_in, dim_out)
}

# Skipping Layer --------------------------------------------------------------
create_skipping_layer <- function(layer_as_list) {
  # Get arguments
  dim_in <- layer_as_list$dim_in
  dim_out <- layer_as_list$dim_out

  # Check arguments
  assertIntegerish(dim_in, min.len = 1, max.len = 3, null.ok = TRUE)
  assertIntegerish(dim_out, min.len = 1, max.len = 3, null.ok = TRUE)

  skipping_layer(dim_in, dim_out)
}


###############################################################################
#                                 Utils
###############################################################################

create_structered_graph <- function(input_layers, output_layers,
                                    model_as_list) {
  current_nodes <- as.list(model_as_list$input_nodes)
  upper_nodes <- unique(unlist(lapply(
    unlist(current_nodes),
    function(x) output_layers[[x]]
  )))
  idx <- rep(
    seq_along(current_nodes),
    unlist(lapply(
      current_nodes,
      function(x) length(output_layers[[x]])
    ))
  )
  all_used_nodes <- NULL

  graph <- list()
  n <- 1
  added <- 0
  freq <- unlist(lapply(current_nodes, function(x) length(output_layers[[x]])))
  tmp_nodes <- rep(list(0), length(current_nodes))
  for (node in current_nodes) {
    graph[[n]] <- list(
      current_nodes = tmp_nodes, used_idx = n + added,
      used_node = node,
      times = length(output_layers[[node]])
    )
    tmp_nodes[[n + added]] <- node
    all_used_nodes <- c(all_used_nodes, node)
    times <- freq[[n]] - 1
    tmp_nodes <- append(tmp_nodes, rep(list(node), times), n + added)
    added <- added + times
    n <- n + 1
  }

  current_nodes <- current_nodes[idx]

  is_contained <- function(a, b) {
    tmp_a <- rle(sort(a))
    tmp_b <- rle(sort(b))

    if (!all(tmp_a$values %in% tmp_b$values)) {
      res <- FALSE
    } else {
      res <- unlist(lapply(tmp_a$values, function(value) {
        idx <- which(tmp_b$values == value)
        tmp_a$lengths[which(tmp_a$values == value)] <= tmp_b$lengths[idx]
      }))
      res <- all(res)
    }

    res
  }

  while (any(unlist(upper_nodes) != -1)) {
    next_node <- FALSE

    for (upper_node in upper_nodes) {
      if (next_node == FALSE && upper_node != -1) {
        if (is_contained(input_layers[[upper_node]], unlist(current_nodes))) {
          used_idx <- input_layers[[upper_node]]
          tmp_list <- current_nodes
          idx_list <- c()
          for (used in used_idx) {
            id <- which(used == tmp_list)[[1]]
            tmp_list[[id]] <- -1
            idx_list <- c(idx_list, id)
          }
          used_idx <- idx_list
          used_node <- upper_node
          next_node <- TRUE

          break
        }
      }
    }
    num_replicates <- length(output_layers[[used_node]])
    all_used_nodes <- c(all_used_nodes, used_node)
    graph[[n]] <- list(
      current_nodes = current_nodes, used_idx = used_idx,
      used_node = used_node, times = num_replicates
    )
    current_nodes <- current_nodes[-used_idx]
    current_nodes <- append(current_nodes, rep(list(used_node),
                                               num_replicates),
      after = min(used_idx) - 1
    )
    upper_nodes <-
      unique(unlist(lapply(
        unlist(current_nodes),
        function(x) output_layers[[x]]
      )))
    upper_nodes <- setdiff(upper_nodes, all_used_nodes)
    n <- n + 1
  }

  graph
}



check_and_register_shapes <- function(modules_list, graph, model_as_list,
                                      dtype) {
  if (dtype == "float") {
    x <- lapply(model_as_list$input_dim,
                function(shape) torch_randn(c(1, shape)))
  } else {
    x <- lapply(
      model_as_list$input_dim,
      function(shape) torch_randn(c(1, shape), dtype = torch_double())
    )
  }


  for (step in graph) {
    input <- x[step$used_idx]
    if (length(input) == 1) {
      input <- input[[1]]
      calculated_input_shape <- input$shape[-1]
    } else {
      calculated_input_shape <- lapply(input,
                                       function(tensor) tensor$shape[-1])
    }

    # Check input shape
    given_input_shape <- model_as_list$layers[[step$used_node]]$dim_in
    if (!is.null(given_input_shape) &&
      !all_equal(calculated_input_shape, given_input_shape)) {
      given <- shape_to_char(given_input_shape)
      calc <- shape_to_char(calculated_input_shape)

      stop(
        "Missmatch between the calculated and given input shape for layer ",
        "index '", step$used_node, "':\n",
        "Calculated: ", calc, "\n",
        "Given:      ", given
      )
    }

    # Register input shape for this layer
    modules_list[[step$used_node]]$input_dim <- calculated_input_shape

    # Check layer
    tryCatch(
      out <- modules_list[[step$used_node]](input),
      error = function(e) {
        layer_as_list <- model_as_list$layers[[step$used_node]]
        e$message <-
          paste0(
            "Could not create layer of index '", step$used_node, "' of type '",
            layer_as_list$type, "'. Maybe you used incorrect parameters or a ",
            "wrong dimension order of your weight matrix. The weight matrix ",
            "for a dense layer has to be stored as an array of shape ",
            "(dim_in, dim_out), for a 1D-convolutional as an array of ",
            "(out_channels, in_channels, kernel_length) and for ",
            "2D-convolutional (out_channels, in_channels, kernel_height, ",
            "kernel_width).\n\n",
            "Your weight dimension: ",
            ifelse(is.null(layer_as_list$weight), "(no weights available)",
              paste0("(", paste(dim(layer_as_list$weight), collapse = ","),
                     ")")
            ),
            "\n", "Your layer input shape: ",
            paste0("(", paste("*", input$shape[-1], collapse = ","), ")"),
            "\n\n\n Original message:\n", e
          )
        stop(e)
      }
    )

    # Check output shape
    calculated_output_shape <- out$shape[-1]
    given_output_shape <- model_as_list$layers[[step$used_node]]$dim_out
    if (!is.null(given_output_shape) &&
      !all(calculated_output_shape == given_output_shape)) {
      given <- paste0("(*,", paste(given_output_shape, collapse = ","), ")")
      calc <- paste0("(*,", paste(calculated_output_shape, collapse = ","), ")")

      stop(
        "Missmatch between the calculated and given output shape for ",
        "layer index '", step$used_node, "':\n",
        "Calculated: ", calc, "\n",
        "Given:      ", given
      )
    }

    # Register output shape for this layer
    modules_list[[step$used_node]]$output_dim <- calculated_output_shape


    x <- x[-step$used_idx]
    x <- append(x, rep(list(out), step$times),
      after = min(step$used_idx) - 1
    )
  }

  # Order outputs
  output_nodes <- step$current_nodes
  output_nodes[[step$used_idx]] <- step$used_node
  output_nodes <- unlist(output_nodes)
  x <- x[match(model_as_list$output_nodes, output_nodes)]

  # Get output shapes
  calculated_output_shapes <- lapply(x, function(tensor) tensor$shape[-1])

  list(modules_list = modules_list,
       calc_output_shapes = calculated_output_shapes)
}


shape_to_char <- function(shape, use_batch = TRUE) {
  if (!is.list(shape)) {
    if (use_batch) {
      shape_as_char <- paste0("(*,", paste(shape, collapse = ","), ")")
    } else {
      shape_as_char <- paste0("(", paste(shape, collapse = ","), ")")
    }
  } else {
    if (use_batch) {
      shape_as_char <- lapply(
        shape,
        function(x) {
          paste0("(*,", paste(x, collapse = ","), ")")
        }
      )
    } else {
      shape_as_char <- lapply(
        shape,
        function(x) {
          paste0("(", paste(x, collapse = ","), ")")
        }
      )
    }
    shape_as_char <- lapply(
      seq_along(shape_as_char),
      function(i) {
        paste0("[[", i, "]] ", shape_as_char[[i]])
      }
    )
    shape_as_char <- paste(unlist(shape_as_char))
  }

  shape_as_char
}


get_input_names <- function(input_dims) {
  lapply(input_dims, function(input_dim) {
    if (length(input_dim) == 1) {
      short_names <- c("X")
    } else if (length(input_dim) == 2) {
      short_names <- c("C", "L")
    } else if (length(input_dim) == 3) {
      short_names <- c("C", "H", "W")
    } else {
      stop(
        "Too many input dimensions. This package only allows model ",
        "inputs with '1', '2' or '3' dimensions and not '",
        length(input_dim), "'!"
      )
    }

    mapply(function(x, y) paste0(rep(y, times = x), 1:x),
      input_dim,
      short_names,
      SIMPLIFY = FALSE
    )
  })
}

get_output_names <- function(output_dims) {
  lapply(output_dims, function(output_dim) {
    lapply(
      output_dim,
      function(x) paste0(rep("Y", times = x), 1:x)
    )
  })
}

is_keras_model <- function(model) {
  inherits(model, c(
    "keras.engine.sequential.Sequential",
    "keras.engine.functional.Functional",
    "keras.engine.training.Model"
  ))
}

set_name_format <- function(in_or_out_names) {
  if (is.list(in_or_out_names)) {
    if (!is.list(in_or_out_names[[1]])) {
      in_or_out_names <- list(in_or_out_names)
    }
  } else {
    in_or_out_names <- list(list(in_or_out_names))
  }
  # Do the checks
  for (i in seq_along(in_or_out_names)) {
    for (j in seq_along(in_or_out_names[[i]])) {
      assertCharacter(in_or_out_names[[i]][[j]], null.ok = TRUE)
    }
  }

  in_or_out_names
}

convert_torch <- function(model, input_dim) {
  if (inherits(model, "nn_sequential")) {
    if (!is.list(input_dim)) {
      input_dim <- list(input_dim)
    }
    for (i in seq_along(input_dim)) {
      if (!testNumeric(input_dim[[i]], lower = 1)) {
        stop(
          "For a 'torch' model, you have to specify the argument ",
          "'input_dim'!"
        )
      }
    }
    model_as_list <- convert_torch_sequential(model)
    model_as_list$input_dim <- input_dim
  } else {
    stop("At the moment, only sequential models are allowed!")
  }

  model_as_list
}

all_equal <- function(x,y) {
  if (identical(length(x), length(y))) {
    error <- all(unlist(lapply(seq_along(x), function(i) x[[i]] == y[[i]])))
  } else {
    error <- FALSE
  }

  error
}
