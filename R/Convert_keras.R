implemented_layers_keras <- c(
  "Dense", "Dropout", "InputLayer", "Conv1D", "Conv2D", "Flatten",
  "MaxPooling1D", "MaxPooling2D", "AveragePooling1D", "AveragePooling2D",
  "Concatenate", "Add"
)


###############################################################################
#                           Convert Keras Model
###############################################################################

convert_keras_model <- function(model) {

  # Define parameters for the data format and the layer index
  data_format <- NULL
  n <- 1
  add_n <- 0

  # Get layer names and reconstruct graph
  names <- unlist(lapply(model$layers, FUN = function(x) x$name))
  sequential <- inherits(model, "keras.engine.sequential.Sequential")
  graph <- keras_reconstruct_graph(model$get_config(), sequential)

  # Declare list for the list-converted layers
  model_as_list <- list()

  for (layer in model$layers) {
    # Get the layer type and check whether it is implemented
    type <- layer$`__class__`$`__name__`
    assertChoice(type, implemented_layers_keras)

    # Convert the layer to a list based on its type
    # Note: It is assumed that the same data format was used for all
    # convolutional layers!
    layer_list <-
      switch(type,
        InputLayer = convert_keras_skipping(type),
        Dropout = convert_keras_skipping(type),
        Dense = convert_keras_dense(layer),
        Conv1D = {
          # check for consistent data format
          data_format <- check_consistent_data_format(
            data_format, layer$data_format
          )
          convert_keras_convolution(layer, type)
        },
        Conv2D = {
          # check for consistent data format
          data_format <- check_consistent_data_format(
            data_format, layer$data_format
          )
          convert_keras_convolution(layer, type)
        },
        Flatten = convert_keras_flatten(layer),
        MaxPooling1D = {
          # check for consistent data format
          data_format <- check_consistent_data_format(
            data_format, layer$data_format
          )
          convert_keras_pooling(layer, type)
        },
        MaxPooling2D = {
          # check for consistent data format
          data_format <- check_consistent_data_format(
            data_format, layer$data_format
          )
          convert_keras_pooling(layer, type)
        },
        AveragePooling1D = {
          # check for consistent data format
          data_format <- check_consistent_data_format(
            data_format, layer$data_format
          )
          convert_keras_pooling(layer, type)
        },
        AveragePooling2D = {
          # check for consistent data format
          data_format <- check_consistent_data_format(
            data_format, layer$data_format
          )
          convert_keras_pooling(layer, type)
        },
        Concatenate = convert_keras_concatenate(layer),
        Add = convert_keras_add(layer)
      )

    # Define the incoming and outgoing layers of this layer
    # Thereby means '0' Input-Node and '-1' Output-Node
    if (length(layer_list) == 1) {
      layer_list$input_layers <- graph[[n]]$input_layers
      layer_list$output_layers <- graph[[n]]$output_layers
    } else {
      in_layer <- graph[[n]]$input_layers
      out_layer <-
      for (i in seq_along(layer_list)) {

      }
    }

    # Set name of this layer and save it
    model_as_list[[n]] <- layer_list

    n <- n + 1
  }

  # Get in- and output shape of the model
  input_dim <- model$input_shape
  output_dim <- model$output_shape
  if (length(model$input_names) == 1) {
    input_dim <- list(unlist(input_dim))
  } else {
    input_dim <- lapply(input_dim, unlist)
  }
  if (length(model$output_names) == 1) {
    output_dim <- list(unlist(output_dim))
  } else {
    output_dim <- lapply(output_dim, unlist)
  }

  # In this package only 'channels_first' is allowed, i.e. convert the format
  # to 'channels_first' if necessary
  for (i in seq_along(input_dim)) {
    in_dim <- input_dim[[i]]
    if (length(in_dim) > 1 && is.character(data_format) &&
      data_format == "channels_last") {
      input_dim[[i]] <- c(rev(in_dim)[1], in_dim[-length(in_dim)])
    }
  }
  for (i in seq_along(output_dim)) {
    out_dim <- output_dim[[i]]
    if (length(out_dim) > 1 && is.character(data_format) &&
      data_format == "channels_last") {
      output_dim[[i]] <- c(rev(out_dim)[1], out_dim[-length(out_dim)])
    }
  }

  # Get input and output nodes
  input_names <- model$input_names
  if (any(grepl("_input", input_names))) {
    input_names <- c(input_names, gsub("_input", "", input_names))
  }
  input_nodes <- match(input_names, names)
  input_nodes <- input_nodes[!is.na(input_nodes)]
  output_nodes <- match(model$output_names, names)

  # Return the list-converted model with in- and output shapes and nodes
  list(
    input_dim = input_dim,
    input_nodes = input_nodes,
    output_dim = output_dim,
    output_nodes = output_nodes,
    layers = model_as_list
  )
}

###############################################################################
#                           Convert Keras Layers
###############################################################################

# Dense Layer -----------------------------------------------------------------

convert_keras_dense <- function(layer) {
  act_name <- layer$activation$`__name__`
  weights <- as.array(t(layer$get_weights()[[1]]))

  if (layer$use_bias) {
    bias <- as.vector(layer$get_weights()[[2]])
  } else {
    bias <- rep(0, times = dim(weights)[1])
  }

  list(
    type = "Dense",
    weight = weights,
    bias = bias,
    activation_name = act_name,
    dim_in = dim(weights)[2],
    dim_out = dim(weights)[1]
  )
}

# Convolution Layer -----------------------------------------------------------

convert_keras_convolution <- function(layer, type) {
  act_name <- layer$get_config()$activation
  kernel_size <- as.integer(unlist(layer$get_config()$kernel_size))
  stride <- as.integer(unlist(layer$get_config()$strides))
  padding <- layer$get_config()$padding
  dilation <- as.integer(unlist(layer$get_config()$dilation_rate))

  # input_shape:
  #     channels_first:  [batch_size, in_channels, in_length]
  #     channels_last:   [batch_size, in_length, in_channels]
  input_dim <- as.integer(unlist(layer$input_shape))
  output_dim <- as.integer(unlist(layer$output_shape))

  # in this package only 'channels_first'
  if (layer$data_format == "channels_last") {
    input_dim <- move_channels_first(input_dim)
    output_dim <- move_channels_first(output_dim)
  }

  # padding differs in keras and torch
  assertChoice(padding, c("valid", "same"))
  if (padding == "valid") {
    padding <- rep(c(0L, 0L), length(kernel_size))
  } else if (padding == "same") {
    padding <- get_same_padding(input_dim, kernel_size, dilation, stride)
  }

  weight <- as.array(layer$get_weights()[[1]])

  if (layer$use_bias) {
    bias <- as.vector(layer$get_weights()[[2]])
  } else {
    bias <- rep(0, times = dim(weight)[length(dim(weight))])
  }

  # Conv1D
  # keras weight format: [kernel_length, in_channels, out_channels]
  # torch weight format: [out_channels, in_channels, kernel_length]
  # Conv2D
  # keras weight format:
  #   [kernel_height, kernel_width, in_channels, out_channels]
  # torch weight format:
  #   [out_channels, in_channels, kernel_height, kernel_width]
  if (length(dim(weight)) == 3) {
    weight <- aperm(weight, c(3, 2, 1))
  } else {
    weight <- aperm(weight, c(4, 3, 1, 2))
  }

  list(
    type = type,
    weight = weight,
    bias = bias,
    activation_name = act_name,
    dim_in = input_dim,
    dim_out = output_dim,
    stride = stride,
    padding = padding,
    dilation = dilation
  )
}

# Pooling Layer ---------------------------------------------------------------

convert_keras_pooling <- function(layer, type) {
  input_dim <- unlist(layer$input_shape)
  output_dim <- unlist(layer$output_shape)
  kernel_size <- unlist(layer$pool_size)
  strides <- unlist(layer$strides)

  if (layer$padding != "valid") {
    stop(sprintf(
      "Padding mode '%s' is not implemented yet!",
      layer$padding
    ))
  }

  # in this package only 'channels_first'
  if (layer$data_format == "channels_last") {
    input_dim <- move_channels_first(input_dim)
    output_dim <- move_channels_first(output_dim)
  }

  list(
    type = type,
    dim_in = input_dim,
    dim_out = output_dim,
    kernel_size = kernel_size,
    strides = strides
  )
}

# Flatten Layer ---------------------------------------------------------------

convert_keras_flatten <- function(layer) {
  input_dim <- unlist(layer$input_shape)
  output_dim <- unlist(layer$output_shape)

  # in this package only 'channels_first'
  if (layer$data_format == "channels_last") {
    input_dim <- move_channels_first(input_dim)
  }

  list(
    type = "Flatten",
    start_dim = 2,
    end_dim = -1,
    dim_in = input_dim,
    dim_out = output_dim
  )
}

# Concatenate Layer -----------------------------------------------------------

convert_keras_concatenate <- function(layer) {
  num_input_dims <- lapply(layer$input_shape, function(x) length(unlist(x)))
  if (any(unlist(num_input_dims) > 1)) {
    warning(
      "I assume that the concatenations axis points to the channel axis.",
      " Otherwise, an error can be thrown in the further process."
    )
  }
  list(
    type = "Concatenate",
    axis = layer$axis,
    dim_in = lapply(layer$input_shape, unlist),
    dim_out = unlist(layer$output_shape)
  )
}

# Add Layer -------------------------------------------------------------------

convert_keras_add <- function(layer) {
  list(
    type = "Add",
    dim_in = lapply(layer$input_shape, unlist),
    dim_out = unlist(layer$output_shape)
  )
}

# Skipping Layers -------------------------------------------------------------

convert_keras_skipping <- function(type) {
  message(sprintf("Skipping %s ...", type))

  list(type = "Skipping")
}

# utils -----------------------------------------------------------------------


get_same_padding <- function(input_dim, kernel_size, dilation, stride) {
  if (length(kernel_size) == 1) {
    in_length <- input_dim[2]
    filter_length <- (kernel_size - 1) * dilation + 1

    if ((in_length %% stride[1]) == 0) {
      pad <- max(filter_length - stride[1], 0)
    } else {
      pad <- max(filter_length - (in_length %% stride[1]), 0)
    }

    pad_left <- pad %/% 2
    pad_right <- pad - pad_left

    padding <- as.integer(c(pad_left, pad_right))
  } else if (length(kernel_size) == 2) {
    in_height <- input_dim[2]
    in_width <- input_dim[3]
    filter_height <- (kernel_size[1] - 1) * dilation[1] + 1
    filter_width <- (kernel_size[2] - 1) * dilation[2] + 1

    if ((in_height %% stride[1]) == 0) {
      pad_along_height <- max(filter_height - stride[1], 0)
    } else {
      pad_along_height <- max(filter_height - (in_height %% stride[1]), 0)
    }
    if ((in_width %% stride[2]) == 0) {
      pad_along_width <- max(filter_width - stride[2], 0)
    } else {
      pad_along_width <- max(filter_width - (in_width %% stride[2]), 0)
    }

    pad_top <- pad_along_height %/% 2
    pad_bottom <- pad_along_height - pad_top
    pad_left <- pad_along_width %/% 2
    pad_right <- pad_along_width - pad_left

    padding <- as.integer(c(pad_left, pad_right, pad_top, pad_bottom))
  }

  padding
}


keras_reconstruct_graph <- function(config, sequential = TRUE) {
  # Create list with layer names
  names <- NULL
  for (layer in config$layers)  {
    times <- sum(
      has_padding(layer),
      if (sequential) TRUE else length(layer$inbound_nodes) > 0,
      has_activation(layer)
    )
    times <- max(times, 1)
    names <- c(names, rep(layer$name, times))
  }

  if (sequential) {
    graph <- lapply(
      seq_along(names),
      function(i) list(input_layers = i - 1, output_layers = i + 1)
    )
    graph[[length(names)]]$output_layers <- -1
  } else {
    # Create empty graph
    graph <- lapply(
      seq_along(names),
      function(a) list(input_layers = NULL, output_layers = NULL)
    )

    for (layer in config$layers) {
      if (length(layer$inbound_nodes) > 0) {
        in_layer_names <- unlist(
          lapply(layer$inbound_nodes[[1]], function(x) x[[1]])
        )
        in_idx <- length(names) + 1 - match(in_layer_names, rev(names))

        # Register output layers for all input layers of this layer
        first_idx <- match(layer$name, names)
        for (idx in in_idx) {
          graph[[idx]]$output_layers <-
            c(graph[[idx]]$output_layers, first_idx)
        }
      } else {
        in_idx <- 0
      }
      # Register input layers and output layers for the current layer
      last_element <- rev(which(names == layer$name))[1]
      for (idx in which(names == layer$name)) {
        graph[[idx]]$input_layers <- in_idx
        if (idx != last_element) {
          graph[[idx]]$output_layers <-
            c(graph[[idx]]$output_layers, idx + 1)
        }
        in_idx <- idx
      }
    }

    output_nodes <- unlist(lapply(config$output_layers, function(x) x[[1]]))
    for (node in output_nodes) {
      idx <- rev(which(node == names))[1]
      graph[[idx]]$output_layers <- -1L
    }
  }

  graph
}

has_padding <- function(layer) {
  res <- FALSE
  type <- layer$class_name
  if (type %in% c("Conv1D", "Conv2D", "MaxPooling1D", "MaxPooling2D",
                  "AveragePooling1D", "AveragePooling2D")) {
    if (layer$config$padding != "valid") {
      res <- TRUE
    }
  }

  res
}

has_activation <- function(layer) {
  res <- FALSE
  type <- layer$class_name
  if (type %in% c("Conv1D", "Conv2D", "Dense")) {
    if (layer$config$activation != "linear") {
      res <- TRUE
    }
  }

  res
}


check_consistent_data_format <- function(current_format, given_format) {
  # Everything is fine if the data format is unset
  if (is.null(current_format)) {
    data_format <- given_format
  } else if (current_format == given_format) {
    # or if the data format doesn't change
    data_format <- current_format
  } else {
    # The package can not handle different data formats
    stop(paste0(
      "The package innsight can not handle unconsistent data formats. ",
      "I found the format '", given_format, "', but the data format ",
      "of a previous layer was '", current_format, "'! \n",
      "Choose either only the format 'channels_first' or only ",
      "'channels_last' for all layers."
    ))
  }

  data_format
}

move_channels_first <- function(shape) {
  as.integer(c(rev(shape)[1], shape[-length(shape)]))
}

