implemented_layers_keras <- c(
  "Dense", "Dropout", "InputLayer", "Conv1D", "Conv2D", "Flatten",
  "MaxPooling1D", "MaxPooling2D", "AveragePooling1D", "AveragePooling2D",
  "Concatenate", "Add", "Activation", "ZeroPadding1D", "ZeroPadding2D",
  "BatchNormalization", "GlobalAveragePooling1D", "GlobalAveragePooling2D",
  "GlobalMaxPooling1D", "GlobalMaxPooling2D"
)


###############################################################################
#                           Convert Keras Model
###############################################################################

convert_keras_model <- function(model) {

  # Define parameters for the data format and the layer index
  data_format <- NULL
  n <- 1

  # Get layer names and reconstruct graph
  if (inherits(model, "keras.engine.sequential.Sequential")) {
    # If the model is a sequential model, the first layer is the only input
    # layer and the last layer is the only output layer
    graph <- lapply(seq_along(model$layers), function(i) {
      list(input_layers = i - 1, output_layers = i + 1)
    })
    graph[[length(graph)]]$output_layers <- -1
    names <- unlist(lapply(model$layers, FUN = function(x) x$name))
    layers <- model$layers
  } else {
    # Otherwise, we have to reconstruct the computational graph from the
    # model config
    res <- keras_reconstruct_graph(model$layers, model$get_config())
    graph <- res$graph
    layers <- res$layers
    names <- names(layers)
  }

  # Declare list for the list-converted layers
  model_as_list <- vector("list", length = length(names))

  for (layer in layers) {
    # Get the layer type and check whether it is implemented
    type <- layer$`__class__`$`__name__`
    cli_check(checkChoice(type, implemented_layers_keras), "type")

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
        Add = convert_keras_add(layer),
        Activation = convert_keras_activation(layer$get_config()$activation),
        ZeroPadding1D = convert_keras_zeropadding(layer, type),
        ZeroPadding2D = convert_keras_zeropadding(layer, type),
        BatchNormalization = convert_keras_batchnorm(layer),
        GlobalAveragePooling1D = convert_keras_globalpooling(layer, type),
        GlobalAveragePooling2D = convert_keras_globalpooling(layer, type),
        GlobalMaxPooling1D = convert_keras_globalpooling(layer, type),
        GlobalMaxPooling2D = convert_keras_globalpooling(layer, type)
      )

    # Define the incoming and outgoing layers of this layer
    # Thereby means '0' Input-Node and '-1' Output-Node
    layer_list$input_layers <- graph[[n]]$input_layers
    layer_list$output_layers <- graph[[n]]$output_layers

    # Set name of this layer and save it
    model_as_list[[n]] <- layer_list

    n <- n + 1
  }

  # Combine activation functions with convolution or dense layers
  model_as_list <- combine_activations(model_as_list)

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
  if (layer_list$type == "Activation") {
    if ((n - 1) %in% output_nodes) {
      idx <- layer_list$input_layers
      output_nodes[output_nodes == n - 1] <- idx
    }
  }

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
  cli_check(checkChoice(padding, c("valid", "same")), "padding")
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
    stopf("Padding mode '", layer$padding, "' is not implemented yet!")
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

# GlobalPooling Layer --------------------------------------------------

convert_keras_globalpooling <- function(layer, type) {
  if (startsWith(type, "GlobalAverage")) {
    method <- "average"
  } else {
    method <- "max"
  }

  dim_in <- unlist(layer$input_shape)
  dim_out <- unlist(layer$output_shape)
  data_format <- layer$data_format

  # in this package only 'channels_first'
  if (data_format == "channels_last") {
    dim_in <- move_channels_first(dim_in)
    dim_out <- move_channels_first(dim_out)
  }

  list(
    type = "GlobalPooling",
    dim_in = dim_in,
    dim_out = dim_out,
    method = method
  )
}

# ZeroPadding layer -----------------------------------------------------------

convert_keras_zeropadding <- function(layer, type) {
  # padding size: either [left, right] or [top, bottom, left, right]
  padding <- unlist(layer$padding)
  if (length(padding) == 4) {
    padding <- c(padding[3:4], padding[1:2]) # in torch [left,right,top,bottom]
    data_format <- layer$data_format
  } else {
    data_format <- "channels_last"
  }
  dim_in <- unlist(layer$input_shape)
  dim_out <- unlist(layer$output_shape)

  # in this package only 'channels_first'
  if (data_format == "channels_last") {
    dim_in <- move_channels_first(dim_in)
    dim_out <- move_channels_first(dim_out)
  }

  list(
    type = "Padding",
    dim_in = dim_in,
    dim_out = dim_out,
    padding = padding,
    mode = "constant",
    value = 0
  )
}

# BatchNormalization Layer ----------------------------------------------------

convert_keras_batchnorm <- function(layer) {
  input_dim <- unlist(layer$input_shape)
  output_dim <- unlist(layer$output_shape)
  if (is.numeric(layer$axis)) axis <- layer$axis
  else if (is.list(layer$axis)) axis <- as.numeric(layer$axis)
  else axis <- as.numeric(layer$axis[[0]])
  gamma <- as.numeric(layer$gamma$value())
  eps <- as.numeric(layer$epsilon)
  beta <- as.numeric(layer$beta)
  run_mean <- as.numeric(layer$moving_mean)
  run_var <- as.numeric(layer$moving_variance)

  if (axis == length(input_dim)) { # i.e. channels last
    input_dim <- move_channels_first(input_dim)
    output_dim <- move_channels_first(output_dim)
  } else if (axis != 1L) { # i.e. neither first nor last axis
    stopf(
      "Only batchnormalzation on axis '1' or '-1' are accepted! ",
      "Your axis: '", axis, "'")
  }

  list(
    type = "BatchNorm",
    dim_in = input_dim,
    dim_out = output_dim,
    num_features = input_dim[1],
    gamma = gamma,
    eps = eps,
    beta = beta,
    run_mean = run_mean,
    run_var = run_var
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
    warningf(
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
    dim_in = NULL,
    dim_out = NULL
  )
}

# Activation Layer ------------------------------------------------------------

convert_keras_activation <- function(name) {
  list(type = "Activation", act_name = name)
}

# Skipping Layers -------------------------------------------------------------

convert_keras_skipping <- function(type) {
  messagef("Skipping ", type, " ...")

  list(type = "Skipping")
}

###############################################################################
#                 Utility methods: Graph reconstruction
###############################################################################

keras_reconstruct_graph <- function(layers, config) {

  res <- get_layers_graph(layers, config)

  graph <- res$graph
  layer_list <- res$layers
  name_changes <- res$name_changes


  # Rename sequential layers
  for (name in name_changes) {
    for (i in seq_along(graph)) {
      if (!is.null(graph[[i]]$input_layers)) {
        graph[[i]]$input_layers[graph[[i]]$input_layers == name$old] <- name$new
      }
    }
  }

  # Transform input layers to the layer indices and register output nodes
  # for each layer
  names <- names(layer_list)
  for (i in seq_along(graph)) {
    if (!is.null(graph[[i]]$input_layers)) {
      input_layers <- match(graph[[i]]$input_layers, names)
      graph[[i]]$input_layers <- input_layers

      # Register output layers
      for (node_idx in input_layers) {
        graph[[node_idx]]$output_layers <- c(graph[[node_idx]]$output_layers, i)
      }
    }
  }

  # Register model input and output nodes
  input_nodes <- unlist(lapply(config$input_layers, function(x) x[[1]]))
  for (node in input_nodes) {
    idx <- which(node == names)
    graph[[idx]]$input_layers <- 0L
  }
  output_nodes <- unlist(lapply(config$output_layers, function(x) x[[1]]))
  for (node in output_nodes) {
    idx <- which(node == names)
    graph[[idx]]$output_layers <- -1L
  }

  list(graph = graph, layers = layer_list)
}


get_layers_graph <- function(layers, config) {
  config_names <- unlist(lapply(config$layers, function(x) x$name))

  graph <- list()
  layer_list <- list()
  name_changes <- list()

  for (layer in layers) {
    # Get layer index in the config
    config_idx <- which(layer$name == config_names)
    layer_config <- config$layers[[config_idx]]

    # Layers in a 'Sequential' model doesn't contain the config key
    # 'inbound_nodes', hence they need a separat treatment
    if (layer_config$class_name == "Sequential") {
      is_first <- TRUE
      l_names <- unlist(lapply(layer$layers, function(x) x$name))
      for (i in seq_along(layer$layers)) {
        if (is_first) {
          in_names <- unlist(
            lapply(layer_config$inbound_nodes[[1]], function(x) x[[1]])
          )
          is_first <- FALSE
        } else {
          in_names <- l_names[i - 1]
        }

        graph[[l_names[i]]] <-
          list(input_layers = in_names, output_layers = NULL)
        layer_list[[l_names[i]]] <- layer$layers[[i]]
      }

      # A sequential model is saved as a single layer with a default name
      # 'sequential_*'. Therefore, the in- or output layer of the sequential
      # model refers to this name. The list 'name_changes' stores all the
      # relevant name changes.
      name_changes <- append(name_changes,
                             list(list(old = layer$name,
                                       new = l_names[i])))
    }  else if (layer_config$class_name == "Functional") {
      res <- get_layers_graph(layer$layers, layer_config$config)
      # Register input layers
      layer_graph <- res$graph
      layer_graph[[layer_config$config$input_layers[[1]][[1]]]]$input_layers <-
        layer_config$inbound_nodes[[1]][[1]][[1]]

      graph <- append(graph, layer_graph)
      layer_list <- append(layer_list, res$layers)
      name_changes <- append(name_changes, res$name_changes)
      name_changes <- append(
        name_changes,
        list(list(old = layer$name,
                  new = layer_config$config$output_layers[[1]][[1]])))
    } else {
      if (length(layer_config$inbound_nodes) == 1) { # non InputLayer
        in_names <- unlist(
          lapply(layer_config$inbound_nodes[[1]], function(x) x[[1]]))
      } else if (length(layer_config$inbound_nodes) == 0) { # InputLayer
        in_names <- NULL
      } else { # Weight-Sharing is not supported
        stopf("Models that share weights are not supported yet!")
      }

      graph[[layer$name]] <-
        list(input_layers = in_names, output_layers = NULL)
      layer_list[[layer$name]] <- layer
    }
  }

  list(graph = graph, layers = layer_list, name_changes = name_changes)
}

###############################################################################
#                         Other utility methods
###############################################################################

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

check_consistent_data_format <- function(current_format, given_format) {
  # Everything is fine if the data format is unset
  if (is.null(current_format)) {
    data_format <- given_format
  } else if (current_format == given_format) {
    # or if the data format doesn't change
    data_format <- current_format
  } else {
    # The package can not handle different data formats
    stopf(
      "The package {.pkg innsight} can not handle unconsistent data formats. ",
      "I found the format '", given_format, "', but the data format ",
      "of a previous layer was '", current_format, "'! ",
      "Choose either only the format 'channels_first' or only ",
      "'channels_last' for all layers.")
  }

  data_format
}


combine_activations <- function(model_as_list) {
  for (i in seq_along(model_as_list)) {
    if (model_as_list[[i]]$type == "Activation") {
      keep <- FALSE
      for (in_layer in model_as_list[[i]]$input_layers) {
        act_name <- model_as_list[[in_layer]]$activation_name
        if (identical(act_name, "linear")) {
          model_as_list[[in_layer]]$activation_name <-
            model_as_list[[i]]$act_name
        } else if (is.character(act_name)) {
          stopf("It is not allowed to use several activation functions in ",
               "consecutive order! You used a '", model_as_list[[i]]$act_name,
               "' activation directly after a '",
               model_as_list[[in_layer]]$activation_name, "' activation.")
        } else {
          keep <- TRUE
        }
      }

      if (i == length(model_as_list) &&
          all(model_as_list[[i]]$output_layers == -1) &&
          length(model_as_list[[i]]$input_layers == 1)) {
        in_layer <- model_as_list[[i]]$input_layers
        out_layers <- model_as_list[[in_layer]]$output_layers
        model_as_list[[in_layer]]$output_layers <-
            ifelse(out_layers == i, -1, out_layers)
        model_as_list[[i]] <- NULL
      } else if (!keep) {
        # Convert to 'Skipping Layer'
        model_as_list[[i]]$type <- "Skipping"
        model_as_list[[i]]$act_name <- NULL
      }
    }
  }

  model_as_list
}

move_channels_first <- function(shape) {
  as.integer(c(rev(shape)[1], shape[-length(shape)]))
}
