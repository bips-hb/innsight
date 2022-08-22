
###############################################################################
#                         Convert Torch Sequential
###############################################################################

convert_torch_sequential <- function(model) {
  model_as_list <- list()
  num <- 1

  torch_activations <- c(
    "nn_relu", "nn_leaky_relu", "nn_softplus", "nn_sigmoid", "nn_softmax",
    "nn_tanh"
  )

  for (modul in model$modules[-1]) {
    if (inherits(modul, "nn_flatten")) {
      layer_as_list <- convert_torch_flatten()
    } else if (inherits(modul, "nn_linear")) {
      layer_as_list <- convert_torch_linear(modul)
    } else if (inherits(modul, "nn_conv1d")) {
      layer_as_list <- convert_torch_conv1d(modul)
    } else if (inherits(modul, "nn_conv2d")) {
      layer_as_list <- convert_torch_conv2d(modul)
    } else if (inherits(modul, "nn_avg_pool1d")) {
      layer_as_list <- convert_torch_avg_pool1d(modul)
    } else if (inherits(modul, "nn_avg_pool2d")) {
      layer_as_list <- convert_torch_avg_pool2d(modul)
    } else if (inherits(modul, "nn_max_pool1d")) {
      layer_as_list <- convert_torch_max_pool1d(modul)
    } else if (inherits(modul, "nn_max_pool2d")) {
      layer_as_list <- convert_torch_max_pool2d(modul)
    } else if (inherits(modul, "nn_dropout")) {
      layer_as_list <- convert_torch_skipping("nn_dropout")
    } else if (inherits(modul, torch_activations)) {
      idx <- match(torch_activations, class(modul))
      name <- gsub("nn_", "", torch_activations[!is.na(idx)][[1]])
      layer_as_list <- convert_torch_activation(name)
    } else {
      stop(sprintf(
        "Unknown module of classes: '%s'!",
        paste(class(modul), collapse = "', '")
      ))
    }
    model_as_list$layers <- append(model_as_list$layers, layer_as_list)
  }

  # Register input and output layers
  num_layers <- length(model_as_list$layers)
  for (i in seq_len(num_layers)) {
    model_as_list$layers[[i]]$input_layers <- i - 1
    model_as_list$layers[[i]]$output_layers <-
      if (i == num_layers) -1 else i + 1
  }

  model_as_list$input_nodes <- c(1)
  model_as_list$output_nodes <- c(num_layers)

  model_as_list
}


###############################################################################
#                           Convert Torch Layers
###############################################################################

# Convert nn_linear -----------------------------------------------------------
convert_torch_linear <- function(modul) {
  if (is.null(modul$bias)) {
    bias <- rep(0, times = dim(modul$weight)[1])
  } else {
    bias <- as_array(modul$bias)
  }

  list(list(
    type = "Dense",
    weight = as_array(modul$weight),
    bias = bias,
    dim_in = NULL,
    dim_out = NULL
  ))
}

# Convert nn_conv1d -----------------------------------------------------------
convert_torch_conv1d <- function(modul) {
  if (modul$padding_mode != "zeros") {
    stop(sprintf(
      "Padding mode '%s' is not allowed! Use 'zeros' instead.",
      modul$padding_mode
    ))
  }
  if (length(modul$padding) == 1) {
    padding <- rep(modul$padding, 2)
  } else {
    padding <- modul$padding
  }

  if (is.null(modul$bias)) {
    bias <- rep(0, times = dim(modul$weight)[1])
  } else {
    bias <- as_array(modul$bias)
  }

  layer_as_list <- list(list(
      type = "Conv1D",
      weight = as_array(modul$weight),
      bias = bias,
      dim_in = NULL,
      dim_out = NULL,
      stride = modul$stride,
      padding = padding,
      dilation = modul$dilation
    ))
}

# Convert nn_conv2d -----------------------------------------------------------
convert_torch_conv2d <- function(modul) {
  if (modul$padding_mode != "zeros") {
    stop(sprintf(
      "Padding mode '%s' is not allowed! Use 'zeros' instead.",
      modul$padding_mode
    ))
  }
  if (length(modul$padding) == 1) {
    padding <- rep(modul$padding, 4)
  } else {
    padding <- rep(rev(modul$padding), each = 2)
  }
  if (is.null(modul$bias)) {
    bias <- rep(0, times = dim(modul$weight)[1])
  } else {
    bias <- as_array(modul$bias)
  }

  layer_as_list <- list(list(
      type = "Conv2D",
      weight = as_array(modul$weight),
      bias = bias,
      dim_in = NULL,
      dim_out = NULL,
      padding = padding,
      stride = modul$stride,
      dilation = modul$dilation
    ))
}

# Convert nn_avg_pool1d -------------------------------------------------------
convert_torch_avg_pool1d <- function(modul) {
  if (sum(modul$padding) != 0) {
    stop("Padding for pooling layers is not implemented yet!")
  }

  list(list(
    type = "AveragePooling1D",
    kernel_size = modul$kernel_size,
    dim_in = NULL,
    dim_out = NULL,
    strides = modul$stride
  ))
}

# Convert nn_avg_pool2d -------------------------------------------------------
convert_torch_avg_pool2d <- function(modul) {
  if (sum(modul$padding) != 0) {
    stop("Padding for pooling layers is not implemented yet!")
  }

  list(list(
    type = "AveragePooling2D",
    kernel_size = modul$kernel_size,
    dim_in = NULL,
    dim_out = NULL,
    strides = modul$stride
  ))
}

# Convert nn_max_pool1d -------------------------------------------------------
convert_torch_max_pool1d <- function(modul) {
  if (sum(modul$padding) != 0) {
    stop("Padding for pooling layers is not implemented yet!")
  }

  list(list(
    type = "MaxPooling1D",
    kernel_size = modul$kernel_size,
    dim_in = NULL,
    dim_out = NULL,
    strides = modul$stride
  ))
}

# Convert nn_max_pool2d -------------------------------------------------------
convert_torch_max_pool2d <- function(modul) {
  if (sum(modul$padding) != 0) {
    stop("Padding for pooling layers is not implemented yet!")
  }

  list(list(
    type = "MaxPooling2D",
    kernel_size = modul$kernel_size,
    dim_in = NULL,
    dim_out = NULL,
    strides = modul$stride
  ))
}

# Convert nn_flatten ----------------------------------------------------------
convert_torch_flatten <- function() {
  list(list(
    type = "Flatten",
    dim_in = NULL,
    dim_out = NULL
  ))
}

# Convert activation ----------------------------------------------------------
convert_torch_activation <- function(act_name) {
  list(list(
    type = "Activation",
    act_name = act_name,
    dim_in = NULL,
    dim_out = NULL
  ))
}

# Convert skipping layers -----------------------------------------------------
convert_torch_skipping <- function(type) {
  message(sprintf("Skipping %s ...", type))

  list(list(
    type = "Skipping"
  ))
}
