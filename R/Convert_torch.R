
###############################################################################
#                         Convert Torch Sequential
###############################################################################

convert_torch_sequential <- function(model) {
  model_as_list <- list()
  num <- 1
  modules_list <- model$modules[-1]

  # Check for empty model
  if (length(modules_list) == 0) {
    stopf("You passed an empty {.pkg torch} model!")
  }

  activation_possible <- FALSE

  for (modul in modules_list) {
    classes <- class(modul)

    if ("nn_flatten" %in% classes) {
      model_as_list$layers[[num]] <- convert_torch_flatten(num)
      activation_possible <- FALSE
      num <- num + 1
    } else if ("nn_linear" %in% classes) {
      model_as_list$layers[[num]] <- convert_torch_linear(modul, num)
      activation_possible <- TRUE
      num <- num + 1
    } else if ("nn_conv1d" %in% classes) {
      model_as_list$layers[[num]] <- convert_torch_conv1d(modul, num)
      activation_possible <- TRUE
      num <- num + 1
    } else if ("nn_conv2d" %in% classes) {
      model_as_list$layers[[num]] <- convert_torch_conv2d(modul, num)
      activation_possible <- TRUE
      num <- num + 1
    } else if ("nn_avg_pool1d" %in% classes) {
      activation_possible <- TRUE
      model_as_list$layers[[num]] <- convert_torch_avg_pool1d(modul, num)
      num <- num + 1
    } else if ("nn_avg_pool2d" %in% classes) {
      activation_possible <- TRUE
      model_as_list$layers[[num]] <- convert_torch_avg_pool2d(modul, num)
      num <- num + 1
    } else if ("nn_max_pool1d" %in% classes) {
      activation_possible <- TRUE
      model_as_list$layers[[num]] <- convert_torch_max_pool1d(modul, num)
      num <- num + 1
    } else if ("nn_max_pool2d" %in% classes) {
      activation_possible <- TRUE
      model_as_list$layers[[num]] <- convert_torch_max_pool2d(modul, num)
      num <- num + 1
    } else if ("nn_dropout" %in% classes) {
      activation_possible <- FALSE
      model_as_list$layers[[num]] <- convert_torch_skipping("nn_dropout", num)
      num <- num + 1
    } else if ("nn_relu" %in% classes) {
      if (activation_possible) {
        model_as_list$layers[[num - 1]]$activation_name <- "relu"
        activation_possible <- FALSE
      } else {
        activation_error("relu", num, model_as_list$layers)
      }
    } else if ("nn_leaky_relu" %in% classes) {
      if (activation_possible) {
        model_as_list$layers[[num - 1]]$activation_name <- "leaky_relu"
        activation_possible <- FALSE
      } else {
        activation_error("leaky_relu", num, model_as_list$layers)
      }
    } else if ("nn_softplus" %in% classes) {
      if (activation_possible) {
        model_as_list$layers[[num - 1]]$activation_name <- "softplus"
        activation_possible <- FALSE
      } else {
        activation_error("softplus", num, model_as_list$layers)
      }
    } else if ("nn_sigmoid" %in% classes) {
      if (activation_possible) {
        model_as_list$layers[[num - 1]]$activation_name <- "sigmoid"
        activation_possible <- FALSE
      } else {
        activation_error("sigmoid", num, model_as_list$layers)
      }
    } else if ("nn_softmax" %in% classes) {
      if (activation_possible) {
        model_as_list$layers[[num - 1]]$activation_name <- "softmax"
        activation_possible <- FALSE
      } else {
        activation_error("softmax", num, model_as_list$layers)
      }
    } else if ("nn_tanh" %in% classes) {
      if (activation_possible) {
        model_as_list$layers[[num - 1]]$activation_name <- "tanh"
        activation_possible <- FALSE
      } else {
        activation_error("tanh", num, model_as_list$layers)
      }
    } else if (any(c("nn_batch_norm1d", "nn_batch_norm2d") %in% classes)) {
      model_as_list$layers[[num]] <- convert_torch_batchnorm(modul, num)
      activation_possible <- FALSE
      num <- num + 1
    } else {
      stopf("Unknown module of class(es): '",
            paste(classes, collapse = "', '"), "'!")
    }
  }
  model_as_list$layers[[num - 1]]$output_layers <- -1
  model_as_list$input_nodes <- c(1)
  model_as_list$output_nodes <- c(num - 1)

  model_as_list
}


###############################################################################
#                           Convert Torch Layers
###############################################################################

# Convert nn_linear -----------------------------------------------------------
convert_torch_linear <- function(modul, num) {
  if (is.null(modul$bias)) {
    bias <- rep(0, times = dim(modul$weight)[1])
  } else {
    bias <- as_array(modul$bias)
  }

  list(
    type = "Dense",
    weight = as_array(modul$weight),
    bias = bias,
    activation_name = "linear",
    dim_in = NULL,
    dim_out = NULL,
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert nn_conv1d -----------------------------------------------------------
convert_torch_conv1d <- function(modul, num) {
  if (modul$padding_mode != "zeros") {
    stopf(
      "Padding mode '", modul$padding_mode, "' is not allowed! Use 'zeros' ",
      "instead.")
  }
  if (is.null(modul$bias)) {
    bias <- rep(0, times = dim(modul$weight)[1])
  } else {
    bias <- as_array(modul$bias)
  }

  list(
    type = "Conv1D",
    weight = as_array(modul$weight),
    bias = bias,
    activation_name = "linear",
    dim_in = NULL,
    dim_out = NULL,
    stride = modul$stride,
    padding = modul$padding,
    dilation = modul$dilation,
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert nn_conv2d -----------------------------------------------------------
convert_torch_conv2d <- function(modul, num) {
  if (modul$padding_mode != "zeros") {
    stopf(
      "Padding mode '", modul$padding_mode, "' is not allowed! Use 'zeros' ",
      "instead.")
  }
  if (is.null(modul$bias)) {
    bias <- rep(0, times = dim(modul$weight)[1])
  } else {
    bias <- as_array(modul$bias)
  }
  if (length(modul$padding) == 1) {
    padding <- rep(modul$padding, 4)
  } else {
    padding <- rep(rev(modul$padding), each = 2)
  }

  list(
    type = "Conv2D",
    weight = as_array(modul$weight),
    bias = bias,
    activation_name = "linear",
    dim_in = NULL,
    dim_out = NULL,
    stride = modul$stride,
    padding = padding,
    dilation = modul$dilation,
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert nn_avg_pool1d -------------------------------------------------------
convert_torch_avg_pool1d <- function(modul, num) {
  if (sum(modul$padding) != 0) {
    stopf("Padding for pooling layers is not implemented yet!")
  }

  list(
    type = "AveragePooling1D",
    kernel_size = modul$kernel_size,
    dim_in = NULL,
    dim_out = NULL,
    strides = modul$stride,
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert nn_avg_pool2d -------------------------------------------------------
convert_torch_avg_pool2d <- function(modul, num) {
  if (sum(modul$padding) != 0) {
    stopf("Padding for pooling layers is not implemented yet!")
  }

  list(
    type = "AveragePooling2D",
    kernel_size = modul$kernel_size,
    dim_in = NULL,
    dim_out = NULL,
    strides = modul$stride,
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert nn_max_pool1d -------------------------------------------------------
convert_torch_max_pool1d <- function(modul, num) {
  if (sum(modul$padding) != 0) {
    stopf("Padding for pooling layers is not implemented yet!")
  }

  list(
    type = "MaxPooling1D",
    kernel_size = modul$kernel_size,
    dim_in = NULL,
    dim_out = NULL,
    strides = modul$stride,
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert nn_max_pool2d -------------------------------------------------------
convert_torch_max_pool2d <- function(modul, num) {
  if (sum(modul$padding) != 0) {
    stopf("Padding for pooling layers is not implemented yet!")
  }

  list(
    type = "MaxPooling2D",
    kernel_size = modul$kernel_size,
    dim_in = NULL,
    dim_out = NULL,
    strides = modul$stride,
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert nn_flatten ----------------------------------------------------------
convert_torch_flatten <- function(num) {
  list(
    type = "Flatten",
    dim_in = NULL,
    dim_out = NULL,
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert nn_batch_norm* ------------------------------------------------------
convert_torch_batchnorm <- function(modul, num) {
  list(
    type = "BatchNorm",
    dim_in = NULL,
    dim_out = NULL,
    num_features = modul$num_features,
    gamma = as_array(modul$weight),
    eps = modul$eps,
    beta = as_array(modul$bias),
    run_mean = as_array(modul$running_mean),
    run_var = as_array(modul$running_var),
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert skipping layers -----------------------------------------------------
convert_torch_skipping <- function(type, num) {
  messagef("Skipping ", type, " ...")

  list(
    type = "Skipping",
    input_layers = num - 1,
    output_layers = num + 1
  )
}


###############################################################################
#                                 Utils
###############################################################################

activation_error <- function(type, num, layers) {
  if (num == 1) {
    stopf(
      "In this package, it is not allowed to start with an activation",
      " function. Your activation function: '", type, "'")
  } else if (layers[[num - 1]]$type %in% c("Skipping", "Flatten", "BatchNorm")) {
    stopf(
      "In this package, it is not allowed to use an activation function",
      " ('", type, "') after a dropout, flatten or batchnormalization layer.")
  } else {
    stopf(
      "In this package, it is not allowed to apply several activation",
      " functions in a row (..., '", layers[[num - 1]]$activation_name,
      "' ,'", type, "').")
  }
}
