
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

  include_act <- FALSE
  impl_acts <- c("relu", "leaky_relu", "softplus", "sigmoid", "softmax", "tanh")

  for (modul in modules_list) {
    if (inherits(modul, "nn_flatten")) {
      model_as_list$layers[[num]] <- convert_torch_flatten(num)
      include_act <- TRUE
    } else if (inherits(modul, "nn_linear")) {
      model_as_list$layers[[num]] <- convert_torch_linear(modul, num)
      include_act <- TRUE
    } else if (inherits(modul, "nn_conv1d")) {
      model_as_list$layers[[num]] <- convert_torch_conv1d(modul, num)
      include_act <- TRUE
    } else if (inherits(modul, "nn_conv2d")) {
      model_as_list$layers[[num]] <- convert_torch_conv2d(modul, num)
      include_act <- TRUE
    } else if (inherits(modul, "nn_avg_pool1d")) {
      model_as_list$layers[[num]] <- convert_torch_avg_pool1d(modul, num)
      include_act <- TRUE
    } else if (inherits(modul, "nn_avg_pool2d")) {
      model_as_list$layers[[num]] <- convert_torch_avg_pool2d(modul, num)
      include_act <- TRUE
    } else if (inherits(modul, "nn_max_pool1d")) {
      model_as_list$layers[[num]] <- convert_torch_max_pool1d(modul, num)
      include_act <- TRUE
    } else if (inherits(modul, "nn_max_pool2d")) {
      model_as_list$layers[[num]] <- convert_torch_max_pool2d(modul, num)
      include_act <- TRUE
    } else if (inherits(modul, "nn_dropout")) {
      model_as_list$layers[[num]] <- convert_torch_skipping("nn_dropout", num)
      include_act <- FALSE
    } else if (inherits(modul, paste0("nn_", impl_acts))) {
      idx <-
        which(inherits(modul, paste0("nn_", impl_acts),  which = TRUE) == 1)
      act_name <- impl_acts[idx]
      if (include_act) {
        num <- num - 1
        model_as_list$layers[[num]]$activation_name <- act_name
      } else {
        model_as_list$layers[[num]] <- convert_torch_activation(act_name, num)
      }
      include_act <- FALSE
    } else if (inherits(modul, c("nn_batch_norm1d", "nn_batch_norm2d"))) {
      model_as_list$layers[[num]] <- convert_torch_batchnorm(modul, num)
      include_act <- FALSE
    } else {
      stopf("Unknown module of class(es): '",
            paste(class(modul), collapse = "', '"), "'!")
    }

    num <- num + 1
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
  if (modul$affine) {
    gamma <- as_array(modul$weight)
    beta <- as_array(modul$bias)
  } else {
    gamma <- rep(1, modul$num_features)
    beta <- rep(0, modul$num_features)
  }
  list(
    type = "BatchNorm",
    dim_in = NULL,
    dim_out = NULL,
    num_features = modul$num_features,
    gamma = gamma,
    eps = modul$eps,
    beta = beta,
    run_mean = as_array(modul$running_mean),
    run_var = as_array(modul$running_var),
    input_layers = num - 1,
    output_layers = num + 1
  )
}

# Convert activation ----------------------------------------------------------
convert_torch_activation <- function(act_name, num) {
  list(
    type = "Activation",
    dim_in = NULL,
    dim_out = NULL,
    act_name = act_name,
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
