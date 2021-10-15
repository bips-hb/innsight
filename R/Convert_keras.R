implemented_layers <- c(
  "Dense", "Dropout", "InputLayer", "Conv1D", "Conv2D", "Flatten"
)

convert_keras_model <- function(model) {
  if (!requireNamespace("keras")) {
    stop("Please install the 'keras' package.")
  }
  model_dict <- list()
  data_format <- NULL
  num <- 1
  for (layer in model$layers) {
    type <- layer$`__class__`$`__name__`
    name <- paste(type, num, sep = "_")

    assertChoice(type, implemented_layers)

    if (type == "Dropout" || type == "InputLayer") {
      message(sprintf("Skipping %s-Layer...", type))
    } else if (type == "Dense") {
      model_dict$layers[[name]] <- convert_keras_dense(layer)
      num <- num + 1
    } else if (type == "Conv1D" || type == "Conv2D") {
      # set the data_format
      if (is.null(data_format)) {
        data_format <- layer$data_format
      }
      result <- convert_keras_convolution(layer)
      result$type <- type
      model_dict$layers[[name]] <- result
      num <- num + 1
    } else if (type == "Flatten") {
      input_dim <- unlist(layer$input_shape)
      output_dim <- unlist(layer$output_shape)

      # in this package only 'channels_first'
      if (layer$data_format == "channels_last") {
        input_dim <- c(rev(input_dim)[1], input_dim[-length(input_dim)])
        output_dim <- c(rev(output_dim)[1], output_dim[-length(output_dim)])
      }

      model_dict$layers[[name]] <-
        list(type = type, dim_in = input_dim, dim_out = output_dim)
      num <- num + 1
    }
  }

  input_dim <- unlist(model$input_shape)
  output_dim <- unlist(model$output_shape)
  # in this package only 'channels_first'
  if (is.character(data_format) && data_format == "channels_last") {
    input_dim <- c(rev(input_dim)[1], input_dim[-length(input_dim)])
    output_dim <- c(rev(output_dim)[1], output_dim[-length(output_dim)])
  }

  model_dict$input_dim <- input_dim
  model_dict$output_dim <- output_dim

  model_dict
}



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

convert_keras_convolution <- function(layer) {
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
    input_dim <-
      as.integer(c(rev(input_dim)[1], input_dim[-length(input_dim)]))
    output_dim <-
      as.integer(c(rev(output_dim)[1], output_dim[-length(output_dim)]))
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

##### utils


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
