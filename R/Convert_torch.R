# model has to be a nn_sequential model

convert_torch_sequential <- function(model) {
  model_dict <- list()
  num <- 1
  modules_list <- model$modules[-1]

  for (modul in modules_list) {
    classes <- class(modul)

    if ("nn_flatten" %in% classes) {
      model_dict$layers[[num]] <- list(
        type = "Flatten",
        dim_in = NULL,
        dim_out = NULL
      )
      num <- num + 1
    }
    else if ("nn_linear" %in% classes) {
      if (is.null(modul$bias)) {
        bias <- rep(0, times = dim(modul$weight)[1])
      } else {
        bias <- as_array(modul$bias)
      }
      model_dict$layers[[num]] <- list(
        type = "Dense",
        weight = as_array(modul$weight),
        bias = bias,
        activation_name = "linear",
        dim_in = NULL,
        dim_out = NULL
      )
      num <- num + 1
    } else if ("nn_conv1d" %in% classes) {
      if (modul$padding_mode != "zeros") {
        stop(sprintf("Padding mode '%s' is not allowed! Use 'zeros' instead.",
                     modul$padding_mode))
      }
      if (is.null(modul$bias)) {
        bias <- rep(0, times = dim(modul$weight)[1])
      } else {
        bias <- as_array(modul$bias)
      }
      model_dict$layers[[num]] <- list(
        type = "Conv1D",
        weight = as_array(modul$weight),
        bias = bias,
        activation_name = "linear",
        dim_in = NULL,
        dim_out = NULL,
        stride = modul$stride,
        padding = modul$padding,
        dilation = modul$dilation
      )
      num <- num + 1
    } else if ("nn_conv2d" %in% classes) {
      if (modul$padding_mode != "zeros") {
        stop(sprintf("Padding mode '%s' is not allowed! Use 'zeros' instead.",
                     modul$padding_mode))
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
      model_dict$layers[[num]] <- list(
        type = "Conv2D",
        weight = as_array(modul$weight),
        bias = bias,
        activation_name = "linear",
        dim_in = NULL,
        dim_out = NULL,
        stride = modul$stride,
        padding = padding,
        dilation = modul$dilation
      )
      num <- num + 1
    } else if ("nn_dropout" %in% classes) {
      message("Skipping Dropout-Layer...")
    } else if ("nn_relu" %in% classes) {
      model_dict$layers[[num - 1]]$activation_name <- "relu"
    } else if ("nn_leaky_relu" %in% classes) {
      model_dict$layers[[num - 1]]$activation_name <- "leaky_relu"
    } else if ("nn_softplus" %in% classes) {
      model_dict$layers[[num - 1]]$activation_name <- "softplus"
    } else if ("nn_sigmoid" %in% classes) {
      model_dict$layers[[num - 1]]$activation_name <- "sigmoid"
    } else if ("nn_softmax" %in% classes) {
      model_dict$layers[[num - 1]]$activation_name <- "softmax"
    } else if ("nn_tanh" %in% classes) {
      model_dict$layers[[num - 1]]$activation_name <- "tanh"
    } else {
      stop(sprintf("Unknown module of class '%s'!", classes[1]))
    }
  }

  model_dict

}
