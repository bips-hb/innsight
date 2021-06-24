

Analyzer <- R6::R6Class("Analyzer",
  public = list(

      model = NULL,

      input_last = NULL,
      input_last_ref = NULL,
      input_dim = NULL,
      input_names = NULL,

      output_dim = NULL,
      output_names = NULL,

      initialize = function(model, feature_names = NULL, response_names = NULL) {
          checkmate::assertArray(feature_names, null.ok = TRUE)
          checkmate::assertArray(response_names, null.ok = TRUE)

          # Analyze the passed model and store its internal structure in a list of
          # layers

          if (inherits(model, "nn")) {
             result <- analyze_neuralnet_model(model)
          }
          else if (inherits(model, c("keras.engine.sequential.Sequential", "keras.engine.functional.Functional"))) {
              result <- analyze_keras_model(model)
          }
          else {
            stop(sprintf("Unknown model of class \"%s\".", paste0(class(model), collapse = "\", \"")))
          }

          self$model <- result$model
          self$input_dim <- result$input_dim
          self$output_dim <- result$output_dim
          self$input_names <-result$input_names
          self$output_names <- result$output_names
      },

      forward = function(x, channels_first = TRUE) {
          x <- torch::torch_tensor(as.array(x), dtype = torch::torch_float())
          if (channels_first == FALSE) {
              x <- torch::torch_movedim(x, -1,2)
          }

          out <- self$model(x, channels_first)
          self$input_last <- x

          torch::as_array(out)
      },

      update_ref = function(x_ref, channels_first = TRUE) {
          x_ref <- torch::torch_tensor(as.array(x_ref), dtype = torch::torch_float())
          if (channels_first == FALSE) {
            x_ref <- torch::torch_movedim(x_ref, -1,2)
          }
          out_ref <- self$model$update_ref(x_ref, channels_first)
          self$input_last_ref <- x_ref

          torch::as_array(out_ref)
      },

      LRP = function(rule_name = "simple", rule_param = NULL) {

      }
    )
)


analyzed_model <- torch::nn_module(
    classname = "Analyzed_Model",
    modules_list = NULL,

    initialize = function(modules_list) {
      self$modules_list <- modules_list
    },

    forward = function(x, channels_first = TRUE) {
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

    update_ref = function(x_ref, channels_first = TRUE) {
        for (module in self$modules_list) {
          if ("Flatten_Layer" %in% module$.classes) {
            x_ref <- module(x_ref, channels_first)
          }
          else {
            x_ref <- module$update_ref(x_ref)
          }
        }
      x_ref
    }

)

analyze_neuralnet_model <- function(model) {
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
                                           activation_name = "linear")
    }
    else {
      modules_list[[name]] <- dense_layer(weight = w,
                                           bias = b,
                                           activation_name = act_name)
    }
  }

  result <- NULL

  result$model <- analyzed_model(modules_list)
  result$input_dim <- ncol(model$covariate)
  result$output_dim <- ncol(model$response)
  result$input_names <- model$model.list$variables
  result$output_names <- model$model.list$response

  result

}

implemented_layers <- c("Dense", "Dropout", "InputLayer", "Conv1D", "Conv2D", "Flatten")

analyze_keras_model <- function(model) {
  if (!requireNamespace("keras")) {
    stop("Please install the 'keras' package.")
  }
  modules_list = list()
  data_format = NULL
  num = 1
  for (layer in model$layers) {
    type <- layer$`__class__`$`__name__`

    if (type %in% implemented_layers) {
      if (type == "Dropout" || type == "InputLayer") {
        message(sprintf("Skipping %s-Layer...", type))
      }
      else if (type == "Dense") {
        act_name <- layer$activation$`__name__`
        weights <- as.array(t(layer$get_weights()[[1]]))
        bias <- as.vector(layer$get_weights()[[2]])
        name <- paste(type, num, sep = "_")
        num <- num + 1
        modules_list[[name]] <- dense_layer(weight = weights,
                                            bias = bias,
                                            activation_name = act_name)
      }
      else if (type %in% c("Conv1D", "Conv2D") ) {
        # set the data_format
        if (is.null(data_format)) {
          data_format <- layer$data_format
        }
        layer_config <- layer$get_config()

        act_name <- layer_config$activation
        filters <- as.numeric(layer_config$filters)
        kernel_size <- as.numeric(unlist(layer_config$kernel_size))
        stride <- as.numeric(unlist(layer_config$strides))
        padding <- layer_config$padding
        dilation <- unlist(layer_config$dilation_rate)

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
        if (padding == "valid") {
          if (type == "Conv1D") {
            padding <- c(0,0)
          }
          else {
            padding = c(0,0,0,0)
          }
        }
        else if (padding == "same") {
          if (type == "Conv1D") {
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
          else if (type == "Conv2D") {
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
        }
        else {
          stop(sprintf("The padding format \"%s\" is not supported. Use \"same\"", padding))
        }
        name <- paste(type, num, sep = "_")
        num <- num + 1

        weight <-  layer$get_weights()[[1]]
        bias <- as.vector(layer$get_weights()[[2]])

        if (type == "Conv1D") {
          # keras weight format: [kernel_length, in_channels, out_channels]
          # torch weight format: [out_channels, in_channels, kernel_length]
          weight <- aperm(weight, c(3,2,1))

          modules_list[[name]] <- conv1d_layer(weight = weight,
                                               bias = bias,
                                               dim_in = input_dim,
                                               dim_out = output_dim,
                                               stride = stride,
                                               padding = padding,
                                               dilation = dilation,
                                               activation_name = act_name)
        }
        else {
          # Conv2D
          # keras weight format: [kernel_height, kernel_width, in_channels, out_channels]
          # torch weight format: [out_channels, in_channels, kernel_height, kernel_width]
          weight <- aperm(weight, perm = c(4,3,1,2))

          modules_list[[name]] <- conv2d_layer(weight = weight,
                                               bias = bias,
                                               dim_in = input_dim,
                                               dim_out = output_dim,
                                               stride = stride,
                                               padding = padding,
                                               dilation = dilation,
                                               activation_name = act_name)
        }
      }
      else if (type == "Flatten") {
        input_dim <- unlist(layer$input_shape)
        output_dim <- unlist(layer$output_shape)

        # in this package only 'channels_first'
        if (layer$data_format == "channels_last") {
          input_dim <- c(rev(input_dim)[1], input_dim[-length(input_dim)])
          output_dim <- c(rev(output_dim)[1], output_dim[-length(output_dim)])
        }

        name <- paste(type, num, sep = "_")
        num <- num + 1

        modules_list[[name]] <- flatten_layer(input_dim, output_dim)
      }
    }
    else {
      stop(sprintf("Layer of type \"%s\" is not implemented yet. Supported layers are: \"%s\"", type,
                   paste0(implemented_layers, collapse = "\", \"")))
    }
  }
  result <- NULL

  result$model <- analyzed_model(modules_list)
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
  result$input_names <- lapply(result$input_dim, function(x) paste0(rep("X", times = x), 1:x))
  result$output_names <- lapply(result$output_dim, function(x) paste0(rep("Y", times = x), 1:x))

  result
}


#library(neuralnet)
#data(iris)
#nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#                iris, linear.output = TRUE,
#                hidden = c(5,3), act.fct = "logistic", rep = 1)
#
#predict(nn, matrix(1, ncol = 2, nrow = 5))
#an <- Analyzer$new(nn)
#x <- torch::torch_ones(c(5,2), requires_grad = TRUE)
#an$model(x)
#an$model$modules_list$Dense_Layer_2$input_ref
#an$model$update_ref(x)
#an$model$modules_list$Dense_Layer_2$input_ref
