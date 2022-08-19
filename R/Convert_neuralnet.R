

convert_neuralnet_model <- function(model) {

  # Test whether the model has been fitted yet
  if (!("result.matrix" %in% names(model))) {
    stop("The model hasn't been fitted yet!")
  }

  # Get number of best repetition
  if (ncol(model$result.matrix) == 1) {
    best_rep <- 1
  } else {
    best_rep <- which.min(model$result.matrix["error", ])
  }

  # Get the weight matrices and the activation name
  weights <- model$weights[[best_rep]]
  act_name <- attributes(model$act.fct)$type
  if (act_name == "function") {
    stop("You can't use custom activation functions for this package.")
  }

  model_as_list <- list()
  n <- 1
  for (i in seq_along(weights)) {
    # the first row is the bias vector and the rest the weight matrix
    b <- as.vector(weights[[1]][1, ])
    w <- t(matrix(weights[[1]][-1, ], ncol = length(b)))

    # consider the activation of the last layer
    if (i == length(weights) && model$linear.output == TRUE) {
      act_name <- "linear"
    }
    dim_out <- dim(w)[1]

    # convert the layer as a list with all important parameters
    # The argument 'input_layers' saves all the previous layers of the current
    # layer. '0' means that the current layer is an input layer. Similarly,
    # the argument 'output_layers' saves all indices of the succeeding layers,
    # whereby '-1' means that the current layer is an output layer
    model_as_list$layers[[n]] <-
      list(
        type = "Dense",
        weight = w,
        bias = b,
        dim_in = dim(w)[2],
        dim_out = dim_out,
        input_layers = n - 1,
        output_layers =
          ifelse(i == length(weights) && act_name == "linear", -1, n + 1)
      )
    n <- n + 1

    if (act_name != "linear") {
      model_as_list$layers[[n]] <-
        list(
          type = "Activation",
          act_name = act_name,
          dim_in = dim_out,
          dim_out = dim_out,
          input_layers = n - 1,
          output_layers = ifelse(i == length(weights), -1, n + 1))
      n <- n + 1
    }
  }

  # Save other attributes of the model
  model_as_list$input_dim <- list(ncol(model$covariate))
  model_as_list$output_dim <- list(ncol(model$response))
  model_as_list$input_names <- list(list(model$model.list$variables))
  model_as_list$output_names <- list(list(model$model.list$response))
  # Neuralnet implements only sequential models, hence the input node is the
  # first and the output node the last layer
  model_as_list$input_nodes <- c(1)
  model_as_list$output_nodes <- c(n - 1)

  model_as_list
}
