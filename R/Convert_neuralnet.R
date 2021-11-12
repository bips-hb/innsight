

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

  model_dict <- list()
  for (i in seq_along(weights)) {
    name <- sprintf("Dense_%s", i)

    # the first row is the bias vector and the rest the weight matrix
    b <- as.vector(weights[[i]][1, ])
    w <- t(matrix(weights[[i]][-1, ], ncol = length(b)))

    if (i == length(weights) && model$linear.output == TRUE) {
      act_name <- "linear"
    }
    model_dict$layers[[name]] <-
      list(
        type = "Dense",
        weight = w,
        bias = b,
        activation_name = act_name,
        dim_in = dim(w)[2],
        dim_out = dim(w)[1]
      )
  }

  model_dict$input_dim <- ncol(model$covariate)
  model_dict$output_dim <- ncol(model$response)
  model_dict$input_names <- list(model$model.list$variables)
  model_dict$output_names <- list(model$model.list$response)

  model_dict
}
