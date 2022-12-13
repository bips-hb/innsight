###############################################################################
#                   Super-class for Layers with weights
###############################################################################

InterpretingLayer <- nn_module(
  classname = "InterpretingLayer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  preactivation = NULL,
  preactivation_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
  activation_f = NULL,
  activation_name = NULL,
  W = NULL,
  b = NULL,
  dtype = NULL,

  initialize = function() {
  },

  forward = function() {
  },

  get_input_relevances = function(rel_output,
                                  rule_name = "simple",
                                  rule_param = NULL) {

    # Set default rule parameter
    if (is.null(rule_param)) {
      if (rule_name == "epsilon") {
        rule_param <- 0.001
      } else if (rule_name == "alpha_beta") {
        rule_param <- 0.5
      }
    }

    # Get stabilizer
    eps <- self$get_stabilizer()

    # Apply selected LRP-rule
    if (rule_name == "simple") {
      z <- self$preactivation$unsqueeze(-1)
      z <- z + (torch_sgn(z) + torch_eq(z, 0.0)) * eps
      rel_input <-
        self$get_gradient(rel_output / z, self$W) * self$input$unsqueeze(-1)
    } else if (rule_name == "epsilon") {
      z <- self$preactivation$unsqueeze(-1)
      z <- z + torch_sgn(z) * rule_param + torch_eq(z, 0.0) * eps
      rel_input <-
        self$get_gradient(rel_output / z, self$W) * self$input$unsqueeze(-1)
    } else if (rule_name == "alpha_beta") {
      out_part <- self$get_pos_and_neg_outputs(self$input, use_bias = TRUE)
      input_pos <- torch_clamp(self$input, min = 0)$unsqueeze(-1)
      input_neg <- torch_clamp(self$input, max = 0)$unsqueeze(-1)
      W_pos <- torch_clamp(self$W, min = 0)
      W_neg <- torch_clamp(self$W, max = 0)

      # Apply the simple rule for each part:
      # - positive part
      z <- rel_output /
        (out_part$pos +
           out_part$pos$eq(0.0) * eps +
           out_part$pos$sgn() * eps)$unsqueeze(-1)

      rel_pos <-
        self$get_gradient(z, W_pos) * input_pos +
        self$get_gradient(z, W_neg) * input_neg

      # - negative part
      z <- rel_output /
        (out_part$neg +
           out_part$neg$sgn() * eps -
           out_part$neg$eq(0.0) * eps)$unsqueeze(-1)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <-
        self$get_gradient(z, W_pos) * input_neg +
        self$get_gradient(z, W_neg) * input_pos

      # calculate over all relevance for the lower layer
      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
    }

    rel_input
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$preactivation <- NULL
    self$preactivation_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
  },

  set_dtype = function(dtype) {
    if (dtype == "float") {
      self$W <- self$W$to(torch_float())
      self$b <- self$b$to(torch_float())
    } else if (dtype == "double") {
      self$W <- self$W$to(torch_double())
      self$b <- self$b$to(torch_double())
    } else {
      stop("Unknown argument for 'dtype' : '", dtype, "'. ",
           "Use 'float' or 'double' instead!")
    }
    self$dtype <- dtype
  },

  get_activation = function(act_name) {
    activation <- get_activation(act_name)

    self$activation_f <- activation$act_func
    self$activation_name <- activation$act_name
  },

  get_stabilizer = function() {
    # Get stabilizer
    if (self$dtype == "float") {
      eps <- 1e-6 # a bit larger than torch_finfo(torch_float())$eps
    } else {
      eps <- 1e-15 # a bit larger than torch_finfo(torch_double())$eps
    }

    eps
  }

)

###############################################################################
#                                Utils
###############################################################################

get_activation <- function(act_name) {
  result <- NULL

  if (act_name == "relu") {
    act <- nn_relu()
  } else if (act_name == "leaky_relu") {
    act <- nn_leaky_relu()
  } else if (act_name == "softplus") {
    act <- nn_softplus()
  } else if (act_name %in% c("sigmoid", "logistic")) {
    act <- nn_sigmoid()
  } else if (act_name == "softmax") {
    act <- nn_softmax(dim = -1)
  } else if (act_name == "tanh") {
    act <- nn_tanh()
  } else if (act_name == "linear") {
    act <- function(x) x
  } else {
    stop(sprintf(
      "Activation function '%s' is not implementet yet!",
      act_name
    ))
  }

  result$act_func <- act
  result$act_name <- act_name

  result
}
