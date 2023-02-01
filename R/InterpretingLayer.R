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
                                  rule_param = NULL, ...) {

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
      z <- z + torch_eq(z, 0.0) * eps + z$sgn() * eps
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
        (out_part$pos + out_part$pos$eq(0.0) * eps +
           out_part$pos$sgn() * eps)$unsqueeze(-1)

      rel_pos <-
        self$get_gradient(z, W_pos) * input_pos +
        self$get_gradient(z, W_neg) * input_neg

      # - negative part
      z <- rel_output /
        (out_part$neg - out_part$neg$eq(0.0) * eps +
           out_part$neg$sgn() * eps)$unsqueeze(-1)

      rel_neg <-
        self$get_gradient(z, W_pos) * input_neg +
        self$get_gradient(z, W_neg) * input_pos

      # calculate over all relevance for the lower layer
      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
    } else if (rule_name == "pass") {
      stopf("The rule 'pass' is only implemented for layers with the same ",
            "input and output size!")
    }

    rel_input
  },

  get_input_multiplier = function(mult_output, rule_name = "rescale", ...) {

    # --------------------- Non-linear part---------------------------
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      # Get stabilizer
      eps <- self$get_stabilizer()

      if (rule_name == "rescale") {
        delta_output <- (self$output - self$output_ref)$unsqueeze(-1)
        delta_preact <-
          (self$preactivation - self$preactivation_ref)$unsqueeze(-1)

        # Near zero needs special treatment
        mask <- torch_le(abs(delta_preact), eps) * 1.0
        x <-  mask *
          (self$preactivation + self$preactivation_ref)$unsqueeze(-1) / 2
        x$requires_grad <- TRUE

        y <- sum(self$activation_f(x))
        grad <- autograd_grad(y, x)[[1]]

        nonlin_mult <-
          (1 - mask) * (delta_output / (delta_preact + delta_preact$eq(0.0) * eps)) +
          mask * grad

        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult
      } else if (rule_name == "reveal_cancel") {
        output <- self$get_pos_and_neg_outputs(self$input - self$input_ref)

        delta_x_pos <- output$pos
        delta_x_neg <- output$neg

        act <- self$activation_f
        x <- self$preactivation
        x_ref <- self$preactivation_ref

        delta_output_pos <-
          ( 0.5 * (act(x_ref + delta_x_pos) - act(x_ref)) +
              0.5 * (act(x) - act(x_ref + delta_x_neg)))

        delta_output_neg <-
          ( 0.5 * (act(x_ref + delta_x_neg) - act(x_ref)) +
              0.5 * (act(x) - act(x_ref + delta_x_pos)))

        mult_pos <-
          mult_output * (delta_output_pos / (delta_x_pos + delta_x_pos$eq(0.0) * eps))$unsqueeze(-1)
        mult_neg <-
          mult_output * (delta_output_neg / (delta_x_neg - delta_x_neg$eq(0.0) * eps))$unsqueeze(-1)
      }
    }

    # -------------- Linear part -----------------------

    delta_input <- (self$input - self$input_ref)$unsqueeze(-1)
    weight_pos <- torch_clamp(self$W, min = 0)
    weight_neg <- torch_clamp(self$W, max = 0)

    mult_input <-
      self$get_gradient(mult_pos, weight_pos) * torch_greater(delta_input, 0.0) +
      self$get_gradient(mult_pos, weight_neg) * torch_less(delta_input, 0.0) +
      self$get_gradient(mult_neg, weight_pos) * torch_less(delta_input, 0.0) +
      self$get_gradient(mult_neg, weight_neg) * torch_greater(delta_input, 0.0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), self$W) * delta_input$eq(0.0)

    mult_input
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
      stopf("Unknown argument for {.arg dtype} : '", dtype, "'. ",
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
    stopf(sprintf(
      "Activation function '%s' is not implementet yet!",
      act_name
    ))
  }

  result$act_func <- act
  result$act_name <- act_name

  result
}
