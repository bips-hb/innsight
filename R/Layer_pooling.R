###############################################################################
#                       Super-class for pooling layers
###############################################################################

PoolingLayer <- nn_module(
  classname = "PoolingLayer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
  weight = NULL,

  initialize = function(kernel_size, dim_in, dim_out, strides = NULL,
                        dtype = "float") {
    self$kernel_size <- kernel_size
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$dtype <- dtype
    if (is.null(strides)) {
      self$strides <- kernel_size
    } else {
      self$strides <- strides
    }
  },

  forward = function(x, ...) {
    x
  },

  get_gradient = function(x, ...) {
    x
  },

  get_input_relevances = function(rel_output, rule_name = "simple",
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
      z <- self$output$unsqueeze(-1)
      z <- z + torch_eq(z, 0.0) * eps + z$sgn() * eps
      rel_input <-
        self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
    } else if (rule_name == "epsilon") {
      z <- self$output$unsqueeze(-1)
      z <- z + torch_sgn(z) * rule_param + torch_eq(z, 0.0) * eps
      rel_input <-
        self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
    } else if (rule_name == "alpha_beta") {
      out_part <- self$get_pos_and_neg_outputs(self$input)
      input_pos <- torch_clamp(self$input, min = 0)$unsqueeze(-1)
      input_neg <- torch_clamp(self$input, max = 0)$unsqueeze(-1)

      # Apply the simple rule for each part:
      # - positive part
      z <- rel_output /
        (out_part$pos + out_part$pos$eq(0.0) * eps +
           out_part$pos$sgn() * eps)$unsqueeze(-1)

      rel_pos <- self$get_gradient(z) * input_pos

      # - negative part
      z <- rel_output /
        (out_part$neg - out_part$neg$eq(0.0) * eps +
           out_part$neg$sgn() * eps)$unsqueeze(-1)

      rel_neg <- self$get_gradient(z) * input_neg

      # calculate over all relevance for the lower layer
      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
    } else if (rule_name == "pass") {
      stopf("The rule 'pass' is only implemented for layers with the same ",
            "input and output size!")
    }

    rel_input
  },

  get_input_multiplier = function(multiplier, rule_name = NULL, ...) {
    input_multiplier <- self$get_gradient(multiplier)

    input_multiplier
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
  },

  set_dtype = function(dtype) {
    if (!is.null(self$weight) & dtype == "float") {
      self$weight <- self$weight$to(torch_float())
    } else if (!is.null(self$weight) & dtype == "double") {
      self$weight <- self$weight$to(torch_double())
    }
    self$dtype <- dtype
  },

  get_stabilizer = function() {
    # Get stabilizer
    if (self$dtype == "float") {
      eps <- 1e-6 # a bit larger than torch_finfo(torch_float())$eps
    } else {
      eps <- 1e-15 # a bit larger than torch_finfo(torch_double())$eps
    }

    eps
  },

  calc_weight = function(channels) {
    dim <- c(channels, 1, self$kernel_size, 1)
    if (self$dtype == "double") {
      weight <- torch_ones(dim, dtype = torch_double()) / prod(self$kernel_size)
    } else {
      weight <- torch_ones(dim) / prod(self$kernel_size)
    }
    self$weight <- weight

    weight
  }
)

###############################################################################
#                           Average Pooling
###############################################################################

avg_pool1d_layer <- nn_module(
  classname = "AvgPool1D_Layer",
  inherit = PoolingLayer,
  weight = NULL,

  initialize = function(kernel_size, dim_in, dim_out, strides = NULL,
                        dtype = "float") {
    super$initialize(kernel_size, dim_in, dim_out, strides, dtype)
  },

  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE, ...) {
    if (save_input) {
      self$input <- x
    }
    output <- nnf_avg_pool1d(x, self$kernel_size, self$strides)
    if (save_output) {
      self$output <- output
    }

    output
  },

  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE, ...) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    output_ref <- nnf_avg_pool1d(x_ref, self$kernel_size, self$strides)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  get_gradient = function(output, ...) {
    channels <- output$shape[2]

    if (is.null(self$weight)) {
      weight <- self$calc_weight(channels)
    } else {
      weight <- self$weight
    }

    input_grad <- nnf_conv_transpose2d(output, weight,
      stride = c(self$strides, 1),
      groups = channels
    )

    lost_length <- self$input_dim[2] - input_grad$shape[3]
    if (lost_length != 0) {
      input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_length))
    }

    input_grad
  },

  get_pos_and_neg_outputs = function(input) {
    list(
      pos = nnf_avg_pool1d(
        torch_clamp(input, min = 0),
        self$kernel_size, self$strides
      ),
      neg = nnf_avg_pool1d(
        torch_clamp(input, max = 0),
        self$kernel_size, self$strides
      )
    )
  }
)


avg_pool2d_layer <- nn_module(
  classname = "AvgPool2D_Layer",
  inherit = PoolingLayer,
  weight = NULL,

  initialize = function(kernel_size, dim_in, dim_out, strides = NULL,
                        dtype = "float") {
    super$initialize(kernel_size, dim_in, dim_out, strides, dtype)
  },

  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE, ...) {
    if (save_input) {
      self$input <- x
    }
    output <- nnf_avg_pool2d(x, self$kernel_size, self$strides)
    if (save_output) {
      self$output <- output
    }

    output
  },

  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE, ...) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    output_ref <- nnf_avg_pool2d(x_ref, self$kernel_size, self$strides)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  get_gradient = function(output, ...) {
    channels <- output$shape[2]

    if (is.null(self$weight)) {
      weight <- self$calc_weight(channels)
    } else {
      weight <- self$weight
    }

    input_grad <- nnf_conv_transpose3d(output, weight,
      stride = c(self$strides, 1),
      groups = channels
    )

    lost_height <- self$input_dim[2] - input_grad$shape[3]
    lost_width <- self$input_dim[3] - input_grad$shape[4]
    if (lost_height != 0 | lost_width != 0) {
      input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_width, 0, lost_height))
    }

    input_grad
  },

  get_pos_and_neg_outputs = function(input) {
    list(
      pos = nnf_avg_pool2d(
        torch_clamp(input, min = 0),
        self$kernel_size, self$strides
      ),
      neg = nnf_avg_pool2d(
        torch_clamp(input, max = 0),
        self$kernel_size, self$strides
      )
    )
  }
)


###############################################################################
#                           Maximum Pooling
###############################################################################

max_pool1d_layer <- nn_module(
  classname = "MaxPool1D_Layer",
  inherit = PoolingLayer,

  initialize = function(kernel_size, dim_in, dim_out, strides = NULL,
                        dtype = "float") {
    super$initialize(kernel_size, dim_in, dim_out, strides, dtype)
  },

  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE, ...) {
    if (save_input) {
      self$input <- x
    }
    output <- nnf_max_pool1d(x, self$kernel_size, self$strides)
    if (save_output) {
      self$output <- output
    }

    output
  },

  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE, ...) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    output_ref <- nnf_max_pool1d(x_ref, self$kernel_size, self$strides)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  get_input_relevances = function(rel_output, rule_name = "simple",
                                  rule_param = NULL, winner_takes_all = TRUE, ...) {
    if (winner_takes_all) {
      rel_input <- self$get_gradient(rel_output, use_avgpool = FALSE)
    } else {
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

      # Get average pooling output
      out_part <- self$get_pos_and_neg_outputs(self$input)

      # Apply selected LRP-rule
      if (rule_name == "simple") {
        z <- (out_part$pos + out_part$neg)$unsqueeze(-1)
        z <- z + torch_eq(z, 0.0) * eps + z$sgn() * eps
        rel_input <- self$get_gradient(rel_output / z, use_avgpool = TRUE) *
          self$input$unsqueeze(-1)
      } else if (rule_name == "epsilon") {
        z <- (out_part$pos + out_part$neg)$unsqueeze(-1)
        z <- z + torch_sgn(z) * rule_param + torch_eq(z, 0.0) * eps
        rel_input <- self$get_gradient(rel_output / z, use_avgpool = TRUE) *
          self$input$unsqueeze(-1)
      } else if (rule_name == "alpha_beta") {
        input_pos <- torch_clamp(self$input, min = 0)$unsqueeze(-1)
        input_neg <- torch_clamp(self$input, max = 0)$unsqueeze(-1)

        # Apply the simple rule for each part:
        # - positive part
        z <- rel_output /
          (out_part$pos + out_part$pos$eq(0.0) * eps +
             out_part$pos$sgn() * eps)$unsqueeze(-1)

        rel_pos <- self$get_gradient(z, use_avgpool = TRUE) * input_pos

        # - negative part
        z <- rel_output /
          (out_part$neg - out_part$neg$eq(0.0) * eps +
             out_part$neg$sgn() * eps)$unsqueeze(-1)

        rel_neg <- self$get_gradient(z, use_avgpool = TRUE) * input_neg

        # calculate over all relevance for the lower layer
        rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
      }
    }

    rel_input
  },

  get_input_multiplier = function(mult_output, winner_takes_all = TRUE, ...) {
    if (!winner_takes_all) {
      eps <- self$get_stabilizer()
      delta_y <- nnf_avg_pool1d(self$input - self$input_ref, self$kernel_size,
                                self$strides)
      delta_y <- delta_y + delta_y$eq(0.0) * eps
      mult_output <- mult_output *
        ((self$output - self$output_ref) / delta_y)$unsqueeze(-1)
    }

    self$get_gradient(mult_output, use_avgpool = !winner_takes_all)
  },

  get_gradient = function(output, use_avgpool = FALSE, ...) {
    if (use_avgpool) {
      channels <- output$shape[2]

      if (is.null(self$weight)) {
        weight <- self$calc_weight(channels)
      } else {
        weight <- self$weight
      }

      input_grad <- nnf_conv_transpose2d(output, weight,
        stride = c(self$strides, 1),
        groups = channels
      )

      lost_length <- self$input_dim[2] - input_grad$shape[3]
      if (lost_length != 0) {
        input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_length))
      }
    } else {
      num_outputs <- rev(output$shape)[1]
      if (is.null(self$input)) {
        input <- torch_ones(c(1, self$input_dim))
      } else {
        input <- self$input
      }
      input <- input$unsqueeze(-1)$expand(c(input$shape, num_outputs))
      input$requires_grad <- TRUE
      out <- nnf_max_pool2d(input, c(self$kernel_size, 1), c(self$strides, 1))
      input_grad <- autograd_grad(out, input, output)[[1]]
    }

    input_grad
  },

  get_pos_and_neg_outputs = function(input) {
    list(
      pos = nnf_avg_pool1d(
        torch_clamp(input, min = 0),
        self$kernel_size, self$strides
      ),
      neg = nnf_avg_pool1d(
        torch_clamp(input, max = 0),
        self$kernel_size, self$strides
      )
    )
  }
)


max_pool2d_layer <- nn_module(
  classname = "MaxPool2D_Layer",
  inherit = PoolingLayer,

  initialize = function(kernel_size, dim_in, dim_out, strides = NULL,
                        dtype = "float") {
    super$initialize(kernel_size, dim_in, dim_out, strides, dtype)
  },

  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE, ...) {
    if (save_input) {
      self$input <- x
    }
    output <- nnf_max_pool2d(x, self$kernel_size, self$strides)
    if (save_output) {
      self$output <- output
    }

    output
  },

  update_ref = function(x_ref, save_input = TRUE, save_preactivation = TRUE,
                        save_output = TRUE, ...) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    output_ref <- nnf_max_pool2d(x_ref, self$kernel_size, self$strides)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },

  get_input_relevances = function(rel_output, rule_name = "simple",
                                  rule_param = NULL, winner_takes_all = TRUE, ...) {
    if (winner_takes_all) {
      rel_input <- self$get_gradient(rel_output, use_avgpool = FALSE)
    } else {
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

      # Get average pooling output
      out_part <- self$get_pos_and_neg_outputs(self$input)

      # Apply selected LRP-rule
      if (rule_name == "simple") {
        z <- (out_part$pos + out_part$neg)$unsqueeze(-1)
        z <- z + torch_eq(z, 0.0) * eps + z$sgn() * eps
        rel_input <-
          self$get_gradient(rel_output / z, use_avgpool = TRUE) * self$input$unsqueeze(-1)
      } else if (rule_name == "epsilon") {
        z <- (out_part$pos + out_part$neg)$unsqueeze(-1)
        z <- z + torch_sgn(z) * rule_param + torch_eq(z, 0.0) * eps
        rel_input <-
          self$get_gradient(rel_output / z, use_avgpool = TRUE) * self$input$unsqueeze(-1)
      } else if (rule_name == "alpha_beta") {
        input_pos <- torch_clamp(self$input, min = 0)$unsqueeze(-1)
        input_neg <- torch_clamp(self$input, max = 0)$unsqueeze(-1)

        # Apply the simple rule for each part:
        # - positive part
        z <- rel_output /
          (out_part$pos + out_part$pos$eq(0.0) * eps +
             out_part$pos$sgn() * eps)$unsqueeze(-1)

        rel_pos <- self$get_gradient(z, use_avgpool = TRUE) * input_pos

        # - negative part
        z <- rel_output /
          (out_part$neg - out_part$neg$eq(0.0) * eps +
             out_part$neg$sgn() * eps)$unsqueeze(-1)

        rel_neg <- self$get_gradient(z, use_avgpool = TRUE) * input_neg

        # calculate over all relevance for the lower layer
        rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
      }
    }

    rel_input
  },

  get_input_multiplier = function(mult_output, winner_takes_all = TRUE, ...) {
    if (!winner_takes_all) {
      eps <- self$get_stabilizer()
      delta_y <- nnf_avg_pool2d(self$input - self$input_ref, self$kernel_size, self$strides)
      delta_y <- delta_y + delta_y$eq(0.0) * eps
      mult_output <- mult_output * ((self$output - self$output_ref) / delta_y)$unsqueeze(-1)
    }

    self$get_gradient(mult_output, use_avgpool = !winner_takes_all)
  },

  get_gradient = function(output, use_avgpool = FALSE, ...) {
    if (use_avgpool) {
      channels <- output$shape[2]

      if (is.null(self$weight)) {
        weight <- self$calc_weight(channels)
      } else {
        weight <- self$weight
      }

      input_grad <- nnf_conv_transpose3d(output, weight,
        stride = c(self$strides, 1),
        groups = channels
      )

      lost_height <- self$input_dim[2] - input_grad$shape[3]
      lost_width <- self$input_dim[3] - input_grad$shape[4]
      if (lost_height != 0 | lost_width != 0) {
        input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_width, 0, lost_height))
      }
    } else {
      num_outputs <- rev(output$shape)[1]
      if (is.null(self$input)) {
        input <- torch_ones(c(1, self$input_dim))
      } else {
        input <- self$input
      }
      input <- input$unsqueeze(-1)$expand(c(input$shape, num_outputs))
      input$requires_grad <- TRUE
      out <- nnf_max_pool3d(input, c(self$kernel_size, 1), c(self$strides, 1))
      input_grad <- autograd_grad(out, input, output)[[1]]
    }

    input_grad
  },

  get_pos_and_neg_outputs = function(input) {
    list(
      pos = nnf_avg_pool2d(
        torch_clamp(input, min = 0),
        self$kernel_size, self$strides
      ),
      neg = nnf_avg_pool2d(
        torch_clamp(input, max = 0),
        self$kernel_size, self$strides
      )
    )
  }
)
