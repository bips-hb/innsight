

#
# ----------------------------- Average Pooling -------------------------------
#
avg_pool1d_layer <- nn_module(
  classname = "AvgPool1D_Layer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
  initialize = function(kernel_size, dim_in, dim_out, strides = NULL, dtype = "float") {
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

  forward = function(x) {
    self$input <- x
    self$output <- nnf_avg_pool1d(x, self$kernel_size, self$strides)

    self$output
  },

  update_ref = function(x_ref) {
    self$input_ref <- x_ref
    self$output_ref <- nnf_avg_pool1d(x_ref, self$kernel_size, self$strides)

    self$output_ref
  },


  get_input_relevances = function(rel_output, rule_name = "simple", rule_param = NULL) {

    # set default parameter
    if (is.null(rule_param)) {
      if (rule_name == "epsilon") {
        rule_param <- 0.001
      } else if (rule_name == "alpha_beta") {
        rule_param <- 0.5
      }
    }

    z <- self$output$unsqueeze(-1)
    # Apply rule
    if (rule_name == "epsilon") {
      z <- z + rule_param * torch_sgn(z)
      rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
    } else if (rule_name == "alpha_beta") {
      pos_input <- self$input * (self$input >= 0)
      neg_input <- self$input * (self$input <= 0)
      pos_out <- nnf_avg_pool1d(pos_input,
                                self$kernel_size, self$strides)$unsqueeze(-1)
      neg_out <- nnf_avg_pool1d(neg_input,
                                self$kernel_size, self$strides)$unsqueeze(-1)

      rel_pos <- self$get_gradient(rel_output / (pos_out + (pos_out == 0) * 1e-12)) *
        pos_input$unsqueeze(-1)
      rel_neg <- self$get_gradient(rel_output / (neg_out + (neg_out == 0) * 1e-12)) *
        neg_input$unsqueeze(-1)

      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)

    } else {
      z <- z + (z == 0) * 1e-12
      rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
    }

    rel_input
  },


  get_input_multiplier = function(multiplier) {
    input_multiplier <- self$get_gradient(multiplier)

    input_multiplier
  },

  get_gradient = function(output, weight = NULL) {
    channels <- output$shape[2]

    if (self$dtype == "double") {
      weight <-
        torch_ones(c(channels, 1, self$kernel_size, 1), dtype = torch_double()) /
        prod(self$kernel_size)
    } else {
      weight <-
        torch_ones(c(channels, 1, self$kernel_size, 1)) /
        prod(self$kernel_size)
    }
    input_grad <- nnf_conv_transpose2d(output, weight,
                                       stride = c(self$strides, 1),
                                       groups = channels)

    lost_length <- self$input$shape[3] - input_grad$shape[3]
    if (lost_length != 0) {
      input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_length))
    }

    input_grad
  },

  set_dtype = function(dtype) {
    self$dtype <- dtype
  }
)


avg_pool2d_layer <- nn_module(
    classname = "AvgPool2D_Layer",
    input_dim = NULL,
    input = NULL,
    input_ref = NULL,
    output_dim = NULL,
    output = NULL,
    output_ref = NULL,
    initialize = function(kernel_size, dim_in, dim_out, strides = NULL, dtype = "float") {
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

    forward = function(x) {
      self$input <- x
      self$output <- nnf_avg_pool2d(x, self$kernel_size, self$strides)

      self$output
    },

    update_ref = function(x_ref) {
      self$input_ref <- x_ref
      self$output_ref <- nnf_avg_pool2d(x_ref, self$kernel_size, self$strides)

      self$output_ref
    },


    get_input_relevances = function(rel_output, rule_name = "simple", rule_param = NULL) {

      # set default parameter
      if (is.null(rule_param)) {
        if (rule_name == "epsilon") {
          rule_param <- 0.001
        } else if (rule_name == "alpha_beta") {
          rule_param <- 0.5
        }
      }

      z <- self$output$unsqueeze(-1)
      # Apply rule
      if (rule_name == "epsilon") {
        z <- z + rule_param * torch_sgn(z)
        rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
      } else if (rule_name == "alpha_beta") {
        pos_input <- self$input * (self$input >= 0)
        neg_input <- self$input * (self$input <= 0)
        pos_out <- nnf_avg_pool2d(pos_input,
                                  self$kernel_size, self$strides)$unsqueeze(-1)
        neg_out <- nnf_avg_pool2d(neg_input,
                                  self$kernel_size, self$strides)$unsqueeze(-1)

        rel_pos <- self$get_gradient(rel_output / (pos_out + (pos_out == 0) * 1e-12)) *
          pos_input$unsqueeze(-1)
        rel_neg <- self$get_gradient(rel_output / (neg_out + (neg_out == 0) * 1e-12)) *
          neg_input$unsqueeze(-1)

        rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)

      } else {
        z <- z + (z == 0) * 1e-12
        rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
      }

      rel_input
    },


    get_input_multiplier = function(multiplier) {
      input_multiplier <- self$get_gradient(multiplier)

      input_multiplier
    },

    get_gradient = function(output, weight = NULL) {
      channels <- output$shape[2]
      if (self$dtype == "double") {
        weight <-
          torch_ones(c(channels, 1, self$kernel_size, 1), dtype = torch_double()) /
          prod(self$kernel_size)
      } else {
        weight <-
          torch_ones(c(channels, 1, self$kernel_size, 1)) /
          prod(self$kernel_size)
      }

      input_grad <- nnf_conv_transpose3d(output, weight,
                                         stride = c(self$strides, 1),
                                         groups = channels)

      lost_height <- self$input$shape[3] - input_grad$shape[3]
      lost_width <- self$input$shape[4] - input_grad$shape[4]
      if (lost_height != 0 | lost_width != 0) {
        input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_width, 0, lost_height))
      }

      input_grad
    },

    set_dtype = function(dtype) {
      self$dtype <- dtype
    }
)


#
# ----------------------------- Maximum Pooling -------------------------------
#
max_pool1d_layer <- nn_module(
  classname = "MaxPool1D_Layer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
  initialize = function(kernel_size, dim_in, dim_out, strides = NULL, dtype = "float") {
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

  forward = function(x) {
    self$input <- x
    self$output <- nnf_max_pool1d(x, self$kernel_size, self$strides)

    self$output
  },

  update_ref = function(x_ref) {
    self$input_ref <- x_ref
    self$output_ref <- nnf_max_pool1d(x_ref, self$kernel_size, self$strides)

    self$output_ref
  },

  get_input_relevances = function(rel_output, rule_name = "simple", rule_param = NULL) {

    # set default parameter
    if (is.null(rule_param)) {
      if (rule_name == "epsilon") {
        rule_param <- 0.001
      } else if (rule_name == "alpha_beta") {
        rule_param <- 0.5
      }
    }

    z <- nnf_avg_pool1d(self$input, self$kernel_size, self$strides)$unsqueeze(-1)

    # Apply rule
    if (rule_name == "epsilon") {
      z <- z + rule_param * torch_sgn(z)
      rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)

    } else if (rule_name == "alpha_beta") {
      pos_input <- self$input * (self$input >= 0)
      neg_input <- self$input * (self$input <= 0)
      pos_out <- nnf_avg_pool1d(pos_input,
                                self$kernel_size, self$strides)$unsqueeze(-1)
      neg_out <- nnf_avg_pool1d(neg_input,
                                self$kernel_size, self$strides)$unsqueeze(-1)

      rel_pos <- self$get_gradient(rel_output / (pos_out + (pos_out == 0) * 1e-12)) *
        pos_input$unsqueeze(-1)
      rel_neg <- self$get_gradient(rel_output / (neg_out + (neg_out == 0) * 1e-12)) *
        neg_input$unsqueeze(-1)

      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)

    } else {
      z <- z + (z == 0) * 1e-12
      rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
    }

   rel_input
  },

  get_input_multiplier = function(multiplier) {
    input_multiplier <- self$get_gradient(multiplier)

    input_multiplier
  },

  get_gradient = function(output, weight = NULL) {
    channels <- output$shape[2]
    if (self$dtype == "double") {
      weight <-
        torch_ones(c(channels, 1, self$kernel_size, 1), dtype = torch_double()) /
        prod(self$kernel_size)
    } else {
      weight <-
        torch_ones(c(channels, 1, self$kernel_size, 1)) /
        prod(self$kernel_size)
    }

    input_grad <- nnf_conv_transpose2d(output, weight,
                                      stride = c(self$strides, 1),
                                      groups = channels)

    lost_length <- self$input$shape[3] - input_grad$shape[3]
    if (lost_length != 0) {
      input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_length))
    }

    input_grad
  },

  set_dtype = function(dtype) {
    self$dtype <- dtype
  }
)


max_pool2d_layer <- nn_module(
  classname = "MaxPool2D_Layer",
  input_dim = NULL,
  input = NULL,
  input_ref = NULL,
  output_dim = NULL,
  output = NULL,
  output_ref = NULL,
  initialize = function(kernel_size, dim_in, dim_out, strides = NULL, dtype = "float") {
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

  forward = function(x) {
    self$input <- x
    self$output <- nnf_max_pool2d(x, self$kernel_size, self$strides)

    self$output
  },

  update_ref = function(x_ref) {
    self$input_ref <- x_ref
    self$output_ref <- nnf_max_pool2d(x_ref, self$kernel_size, self$strides)

    self$output_ref
  },


  get_input_relevances = function(rel_output, rule_name = "simple", rule_param = NULL) {

    # set default parameter
    if (is.null(rule_param)) {
      if (rule_name == "epsilon") {
        rule_param <- 0.001
      } else if (rule_name == "alpha_beta") {
        rule_param <- 0.5
      }
    }

    # Apply rule
    if (rule_name == "epsilon") {
      z <- nnf_avg_pool2d(self$input, self$kernel_size, self$strides)$unsqueeze(-1)
      z <- z + rule_param * torch_sgn(z)
      rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
    } else if (rule_name == "alpha_beta") {
      pos_input <- self$input * (self$input >= 0)
      neg_input <- self$input * (self$input <= 0)
      pos_out <- nnf_avg_pool2d(pos_input,
                                self$kernel_size, self$strides)$unsqueeze(-1)
      neg_out <- nnf_avg_pool2d(neg_input,
                                self$kernel_size, self$strides)$unsqueeze(-1)

      rel_pos <- self$get_gradient(rel_output / (pos_out + (pos_out == 0) * 1e-12)) *
        pos_input$unsqueeze(-1)
      rel_neg <- self$get_gradient(rel_output / (neg_out + (neg_out == 0) * 1e-12)) *
        neg_input$unsqueeze(-1)

      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)

    } else {
      z <- nnf_avg_pool2d(self$input, self$kernel_size, self$strides)$unsqueeze(-1)
      z <- z + (z == 0) * 1e-12
      rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)
    }

    rel_input
  },


  get_input_multiplier = function(multiplier, rule_name = NULL) {
    input_multiplier <- self$get_gradient(multiplier)

    input_multiplier
  },

  get_gradient = function(output, weight = NULL) {
    channels <- output$shape[2]
    if (self$dtype == "double") {
      weight <-
        torch_ones(c(channels, 1, self$kernel_size, 1), dtype = torch_double()) /
        prod(self$kernel_size)
    } else {
      weight <-
        torch_ones(c(channels, 1, self$kernel_size, 1)) /
        prod(self$kernel_size)
    }

    input_grad <- nnf_conv_transpose3d(output, weight,
                                       stride = c(self$strides, 1),
                                       groups = channels)

    lost_height <- self$input$shape[3] - input_grad$shape[3]
    lost_width <- self$input$shape[4] - input_grad$shape[4]
    if (lost_height != 0 | lost_width != 0) {
      input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_width, 0, lost_height))
    }

    input_grad
  },

  set_dtype = function(dtype) {
    self$dtype <- dtype
  }
)
