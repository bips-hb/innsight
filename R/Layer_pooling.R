

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

  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE) {
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
                        save_output = TRUE) {
    if (save_input) {
      self$input_ref <- x_ref
    }
    output_ref <- nnf_avg_pool1d(x_ref, self$kernel_size, self$strides)
    if (save_output) {
      self$output_ref <- output_ref
    }

    output_ref
  },


  get_input_relevances = function(rel_output, rule_name = "simple", rule_param = NULL) {

    z <- self$output$unsqueeze(-1)
    z <- z + (z == 0) * 1e-12
    rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)

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

    lost_length <- self$input_dim[2] - input_grad$shape[3]
    if (lost_length != 0) {
      input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_length))
    }

    input_grad
  },

  set_dtype = function(dtype) {
    self$dtype <- dtype
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
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

    forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                       save_output = TRUE) {
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
                          save_output = TRUE) {
      if (save_input) {
        self$input_ref <- x_ref
      }
      output_ref <- nnf_avg_pool2d(x_ref, self$kernel_size, self$strides)
      if (save_output) {
        self$output_ref <- output_ref
      }

      output_ref
    },

    get_input_relevances = function(rel_output, rule_name = "simple",
                                    rule_param = NULL) {
      z <- self$output$unsqueeze(-1)
      z <- z + (z == 0) * 1e-12
      rel_input <- self$get_gradient(rel_output / z) * self$input$unsqueeze(-1)

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

      lost_height <- self$input_dim[2] - input_grad$shape[3]
      lost_width <- self$input_dim[3] - input_grad$shape[4]
      if (lost_height != 0 | lost_width != 0) {
        input_grad <- nnf_pad(input_grad, c(0, 0, 0, lost_width, 0, lost_height))
      }

      input_grad
    },

    set_dtype = function(dtype) {
      self$dtype <- dtype
    },

    reset = function() {
      self$input <- NULL
      self$input_ref <- NULL
      self$output <- NULL
      self$output_ref <- NULL
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

  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE) {
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
                        save_output = TRUE) {
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
                                  rule_param = NULL) {
    rel_input <- self$get_gradient(rel_output)

    rel_input
  },

  get_input_multiplier = function(multiplier) {
    input_multiplier <- self$get_gradient(multiplier)

    input_multiplier
  },

  get_gradient = function(output, weight = NULL) {

    output_shape <- output$shape
    num_outputs <- rev(output_shape)[1]

    # Generate inputs for each output
    input <- self$input$torch_repeat_interleave(
      self$input,
      torch_tensor(num_outputs, dtype = torch_long()),
      dim = 1)

    # Track gradients and apply max pooling
    input$requires_grad <- TRUE
    out <- nnf_max_pool1d(input, self$kernel_size, self$strides)

    # Move output dimension after batch dimension and merge both
    output_grad <- torch_movedim(output, -1, 2)$reshape(
      c(-1, output_shape[-c(1, length(output_shape))]))
    input_grad <- autograd_grad(out, input, output_grad)[[1]]

    # Move output dimension again to last position
    input_grad <- torch_movedim(
      torch_reshape(input_grad, c(output_shape[1],-1 , self$input$shape[-1])),
      2, -1)


    input_grad
  },

  set_dtype = function(dtype) {
    self$dtype <- dtype
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
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

  forward = function(x, save_input = TRUE, save_preactivation = TRUE,
                     save_output = TRUE) {
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
                        save_output = TRUE) {
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
                                  rule_param = NULL) {
    rel_input <- self$get_gradient(rel_output)

    rel_input
  },


  get_input_multiplier = function(multiplier, rule_name = NULL) {
    input_multiplier <- self$get_gradient(multiplier)

    input_multiplier
  },

  get_gradient = function(output, weight = NULL, input = NULL) {

    output_shape <- output$shape
    num_outputs <- rev(output_shape)[1]

    # Generate inputs for each output
    if (is.null(input)) {
      input <- self$input
    }
    input <- self$input$torch_repeat_interleave(
      input, torch_tensor(num_outputs, dtype = torch_long()), dim = 1)

    # Track gradients and apply max pooling
    input$requires_grad <- TRUE
    out <- nnf_max_pool2d(input, self$kernel_size, self$strides)

    # Move output dimension after batch dimension and merge both
    output_grad <- torch_movedim(output, -1, 2)$reshape(
      c(-1, output_shape[-c(1, length(output_shape))]))
    input_grad <- autograd_grad(out, input, output_grad)[[1]]

    # Move output dimension again to last position
    input_grad <- torch_movedim(
      torch_reshape(input_grad, c(output_shape[1],-1 , self$input$shape[-1])),
      2, -1)


    input_grad
  },

  set_dtype = function(dtype) {
    self$dtype <- dtype
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
  }
)
