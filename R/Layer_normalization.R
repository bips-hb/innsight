
###############################################################################
#                           BatchNorm Layer
###############################################################################
batchnorm_layer <- nn_module(
  classname = "BatchNorm_Layer",

  initialize = function(num_features, eps, gamma, beta, run_mean, run_var,
                        dim_in, dim_out, dtype = "float") {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$dtype <- dtype
    self$num_features <- num_features
    self$eps <- eps

    self$beta <- torch_tensor(beta)
    self$gamma <- torch_tensor(gamma)
    self$run_mean <- torch_tensor(run_mean)
    self$run_var <- torch_tensor(run_var)

    self$set_dtype(dtype)
  },

  forward = function(x, save_input = TRUE, save_output = TRUE, ...) {
    if (save_input) {
      self$input <- x
    }

    out <- nnf_batch_norm(x, self$run_mean, self$run_var, self$gamma, self$beta,
                          eps = self$eps)

    if (save_output) {
      self$output <- out
    }

    out
  },

  update_ref = function(x_ref, save_input = TRUE, save_output = TRUE, ...) {
    if (save_input) {
      self$input_ref <- x_ref
    }

    out <- nnf_batch_norm(x_ref, self$run_mean, self$run_var, self$gamma,
                          self$beta, eps = self$eps)

    if (save_output) {
      self$output_ref <- out
    }

    out
  },

  get_input_relevances = function(rel_output,
                                  rule_name = "simple",
                                  rule_param = NULL, ...) {
    if (is.null(rule_param)) {
      if (rule_name == "epsilon") {
        rule_param <- 0.001
      } else if (rule_name == "alpha_beta") {
        rule_param <- 0.5
      }
    }
    new_shape <- c(1, self$num_features, rep(1, length(self$input_dim) - 1))
    dtype <- rel_output$dtype
    eps <- self$get_stabilizer()
    fact <- torch_tensor(self$gamma / (self$run_var + self$eps)^0.5,
                         dtype = dtype)$reshape(new_shape)

    if (rule_name == "simple") {
      z <- self$output
      z <- z + z$eq(0.0) * eps
      rel <- (fact * self$input / z)$unsqueeze(-1) * rel_output
    } else if (rule_name == "epsilon") {
      z <- self$output
      z <- z + rule_param * torch_sgn(z) + eps *  z$eq(0.0)
      rel <- (fact * self$input / z)$unsqueeze(-1) * rel_output
    } else if (rule_name == "alpha_beta") {
      bias <- torch_tensor(- self$gamma * self$run_mean /
                             (self$run_var + self$eps)^0.5 + self$beta)$reshape(new_shape)

      z_plus <- torch_maximum(fact * self$input, 0.0)
      z_plus <- z_plus + torch_eq(z_plus, 0) * eps
      z_minus <- torch_minimum(fact * self$input, 0.0)
      z_minus <- z_minus - torch_eq(z_minus, 0) * eps

      rel <- rule_param * z_plus / (z_plus + torch_maximum(bias, 0.0)) +
        (1 - rule_param) * z_minus / (z_minus + torch_minimum(bias, 0.0))
      rel <- rel$unsqueeze(-1) * rel_output
    } else if (rule_name == "pass") {
      rel <- rel_output
    }

    rel
  },

  get_input_multiplier = function(mult_output, ...) {
    self$get_gradient(mult_output)
  },

  get_gradient = function(grad_out, ...) {
    new_shape <- c(1, self$num_features, rep(1, length(self$input_dim) - 1))
    reshaped_gamma <- torch_tensor(self$gamma)$reshape(new_shape)
    reshaped_run_var <- torch_tensor(self$run_var)$reshape(new_shape)
    fac <- (reshaped_run_var + self$eps)^0.5

    (reshaped_gamma / fac)$unsqueeze(-1) * grad_out
  },

  set_dtype = function(dtype) {
    if (dtype == "float") {
      torch_dtype <- torch_float()
    } else if (dtype == "double") {
      torch_dtype <- torch_double()
    } else {
      stop("Unknown argument for 'dtype' : '", dtype, "'. ",
           "Use 'float' or 'double' instead!")
    }

    self$beta <- self$beta$to(dtype = torch_dtype)
    self$gamma <- self$gamma$to(dtype = torch_dtype)
    self$run_mean <- self$run_mean$to(dtype = torch_dtype)
    self$run_var <- self$run_var$to(dtype = torch_dtype)

    self$dtype <- dtype
  },

  reset = function() {
    self$input <- NULL
    self$input_ref <- NULL
    self$output <- NULL
    self$output_ref <- NULL
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
