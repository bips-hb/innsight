test_that("Test for Dense-Layers",{
  library(torch)

  #
  # Define Dense-Layer
  #
  W <- matrix(rnorm(15), nrow = 3, ncol = 5)
  b <- rnorm(3) * 0
  dense <- dense_layer(weight = W, bias = b, "relu")

  #
  # ------------------------ Test forward method -------------------------------
  #
  input <- torch_randn(20,5, dtype = torch_float())
  y <- dense(input)
  y_true <- dense$activation_f(t(W %*% t(as_array(input)) + b))

  expect_true(as_array(mean(mean(abs(y- y_true), dim=2))) < 1e-5)

  #
  # ------------------------ Test update method --------------------------------
  #
  input_ref <- torch_randn(1,5, dtype = torch_float())
  y <- dense$update_ref(input_ref)
  y_true <- dense$activation_f(t(W %*% t(as_array(input_ref)) + b))

  expect_true(as_array(mean(abs(y- y_true))) < 1e-5)

  #
  # ------------------- Test get_input_relevances method -----------------------
  #
  rel <- torch_randn(4,3,2, dtype = torch_float())
  input <- torch_randn(4,5, dtype = torch_float())
  dense(input)

  x_in <- input$reshape(c(4,1,5))
  z_ij <- torch_mul(x_in, dense$W)
  z_j <- torch_add(torch_sum(z_ij, dim = 3, keepdim = TRUE) ,dense$b$reshape(c(-1,1)))

  # Simple rule
  y <- dense$get_input_relevances(rel)
  y_true <- torch_matmul(torch_transpose(z_ij / z_j, 3,2), rel)

  expect_true("torch_tensor" %in% class(y))
  expect_equal(dim(y), c(4,5,2))
  expect_true(as_array(mean(mean(abs(y - y_true), dim= c(2,3)))) < 1e-5)

  # Epsilon rule (default)
  y <- dense$get_input_relevances(rel, rule_name = "epsilon")
  z <- z_j + torch_sgn(z_j) * 0.001
  y_true <- torch_matmul(torch_transpose(z_ij / z, 3,2), rel)

  expect_true("torch_tensor" %in% class(y))
  expect_equal(dim(y), c(4,5,2))

  # Alpha_beta rule
  y <- dense$get_input_relevances(rel, rule_name = "alpha_beta")
  z <- z_j + torch_sgn(z_j) * 0.001
  y_true <- torch_matmul(torch_transpose(z_ij / z, 3,2), rel)

  expect_true("torch_tensor" %in% class(y))
  expect_equal(dim(y), c(4,5,2))

  #
  # ------------------ Test get_input_multiplier method ------------------------
  #
  mult <- torch_randn(10,3,2, dtype = torch_float())
  input <- torch_randn(10,5, dtype = torch_float())
  input_ref <- torch_randn(1,5, dtype = torch_float())

  dense(input)
  dense$update_ref(input_ref)

  delta_input <- (input - input_ref)$unsqueeze(3)
  delta_output <- (dense$output - dense$output_ref)$unsqueeze(3)

  y_true <- sum(mult * delta_output)

  # Rescale rule
  mult_input <- dense$get_input_multiplier(mult)
  y <- sum(mult_input * delta_input)

  expect_equal(dim(mult_input), c(10,5,2))
  expect_true(as_array(abs(y - y_true)) < 1e-5)

  # Reveal-Cancel rule
  mult_input <- dense$get_input_multiplier(mult, rule_name = "reveal_cancel")
  y <- sum(mult_input * delta_input)

  expect_equal(dim(mult_input), c(10,5,2))
  expect_true(as_array(abs(y - y_true)) < 1e-5)


  #
  # ------------------ Test get_pos_and_neg_outputs method ---------------------
  #

  input <- torch_randn(10,5, dtype = torch_float())
  dense(input)
  y <- dense$preactivation
  out <- dense$get_pos_and_neg_outputs(input, use_bias = TRUE)

  expect_equal(dim(y), dim(out$pos))
  expect_equal(dim(y), dim(out$neg))
  expect_true(as_array(mean(abs(y - out$pos - out$neg))) < 1e-5)

  #
  # --------------------------- Test get_gradient ------------------------------
  #

  output <- torch_randn(10, 3, 2, dtype = torch_float())
  W <- dense$W
  grad <- dense$get_gradient(output, W)
  grad_true <- torch_matmul(W$t(), output)

  expect_equal(dim(grad), c(10, 5, 2))
  expect_true(as_array(mean(abs(grad - grad_true))) < 1e-5)


})
