test_that("Test LRP for Dense-Layers",{
  library(torch)

  rel <- torch_tensor(array(rnorm(12), dim = c(4,3,1)))

  # define dense layer
  W <- matrix(rnorm(15), nrow = 3, ncol = 5)
  b <- rnorm(3)
  dense <- dense_layer(weight = W, bias = b, "linear")

  # define input
  x <- torch_tensor(matrix(rnorm(20), nrow = 4, ncol = 5))
  dense(x)

  # calculate z_ij and z_j
  x <- x$reshape(c(4,1,5))
  z_ij <- torch_mul(x, dense$W)
  z_j <- torch_add(torch_sum(z_ij, dim = 3, keepdim = TRUE) ,dense$b$reshape(c(-1,1)))

  #
  #   --------------------------- Simple Rule------------------------------
  #

  out <- lrp_dense(dense, rel)

  # Test: output is torch tensor
  expect_true("torch_tensor" %in% class(out))

  # Test: output has correct dimension
  expect_equal(dim(out), c(4,5,1))

  # Test: output is correct
  res <- torch_matmul(torch_transpose(z_ij / z_j, 3,2), rel)
  expect_true(as_array(torch_mean(torch_abs(out - res))) < 1e-8)


  #
  #   --------------------------- Epsilon Rule------------------------------
  #

  ## Default: epsilon= 0.001

  out <- lrp_dense(dense, rel, rule_name = "epsilon")

  # Test: output is torch tensor
  expect_true("torch_tensor" %in% class(out))

  # Test: output has correct dimension
  expect_equal(dim(out), c(4,5,1))

  # Test: output is correct
  res <- torch_matmul(torch_transpose(z_ij / (z_j + 0.001* torch_sgn(z_j)), 3,2), rel)
  expect_true(as_array(torch_mean(torch_abs(out - res ))) < 1e-8)


  ## epsilon= 10

  out <- lrp_dense(dense, rel, rule_name = "epsilon", rule_param = 10)

  # Test: output is torch tensor
  expect_true("torch_tensor" %in% class(out))

  # Test: output has correct dimension
  expect_equal(dim(out), c(4,3,5))

  # Test: output is correct
  res <- torch_matmul(torch_transpose(z_ij / (z_j + 10 * torch_sgn(z_j)), 3,2), rel)
  expect_true(as_array(torch_mean(torch_abs(out - res ))) < 1e-8)


  #
  #   --------------------------- Alpha-Beta Rule------------------------------
  #
  z_ij_plus <- z_ij * (z_ij >= 0)
  z_ij_minus <- z_ij * (z_ij < 0)

  b <- dense$b$reshape(c(-1,1))

  b_plus <- b * (b >= 0)
  b_minus <- b * (b < 0)

  z_j_plus <- torch_add(torch_sum(z_ij_plus, dim = 3, keepdim = TRUE) , b_plus) + 1e-16
  z_j_minus <- torch_add(torch_sum(z_ij_minus, dim = 3, keepdim = TRUE) , b_minus) - 1e-16

  ## Default: alpha= 0.5

  out <- lrp_dense(dense, rel, rule_name = "alpha_beta")

  # Test: output is torch tensor
  expect_true("torch_tensor" %in% class(out))

  # Test: output has correct dimension
  expect_equal(dim(out), c(4,3,5))

  # Test: output is correct
  res <- torch_matmul(torch_transpose(0.5 * ( z_ij_plus / z_j_plus ) + 0.5 * ( z_ij_minus / z_j_minus), 3,2), rel)
  expect_true(as_array(torch_mean(torch_abs(out - res)) < 1e-8))


  ##  alpha= 3

  out <- lrp_dense(dense, rel, rule_name = "alpha_beta")

  # Test: output is torch tensor
  expect_true("torch_tensor" %in% class(out))

  # Test: output has correct dimension
  expect_equal(dim(out), c(4,3,5))

  # Test: output is correct
  res <- torch_matmul(torch_transpose(3 * ( z_ij_plus / z_j_plus ) - 2 * ( z_ij_minus / z_j_minus), 3,2), rel)
  expect_true(as_array(torch_mean(torch_abs(out - res))) < 1e-8)

})


test_that("Test LRP for Conv1D-Layers",{
  library(torch)

  ## define conv1d layer

  batch_size <- 10
  in_channels <- 2
  out_channels <- 6
  in_length <- 20
  out_length <- 13
  kernel_size <- 8
  model_out <- 3

  # relevance scores [batch_size, out_channels, out_length, model_out]
  rel <- torch_randn(batch_size, out_channels, out_length, model_out)

  # inputs [batch_size, in_channels, in_length]
  inputs <- torch_randn(batch_size, in_channels, in_length)

  # weight matrix [out_channels, in_channels, kernel_size]
  weights <- torch_randn(out_channels, in_channels, kernel_size)
  bias <- torch_randn(out_channels)

  # define conv1d layer

  conv1d <- conv1d_layer(as_array(weights), as_array(bias), c(in_channels, in_length), c(out_channels, out_length))


  #
  # ---------------------------- Simple Rule -----------------------------------
  #

  out <- lrp_conv1d(dense, rel)

  # Test: output is torch tensor
  expect_true("torch_tensor" %in% class(out))

  # Test: output has correct dimension
  expect_equal(dim(out), c(batch_size,in_channels, in_length, model_out))

  # Test: output is correct
  z_j <- nnf_conv1d(inputs, weights, bias)
  res <- nnf_conv_transpose1d(rel/z_j$unsqueeze(4), weights$unsqueeze(4), stride = 1, padding = 0)

  expect_true(as_array(torch_mean(torch_abs(out - res))) < 1e-8)


  #
  # ------------------------- Epsilon Rule ------------------------------------
  #

  ## Default value: epsilon = 0.001
  out <- lrp_conv1d(dense, rel, rule_name = "epsilon")

  # Test: output is torch tensor
  expect_true("torch_tensor" %in% class(out))

  # Test: output has correct dimension
  expect_equal(dim(out), c(batch_size,in_channels, in_length, model_out))

  # Test: output is correct
  z_j <- nnf_conv1d(inputs, weights, bias)
  z_j <- z_j + 0.001*torch_sgn(z_j)
  res <- nnf_conv_transpose1d(rel/z_j$unsqueeze(4), weights$unsqueeze(4), stride = 1, padding = 0)
  expect_true(as_array(torch_mean(torch_abs(out - res))) < 1e-8)


  ## epsilon = 10
  out <- lrp_conv1d(dense, rel, rule_name = "epsilon")

  # Test: output is torch tensor
  expect_true("torch_tensor" %in% class(out))

  # Test: output has correct dimension
  expect_equal(dim(out), c(batch_size,in_channels, in_length, model_out))

  # Test: output is correct
  z_j <- nnf_conv1d(inputs, weights, bias)
  z_j <- z_j + 10*torch_sgn(z_j)
  res <- nnf_conv_transpose1d(rel/z_j$unsqueeze(4), weights$unsqueeze(4), stride = 1, padding = 0)
  res
  expect_true(as_array(torch_mean(torch_abs(out - res))) < 1e-8)

})
