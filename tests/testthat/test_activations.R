
test_that("Test acivation functions for 'Converter' with torch models", {
  library(torch)

  # Relu activation
  model <- nn_sequential(nn_linear(10, 10), nn_relu(), nn_linear(10, 1))
  expect_no_error(conv <- convert(model, input_dim = 10))
  grad <- run_grad(conv, torch_randn(2, 10))

  # Tanh activation
  model <- nn_sequential(nn_linear(10, 10), nn_tanh(), nn_linear(10, 1))
  expect_no_error(conv <- convert(model, input_dim = 10))
  grad <- run_grad(conv, torch_randn(2, 10))

  # Sigmoid activation
  model <- nn_sequential(nn_linear(10, 10), nn_sigmoid(), nn_linear(10, 1))
  expect_no_error(conv <- convert(model, input_dim = 10))
  grad <- run_grad(conv, torch_randn(2, 10))

  # Leaky relu activation
  model <- nn_sequential(nn_linear(10, 10), nn_leaky_relu(), nn_linear(10, 1))
  expect_no_error(conv <- convert(model, input_dim = 10))
  grad <- run_grad(conv, torch_randn(2, 10))

  # Softplus activation
  model <- nn_sequential(nn_linear(10, 10), nn_softplus(), nn_linear(10, 1))
  expect_no_error(conv <- convert(model, input_dim = 10))
  grad <- run_grad(conv, torch_randn(2, 10))

  # Elu activation
  model <- nn_sequential(nn_linear(10, 10), nn_elu(), nn_linear(10, 1))
  expect_no_error(conv <- convert(model, input_dim = 10))
  grad <- run_grad(conv, torch_randn(2, 10))

  # Softmax activation
  model <- nn_sequential(nn_linear(10, 10), nn_softmax(dim = 2), nn_linear(10, 1))
  expect_no_error(conv <- convert(model, input_dim = 10))
  grad <- run_grad(conv, torch_randn(2, 10))


})
