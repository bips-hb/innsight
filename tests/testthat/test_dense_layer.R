library(torch)

test_that("Test initialization and forward of dense_layer", {
  batch_size <- 10
  dim_in <- 20
  dim_out <- 5

  #
  # Test for dtype = "float"
  #

  W <- torch_randn(dim_out,dim_in, dtype = torch_float())
  b <- torch_randn(dim_out, dtype = torch_float())
  dense_float <- dense_layer(weight = W, bias = b, "tanh",dtype = "float")

  input <- torch_randn(batch_size, dim_in, dtype = torch_float())
  input_ref <- torch_randn(1, dim_in, dtype = torch_float())

  # Test forward
  y_float <- dense_float(input)
  expect_true(y_float$dtype == torch_float())
  expect_equal(dim(y_float), c(batch_size, dim_out))

  # Test update_ref
  y_ref_float <- dense_float$update_ref(input_ref)
  expect_equal(dim(y_ref_float), c(1, dim_out))

  #
  # Test for dtype = "double"
  #

  W <- torch_randn(dim_out,dim_in, dtype = torch_double())
  b <- torch_randn(dim_out, dtype = torch_double())
  dense_double <- dense_layer(weight = W, bias = b, "softplus",dtype = "double")

  input <- torch_randn(batch_size, dim_in, dtype = torch_double())
  input_ref <- torch_randn(1, dim_in, dtype = torch_double())

  # Test forward
  y_double <- dense_double(input)
  expect_equal(dim(y_float), c(batch_size, dim_out))

  # Test update_ref
  y_ref_double <- dense_double$update_ref(input_ref)
  expect_equal(dim(y_ref_double), c(1, dim_out))

})


test_that("Test get_pos_and_neg_outputs and get_gradient for dense_layer", {
  batch_size <- 10
  dim_in <- 5
  dim_out <- 20

  W <- torch_randn(dim_out,dim_in)
  b <- torch_randn(dim_out)
  dense <- dense_layer(weight = W, bias = b, "tanh")
  input <- torch_randn(batch_size, dim_in, requires_grad = TRUE)

  # Test get_gradient
  dense(input)
  y <- dense$preactivation
  sum(y)$backward()
  input_grad_true <- input$grad$unsqueeze(3)
  input_grad <- dense$get_gradient(torch_ones(batch_size, dim_out, 1), W)

  expect_equal(dim(input_grad), dim(input_grad_true))
  expect_lt(as_array(mean((input_grad - input_grad_true)^2)), 1e-12)

  # Test get_pos_and_neg_outputs
  out <- dense$get_pos_and_neg_outputs(input, use_bias = TRUE)
  expect_equal(dim(out$pos), c(batch_size, dim_out))
  expect_equal(dim(out$neg), c(batch_size, dim_out))
  expect_lt(as_array(mean((out$pos + out$neg - dense$preactivation)^2)), 1e-12)

  out <- dense$get_pos_and_neg_outputs(input, use_bias = FALSE)
  expect_equal(dim(out$pos), c(batch_size, dim_out))
  expect_equal(dim(out$neg), c(batch_size, dim_out))
  expect_lt(as_array(mean((out$pos + out$neg - dense$preactivation + dense$b)^2)), 1e-12)
})


test_that("Test get_input_relevances for dense_layer", {
  batch_size <- 10
  dim_in <- 50
  dim_out <- 201

  W <- torch_randn(dim_out,dim_in, dtype = torch_double())
  b <- torch_zeros(dim_out, dtype = torch_double())
  dense <- dense_layer(weight = W, bias = b, "softplus", dtype = "double")
  input <- torch_randn(batch_size, dim_in, dtype = torch_double())

  dense(input)

  for (model_out in c(1,3)) {

    rel <- torch_randn(batch_size, dim_out, model_out, dtype = torch_double())

    # Simple rule
    rel_simple <- dense$get_input_relevances(rel)
    expect_equal(dim(rel_simple), c(batch_size, dim_in, model_out))
    rel_simple_in <- sum(rel_simple, dim = 2:3)
    rel_simple_out <- sum(rel, dim = 2:3)
    expect_lt(as_array(mean((rel_simple_in - rel_simple_out)^2)), 1e-12)

    # Epsilon rule
    rel_epsilon <- dense$get_input_relevances(rel, rule_name = "epsilon")
    expect_equal(dim(rel_epsilon), c(batch_size, dim_in, model_out))

    # Alpha_Beta rule
    rel_alpha_beta <- dense$get_input_relevances(rel, rule_name = "alpha_beta")
    expect_equal(dim(rel_alpha_beta), c(batch_size, dim_in, model_out))
  }
})



test_that("Test get_input_relevances for dense_layer", {
  batch_size <- 10
  dim_in <- 201
  dim_out <- 55

  W <- torch_randn(dim_out,dim_in, dtype = torch_double())
  b <- torch_zeros(dim_out, dtype = torch_double())
  dense <- dense_layer(weight = W, bias = b, "softplus", dtype = "double")
  input <- torch_randn(batch_size, dim_in, dtype = torch_double())
  input_ref <- torch_randn(1, dim_in, dtype = torch_double())

  dense(input)
  dense$update_ref(input_ref)

  for (model_out in c(1,3)) {

    mult <- torch_randn(batch_size, dim_out, model_out, dtype = torch_double())
    diff_input <- (input - input_ref)$unsqueeze(3)
    diff_output <- (dense$output - dense$output_ref)$unsqueeze(3)

    contrib_true <- sum(mult * diff_output, dim = 2:3)

    # Rescale rule
    mult_in_rescale <- dense$get_input_multiplier(mult)
    expect_equal(dim(mult_in_rescale), c(batch_size, dim_in, model_out))
    contrib_rescale <- sum(mult_in_rescale * diff_input, dim = 2:3)
    expect_lt(as_array(mean((contrib_rescale - contrib_true)^2)), 1e-12)

    # Reveal Cancel rule
    mult_in_revealcancel <- dense$get_input_multiplier(mult, rule_name = "reveal_cancel")
    expect_equal(dim(mult_in_revealcancel), c(batch_size, dim_in, model_out))
    contrib_revealcancel <- sum(mult_in_revealcancel * diff_input, dim = 2:3)
    expect_lt(as_array(mean((contrib_revealcancel - contrib_true)^2)), 1e-12)
  }
})
