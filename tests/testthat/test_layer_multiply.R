
test_that("Test permute layer", {
  library(torch)

  batch_size <- 10
  dim_in <- c(5)
  dim_out <- c(5)

  mult <- multiply_layer(dim_in, dim_out)
  input <- list(
    torch_randn(c(batch_size, dim_in)),
    torch_randn(c(batch_size, dim_in))
  )
  input_ref <- list(
    torch_randn(c(1, dim_in)),
    torch_randn(c(1, dim_in))
  )

  # Forward function
  out <- mult(input, save_input = TRUE)
  expect_equal(dim(out), c(batch_size, dim_out))

  # Update reference function
  out_ref <- mult$update_ref(input_ref, save_input = TRUE)
  expect_equal(dim(out_ref), c(1, dim_out))

  # Reshape to input function
  rel_out <- torch_randn(c(batch_size, dim_out, 2))
  rel_in <- mult$reshape_to_input(rel_out)
  expect_equal(dim(rel_in[[1]]), c(batch_size, dim_in, 2))
  expect_equal(dim(rel_in[[2]]), c(batch_size, dim_in, 2))
  expect_lt(as_array(mean((rel_in[[1]] + rel_in[[2]] - rel_out)^2)), 1e-10)

  # Get multiplier function
  mult_out <- torch_randn(c(batch_size, dim_out, 2))
  contrib_true <- (out - out_ref)$unsqueeze(-1) * mult_out
  mult_in <- mult$get_input_multiplier(mult_out)
  contrib_1 <- mult_in[[1]] * (input[[1]] - input_ref[[1]])$unsqueeze(-1)
  contrib_2 <- mult_in[[2]] * (input[[2]] - input_ref[[2]])$unsqueeze(-1)

  expect_lt(as_array(mean((contrib_1 + contrib_2 - contrib_true)^2)), 1e-10)
})
