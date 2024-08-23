
test_that("Test permute layer", {
  library(torch)

  batch_size <- 10
  dims <- c(2, 1)
  dim_in <- c(5, 4)
  dim_out <- c(4, 5)

  perm <- permute_layer(dims, dim_in, dim_out)
  input <- torch_randn(c(batch_size, dim_in))
  input_ref <- torch_randn(c(1, dim_in))

  # Forward function
  out <- perm(input)
  expect_equal(dim(out), c(batch_size, dim_out))

  # Update reference function
  out_ref <- perm$update_ref(input_ref)
  expect_equal(dim(out_ref), c(1, dim_out))

  # Reshape to input function
  rel_out <- torch_randn(c(batch_size, dim_out, 2))
  rel_in <- perm$reshape_to_input(rel_out)
  expect_equal(dim(rel_in), c(batch_size, dim_in, 2))
  expect_lt(as_array(mean((rel_in$sum(dims + 1) - rel_out$sum(dims+1))^2)), 1e-10)
})
