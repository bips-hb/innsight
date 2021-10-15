
test_that("Test initialization and forward for conv2d_layer", {
  batch_size <- 10
  in_channels <- 3
  out_channels <- 4
  in_height <- 50
  in_width <- 61
  pad <- c(2, 3, 1, 4)
  kernel_height <- 7
  kernel_width <- 6

  input <-
    torch_randn(batch_size, in_channels, in_height, in_width,
      dtype = torch_double()
    )
  input_pad <- nnf_pad(input, pad = pad)

  bias <- torch_randn(out_channels, dtype = torch_double())
  W <- torch_randn(out_channels, in_channels, kernel_height, kernel_width,
    dtype = torch_double()
  )

  out_stride <- nnf_conv2d(input_pad, bias = bias, W, stride = c(1, 3))
  out_length_stride <- dim(out_stride)[3:4]

  out_dilation <- nnf_conv2d(input_pad, W, bias = bias, dilation = c(3, 1))
  out_length_dilation <- dim(out_dilation)[3:4]

  #
  #------------------------- Stride ----------------------------
  #

  # Test initialization
  conv2d_stride <- conv2d_layer(
    weight = W,
    bias = bias,
    dim_in = c(in_channels, in_height, in_width),
    dim_out = c(out_channels, out_length_stride),
    stride = c(1, 3),
    padding = pad,
    dilation = 1,
    activation_name = "relu",
    dtype = "double"
  )


  # Test forward
  y_stride_true <- conv2d_stride$activation_f(out_stride)
  y_stride <- conv2d_stride(input)
  expect_equal(dim(y_stride_true), dim(y_stride))
  expect_lt(as_array((mean((y_stride_true - y_stride)^2))), 1e-12)


  # Test update_ref
  y_ref_stride_true <- conv2d_stride$activation_f(out_stride)[1:1, , ]
  y_ref_stride <- conv2d_stride$update_ref(input[1:1, , ])
  expect_equal(dim(y_ref_stride), dim(y_ref_stride_true))
  expect_lt(as_array(mean((y_ref_stride_true - y_ref_stride)^2)), 1e-12)


  #
  #------------------------- Dilation --------------------------
  #

  # Test initialization
  conv2d_dilation <- conv2d_layer(
    weight = W,
    bias = bias,
    dim_in = c(in_channels, in_height, in_width),
    dim_out = c(out_channels, out_length_dilation),
    stride = 1,
    padding = pad,
    dilation = c(3, 1),
    activation_name = "tanh",
    dtype = "double"
  )

  # Test forward
  y_dilation_true <- conv2d_dilation$activation_f(out_dilation)
  y_dilation <- conv2d_dilation(input)
  expect_equal(dim(y_dilation_true), dim(y_dilation))
  expect_lt(as_array((mean((y_dilation_true - y_dilation)^2))), 1e-12)


  # Test update_ref
  y_ref_dilation_true <- conv2d_dilation$activation_f(out_dilation)[1:1, , ]
  y_ref_dilation <- conv2d_dilation$update_ref(input[1:1, , ])
  expect_equal(dim(y_ref_dilation), dim(y_ref_dilation_true))
  expect_lt(as_array(mean((y_ref_dilation_true - y_ref_dilation)^2)), 1e-12)
})

test_that("Test get_pos_and_neg_outputs and get_gradient", {
  batch_size <- 10
  in_channels <- 3
  out_channels <- 4
  in_height <- 201
  in_width <- 155
  pad <- c(5, 6, 10, 1)
  kernel_height <- 2
  kernel_width <- 6

  input <- torch_randn(batch_size, in_channels, in_height, in_width,
    dtype = torch_double(),
    requires_grad = TRUE
  )
  input_pad <- nnf_pad(input, pad = pad)

  bias <- torch_randn(out_channels, dtype = torch_double())
  W <- torch_randn(out_channels, in_channels, kernel_height, kernel_width,
    dtype = torch_double()
  )

  out_stride <- nnf_conv2d(input_pad, bias = bias, W, stride = c(2, 4))
  out_length_stride <- dim(out_stride)[3:4]

  out_dilation <- nnf_conv2d(input_pad, W, bias = bias, dilation = c(3, 2))
  out_length_dilation <- dim(out_dilation)[3:4]

  #
  #------------------------- Stride ----------------------------
  #

  # Test initialization
  conv2d_stride <- conv2d_layer(
    weight = W,
    bias = bias,
    dim_in = c(in_channels, in_height, in_width),
    dim_out = c(out_channels, out_length_stride),
    stride = c(2, 4),
    padding = pad,
    dilation = 1,
    activation_name = "relu",
    dtype = "double"
  )

  conv2d_stride(input)
  y_true <- conv2d_stride$preactivation

  # Test get_pos_and_neg_outputs
  out_stride <- conv2d_stride$get_pos_and_neg_outputs(input, use_bias = TRUE)
  expect_equal(dim(out_stride$pos), dim(y_true))
  expect_equal(dim(out_stride$neg), dim(y_true))
  expect_lt(
    as_array(mean((out_stride$pos + out_stride$neg - y_true)^2)), 1e-12
  )

  # Test get_gradient
  sum(y_true)$backward()
  for (model_out in 1:3) {
    grad_stride <-
      conv2d_stride$get_gradient(torch_ones(c(dim(out_stride$pos), model_out),
        dtype = torch_double()
      ), W)
    grad_stride_true <- input$grad$unsqueeze(5)
    expect_equal(dim(grad_stride), c(dim(input), model_out))
    expect_lt(as_array(mean((grad_stride - grad_stride_true)^2)), 1e-12)
  }

  #
  #------------------------- Dilation --------------------------
  #

  input <- torch_randn(batch_size, in_channels, in_height, in_width,
    dtype = torch_double(),
    requires_grad = TRUE
  )

  # Test initialization
  conv2d_dilation <- conv2d_layer(
    weight = W,
    bias = bias,
    dim_in = c(in_channels, in_height, in_width),
    dim_out = c(out_channels, out_length_dilation),
    stride = 1,
    padding = pad,
    dilation = 3,
    activation_name = "tanh",
    dtype = "double"
  )

  conv2d_dilation(input)
  y_true <- conv2d_dilation$preactivation

  # Test get_pos_and_neg_outputs
  out_dilation <-
    conv2d_dilation$get_pos_and_neg_outputs(input, use_bias = TRUE)
  expect_equal(dim(out_dilation$pos), dim(y_true))
  expect_equal(dim(out_dilation$neg), dim(y_true))
  expect_lt(
    as_array(mean((out_dilation$pos + out_dilation$neg - y_true)^2)), 1e-12
  )

  # Test get_gradient
  sum(y_true)$backward()
  for (model_out in 1:3) {
    grad_dilation <-
      conv2d_dilation$get_gradient(torch_ones(c(
        dim(out_dilation$pos),
        model_out
      ),
      dtype = torch_double()
      ), W)
    grad_dilation_true <- input$grad$unsqueeze(5)
    expect_equal(dim(grad_dilation), c(dim(input), model_out))
    expect_lt(as_array(mean((grad_dilation - grad_dilation_true)^2)), 1e-12)
  }
})


test_that("Test get_input_relevances", {
  batch_size <- 10
  in_channels <- 1
  out_channels <- 1
  in_height <- 124
  in_width <- 124
  pad <- c(3, 2, 1, 0)
  kernel_height <- 2
  kernel_width <- 6

  input <- torch_randn(batch_size, in_channels, in_height, in_width,
    dtype = torch_double()
  )
  input_pad <- nnf_pad(input, pad = pad)

  bias <- torch_zeros(out_channels, dtype = torch_double())
  W <- torch_randn(out_channels, in_channels, kernel_height, kernel_width,
    dtype = torch_double()
  )

  out_stride <- nnf_conv2d(input_pad, bias = bias, W, stride = c(2, 3))
  out_length_stride <- dim(out_stride)[3:4]

  out_dilation <- nnf_conv2d(input_pad, W, bias = bias, dilation = c(1, 2))
  out_length_dilation <- dim(out_dilation)[3:4]

  #
  #------------------------- Stride ----------------------------
  #

  # Test initialization
  conv2d_stride <- conv2d_layer(
    weight = W,
    bias = bias,
    dim_in = c(in_channels, in_height, in_width),
    dim_out = c(out_channels, out_length_stride),
    stride = c(2, 3),
    padding = pad,
    dilation = 1,
    activation_name = "tanh",
    dtype = "double"
  )

  conv2d_stride(input)

  # Test get_input_relevances
  for (model_out in c(1, 2, 4)) {
    rel <-
      torch_randn(batch_size, out_channels, out_length_stride[1],
        out_length_stride[2], model_out,
        dtype = torch_double()
      )
    rel_true <- sum(rel, dim = 2:4)

    # Simple rule
    rel_simple_stride <- conv2d_stride$get_input_relevances(rel)
    expect_equal(dim(rel_simple_stride), c(
      batch_size, in_channels,
      in_height, in_width, model_out
    ))
    rel_simple_stride_sum <- sum(rel_simple_stride, dim = 2:4)
    expect_lt(as_array(mean((rel_true - rel_simple_stride_sum)^2)), 1e-12)

    # Epsilon rule
    rel_epsilon_stride <-
      conv2d_stride$get_input_relevances(rel, rule_name = "epsilon")
    expect_equal(
      dim(rel_epsilon_stride),
      c(batch_size, in_channels, in_height, in_width, model_out)
    )

    # Alpha-Beta rule
    rel_alpha_beta_stride <-
      conv2d_stride$get_input_relevances(rel, rule_name = "alpha_beta")
    expect_equal(
      dim(rel_alpha_beta_stride),
      c(batch_size, in_channels, in_height, in_width, model_out)
    )
  }

  #
  #------------------------- Dilation --------------------------
  #

  # Test initialization
  conv2d_dilation <- conv2d_layer(
    weight = W,
    bias = bias,
    dim_in = c(in_channels, in_height, in_width),
    dim_out = c(out_channels, out_length_dilation),
    stride = 1,
    padding = pad,
    dilation = c(1, 2),
    activation_name = "relu",
    dtype = "double"
  )

  conv2d_dilation(input)


  # Test get_input_relevances
  for (model_out in c(1, 2, 4)) {
    rel <- torch_randn(batch_size, out_channels, out_length_dilation[1],
      out_length_dilation[2], model_out,
      dtype = torch_double()
    )
    rel_true <- sum(rel, dim = 2:4)

    # Simple rule
    rel_simple_dilation <- conv2d_dilation$get_input_relevances(rel)
    expect_equal(
      dim(rel_simple_dilation),
      c(batch_size, in_channels, in_height, in_width, model_out)
    )
    rel_simple_dilation_sum <- sum(rel_simple_dilation, dim = 2:4)
    expect_lt(as_array(mean((rel_true - rel_simple_dilation_sum)^2)), 1e-12)

    # Epsilon rule
    rel_epsilon_dilation <-
      conv2d_dilation$get_input_relevances(rel, rule_name = "epsilon")
    expect_equal(
      dim(rel_epsilon_dilation),
      c(batch_size, in_channels, in_height, in_width, model_out)
    )

    # Alpha-Beta rule
    rel_alpha_beta_dilation <-
      conv2d_dilation$get_input_relevances(rel, rule_name = "alpha_beta")
    expect_equal(
      dim(rel_alpha_beta_dilation),
      c(batch_size, in_channels, in_height, in_width, model_out)
    )
  }
})


test_that("Test get_input_multiplier", {
  batch_size <- 10
  in_channels <- 3
  out_channels <- 1
  in_height <- 127
  in_width <- 150
  pad <- c(0, 6, 2, 1)
  kernel_height <- 4
  kernel_width <- 7

  input <- torch_randn(batch_size, in_channels, in_height, in_width,
    dtype = torch_double()
  )
  input_ref <- torch_randn(1, in_channels, in_height, in_width,
    dtype = torch_double()
  )
  input_pad <- nnf_pad(input, pad = pad)

  bias <- torch_zeros(out_channels, dtype = torch_double())
  W <- torch_randn(out_channels, in_channels, kernel_height, kernel_width,
    dtype = torch_double()
  )

  out_stride <- nnf_conv2d(input_pad, bias = bias, W, stride = c(5, 3))
  out_length_stride <- dim(out_stride)[3:4]

  out_dilation <- nnf_conv2d(input_pad, W, bias = bias, dilation = c(2, 5))
  out_length_dilation <- dim(out_dilation)[3:4]

  #
  #------------------------- Stride ----------------------------
  #

  # Test initialization
  conv2d_stride <- conv2d_layer(
    weight = W,
    bias = bias,
    dim_in = c(in_channels, in_height, in_width),
    dim_out = c(out_channels, out_length_stride),
    stride = c(5, 3),
    padding = pad,
    dilation = 1,
    activation_name = "tanh",
    dtype = "double"
  )

  # Forward pass
  out <- conv2d_stride(input)
  out_ref <- conv2d_stride$update_ref(input_ref)

  diff_input <- (input - input_ref)$unsqueeze(5)
  diff_output <- (out - out_ref)$unsqueeze(5)

  for (model_out in c(1, 2, 4)) {
    mult <- torch_randn(batch_size, out_channels, out_length_stride[1],
      out_length_stride[2], model_out,
      dtype = torch_double()
    )
    contrib_true <- sum(diff_output * mult, dim = 2:4)

    # Rescale rule
    mult_rescale_stride <- conv2d_stride$get_input_multiplier(mult)
    contrib_rescale_stride <- sum(mult_rescale_stride * diff_input, dim = 2:4)
    expect_equal(
      dim(mult_rescale_stride),
      c(batch_size, in_channels, in_height, in_width, model_out)
    )
    expect_lt(as_array(mean((contrib_rescale_stride - contrib_true)^2)), 1e-12)

    # Reveal-Cancel rule
    mult_revcancel_stride <-
      conv2d_stride$get_input_multiplier(mult, rule_name = "reveal_cancel")
    contrib_revcancel_stride <-
      sum(mult_revcancel_stride * diff_input, dim = 2:4)
    expect_equal(dim(mult_revcancel_stride), c(
      batch_size, in_channels,
      in_height, in_width, model_out
    ))
    expect_lt(
      as_array(mean((contrib_revcancel_stride - contrib_true)^2)), 1e-12
    )
  }

  #
  #------------------------- Dilation --------------------------
  #

  # Test initialization
  conv2d_dilation <- conv2d_layer(
    weight = W,
    bias = bias,
    dim_in = c(in_channels, in_height, in_width),
    dim_out = c(out_channels, out_length_dilation),
    stride = 1,
    padding = pad,
    dilation = c(2, 5),
    activation_name = "tanh",
    dtype = "double"
  )

  # Forward pass
  out <- conv2d_dilation(input)
  out_ref <- conv2d_dilation$update_ref(input_ref)

  diff_input <- (input - input_ref)$unsqueeze(5)
  diff_output <- (out - out_ref)$unsqueeze(5)

  for (model_out in c(1, 2, 4)) {
    mult <- torch_randn(batch_size, out_channels, out_length_dilation[1],
      out_length_dilation[2], model_out,
      dtype = torch_double()
    )
    contrib_true <- sum(diff_output * mult, dim = 2:4)

    # Rescale rule
    mult_rescale_dilation <- conv2d_dilation$get_input_multiplier(mult)
    contrib_rescale_dilation <-
      sum(mult_rescale_dilation * diff_input, dim = 2:4)
    expect_equal(
      dim(mult_rescale_dilation),
      c(batch_size, in_channels, in_height, in_width, model_out)
    )
    expect_lt(
      as_array(mean((contrib_rescale_dilation - contrib_true)^2)), 1e-12
    )

    # Reveal-Cancel rule
    mult_revcancel_dilation <-
      conv2d_dilation$get_input_multiplier(mult, rule_name = "reveal_cancel")
    contrib_revcancel_dilation <-
      sum(mult_revcancel_dilation * diff_input, dim = 2:4)
    expect_equal(
      dim(mult_revcancel_dilation),
      c(batch_size, in_channels, in_height, in_width, model_out)
    )
    expect_lt(
      as_array(mean((contrib_revcancel_dilation - contrib_true)^2)), 1e-12
    )
  }
})
