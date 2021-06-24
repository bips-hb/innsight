test_that("Test initialization, forward and update_ref", {
  library(torch)

  batch_size <- 10
  for (in_channels in c(1,3)) {
    for (in_height in c(50,61)) {
      for (in_width in c(50,53,61,64)) {

        input <- torch_randn(batch_size, in_channels, in_height, in_width, dtype = torch_double())

        for (pad in list(c(0,0,0,0), c(1,0,1,0), c(0,1,1,0), c(2,4,4,2), c(5,10,8,6))) {

          input_pad <- nnf_pad(input, pad = pad)

          for (out_channels in c(1,4)) {

            bias <- torch_randn(out_channels, dtype = torch_double())

            for (k_height in c(2,5,6)) {
              for (k_width in c(2,5,7)) {

                W <- torch_randn(out_channels, in_channels, k_height, k_width, dtype = torch_double())

                for (stride_dilation in c(1,2,4)) {

                  out_stride <- nnf_conv2d(input_pad, bias = bias, W, stride = stride_dilation)
                  out_size_stride <- dim(out_stride)[3:4]

                  out_dilation <- nnf_conv2d(input_pad, W, bias = bias, dilation = stride_dilation)
                  out_size_dilation <- dim(out_dilation)[3:4]

                  #
                  #------------------------- Stride ----------------------------
                  #

                  #
                  # Test initialization
                  #
                  conv2d_stride <- conv2d_layer(weight = W,
                                                bias = bias,
                                                dim_in = c(in_channels, in_height, in_width),
                                                dim_out = c(out_channels, out_size_stride),
                                                stride = stride_dilation,
                                                padding = pad,
                                                dilation = 1,
                                                activation_name = "tanh",
                                                dtype = "double")

                  #
                  # Test forward
                  #

                  y_true <- torch_tanh(out_stride)
                  y <- conv2d_stride(input)
                  expect_equal(dim(y_true), dim(y))
                  expect_lt(as_array((mean(sum((y_true - y)^2, dim = 2:4)))), 1e-12)

                  #
                  # Test update_ref
                  #

                  y_true <- torch_tanh(out_stride)[1:1,,]
                  y <- conv2d_stride$update_ref(input[1:1,,])
                  expect_equal(dim(y_true), dim(y))
                  expect_lt(as_array((sum((y_true - y)^2, dim = 2:4))), 1e-12)


                  #
                  #------------------------- Dilation --------------------------
                  #

                  #
                  # Test initialization
                  #
                  conv2d_dilation <- conv2d_layer(weight = W,
                                                  bias = bias,
                                                  dim_in = c(in_channels, in_height, in_width),
                                                  dim_out = c(out_channels, out_size_dilation),
                                                  stride = 1,
                                                  padding = pad,
                                                  dilation = stride_dilation,
                                                  activation_name = "softplus",
                                                  dtype = "double")

                  #
                  # Test forward
                  #

                  y_true <- nnf_softplus(out_dilation)
                  y <- conv2d_dilation(input)
                  expect_equal(dim(y_true), dim(y))
                  expect_lt(as_array((mean(sum((y_true - y)^2, dim = 2:3)))), 1e-12)

                  #
                  # Test update_ref
                  #

                  y_true <- nnf_softplus(out_dilation)[1:1,,]
                  y <- conv2d_dilation$update_ref(input[1:1,,])
                  expect_equal(dim(y_true), dim(y))
                  expect_lt(as_array((sum((y_true - y)^2, dim = 2:4))), 1e-12)
                }
              }
            }
          }
        }
      }
    }
  }
})


test_that("Test get_pos_and_neg_outputs and get_gradient", {
  library(torch)

  batch_size <- 10
  for (in_channels in c(1,3)) {
    for (in_height in c(50,61)) {
      for (in_width in c(50,53,61,64)) {

        input <- torch_randn(batch_size, in_channels, in_height, in_width, dtype = torch_double(), requires_grad = TRUE)

        for (pad in list(c(0,0,0,0), c(1,0,1,0), c(0,1,1,0), c(2,4,4,2), c(5,10,8,6))) {

          input_pad <- nnf_pad(input, pad = pad)

          for (out_channels in c(1,4)) {

            bias <- torch_randn(out_channels, dtype = torch_double())

            for (k_height in c(2,5,6)) {
              for (k_width in c(2,5,7)) {

                W <- torch_randn(out_channels, in_channels, k_height, k_width, dtype = torch_double())

                for (stride_dilation in c(1,2,4)) {
                  out_stride <- nnf_conv2d(input_pad, bias = bias, W, stride = stride_dilation)
                  out_size_stride <- dim(out_stride)[3:4]

                  out_dilation <- nnf_conv2d(input_pad, W, bias = bias, dilation = stride_dilation)
                  out_size_dilation <- dim(out_dilation)[3:4]

                  #
                  #------------------------- Stride ----------------------------
                  #

                  conv2d_stride <- conv2d_layer(weight = W,
                                                bias = bias,
                                                dim_in = c(in_channels, in_height, in_width),
                                                dim_out = c(out_channels, out_size_stride),
                                                stride = stride_dilation,
                                                padding = pad,
                                                dilation = 1,
                                                activation_name = "linear",
                                                dtype = "double")

                  y <- conv2d_stride(input)

                  #
                  # Test get_pos_and_neg_outputs
                  #
                  output <- conv2d_stride$get_pos_and_neg_outputs(input, use_bias = TRUE)
                  expect_equal(dim(output$pos), dim(y))
                  expect_equal(dim(output$neg), dim(y))
                  expect_lt(as_array(mean((output$pos + output$neg - y)^2)), 1e-12)

                  #
                  # Test get_gradient
                  #
                  if (is_undefined_tensor(input$grad)) {
                    sum(y)$backward()
                  }
                  else {
                    input$grad$zero_()
                    sum(y)$backward()
                  }

                  for (model_out in 1:3) {
                    grad <- conv2d_stride$get_gradient(torch_ones(c(dim(out_stride), model_out), dtype = torch_double()), W)
                    expect_equal(dim(grad), c(dim(input), model_out))
                    expect_lt(as_array(mean((grad -  input$grad$unsqueeze(5))^2)), 1e-12)
                  }

                  #
                  #------------------------- Dilation --------------------------
                  #

                  conv2d_dilation <- conv2d_layer(weight = W,
                                                  bias = bias,
                                                  dim_in = c(in_channels, in_height, in_width),
                                                  dim_out = c(out_channels, out_size_dilation),
                                                  stride = 1,
                                                  padding = pad,
                                                  dilation = stride_dilation,
                                                  activation_name = "linear",
                                                  dtype = "double")

                  y <- conv2d_dilation(input)

                  #
                  # Test get_pos_and_neg_outputs
                  #
                  output <- conv2d_dilation$get_pos_and_neg_outputs(input, use_bias = TRUE)
                  expect_equal(dim(output$pos), dim(y))
                  expect_equal(dim(output$neg), dim(y))
                  expect_lt(as_array(mean((output$pos + output$neg - y)^2)), 1e-12)

                  #
                  # Test get_gradient
                  #
                  input$grad$zero_()
                  sum(y)$backward()

                  for (model_out in 1:3) {
                    grad <- conv2d_dilation$get_gradient(torch_ones(c(dim(out_dilation), model_out), dtype = torch_double()), W)
                    expect_equal(dim(grad), c(dim(input), model_out))
                    expect_lt(as_array(mean((grad -  input$grad$unsqueeze(5))^2)), 1e-12)
                  }
                }
              }
            }
          }
        }
      }
    }
  }
})



test_that("Test get_input_relevances", {
  library(torch)

  batch_size <- 10
  for (in_channels in c(1,3)) {
    for (in_height in c(50,61)) {
      for (in_width in c(50,53,61,64)) {

        input <- torch_randn(batch_size, in_channels, in_height, in_width, dtype = torch_double())

        for (pad in list(c(0,0,0,0), c(1,0,1,0), c(0,1,1,0), c(2,4,4,2))) {

          input_pad <- nnf_pad(input, pad = pad)

          for (out_channels in c(1,4)) {

            bias <- torch_zeros(out_channels, dtype = torch_double())

            for (k_height in c(5,6)) {
              for (k_width in c(5,7)) {

                W <- torch_randn(out_channels, in_channels, k_height, k_width, dtype = torch_double())

                for (stride_dilation in c(1,2,4)) {

                  out_stride <- nnf_conv2d(input_pad, bias = bias, W, stride = stride_dilation)
                  out_size_stride <- dim(out_stride)[3:4]

                  out_dilation <- nnf_conv2d(input_pad, W, bias = bias, dilation = stride_dilation)
                  out_size_dilation <- dim(out_dilation)[3:4]

                  #
                  #------------------------- Stride ----------------------------
                  #

                  conv2d_stride <- conv2d_layer(weight = W,
                                                bias = bias,
                                                dim_in = c(in_channels, in_height, in_width),
                                                dim_out = c(out_channels, out_size_stride),
                                                stride = stride_dilation,
                                                padding = pad,
                                                dilation = 1,
                                                activation_name = "linear",
                                                dtype = "double")
                  # Forward pass
                  out <- conv2d_stride(input)

                  #
                  # Test get_input_relevances
                  #
                  for (model_out in c(1,2,4)) {
                    rel <- torch_randn(batch_size,out_channels, out_size_stride[1],out_size_stride[2] , model_out, dtype = torch_double())

                    # Simple rule
                    rel_lower <- conv2d_stride$get_input_relevances(rel)
                    expect_equal(dim(rel_lower), c(batch_size, in_channels, in_height, in_width, model_out))
                    expect_lt(as_array(mean((sum(rel, dim = 2:4) - sum(rel_lower, dim = 2:4))^2)), 1e-12)

                    # Epsilon rule
                    rel_lower <- conv2d_stride$get_input_relevances(rel, rule_name = "epsilon")
                    expect_equal(dim(rel_lower), c(batch_size, in_channels, in_height, in_width, model_out))

                    # Alpha-Beta rule
                    rel_lower <- conv2d_stride$get_input_relevances(rel, rule_name = "alpha_beta")
                    expect_equal(dim(rel_lower), c(batch_size, in_channels, in_height, in_width, model_out))
                  }


                  #
                  #------------------------- Dilation --------------------------
                  #

                  conv2d_dilation <- conv2d_layer(weight = W,
                                                  bias = bias,
                                                  dim_in = c(in_channels, in_height, in_width),
                                                  dim_out = c(out_channels, out_size_dilation),
                                                  stride = 1,
                                                  padding = pad,
                                                  dilation = stride_dilation,
                                                  activation_name = "linear",
                                                  dtype = "double")

                  # Forward pass
                  out <- conv2d_dilation(input)

                  #
                  # Test get_input_relevances
                  #
                  for (model_out in c(1,2,4)) {
                    rel <- torch_randn(batch_size,out_channels, out_size_dilation[1],out_size_dilation[2] , model_out, dtype = torch_double())

                    # Simple rule
                    rel_lower <- conv2d_dilation$get_input_relevances(rel)
                    expect_equal(dim(rel_lower), c(batch_size, in_channels, in_height, in_width, model_out))
                    expect_lt(as_array(mean((sum(rel, dim = 2:4) - sum(rel_lower, dim = 2:4))^2)), 1e-12)

                    # Epsilon rule
                    rel_lower <- conv2d_dilation$get_input_relevances(rel, rule_name = "epsilon")
                    expect_equal(dim(rel_lower), c(batch_size, in_channels, in_height, in_width, model_out))

                    # Alpha-Beta rule
                    rel_lower <- conv2d_dilation$get_input_relevances(rel, rule_name = "alph_beta")
                    expect_equal(dim(rel_lower), c(batch_size, in_channels, in_height, in_width, model_out))
                  }
                }
              }
            }
          }
        }
      }
    }
  }
})



test_that("Test get_input_multiplier", {
  library(torch)

  batch_size <- 10
  for (in_channels in c(1,3)) {
    for (in_height in c(50,61)) {
      for (in_width in c(50,53,61,64)) {

        input <- torch_randn(batch_size, in_channels, in_height, in_width, dtype = torch_double())
        input_ref <- torch_randn(1, in_channels, in_height, in_width, dtype = torch_double())

        for (pad in list(c(0,0,0,0), c(1,0,1,0), c(0,1,1,0), c(2,4,4,2),c(10,5,8,7))) {

          input_pad <- nnf_pad(input, pad = pad)

          for (out_channels in c(1,4)) {

            bias <- torch_randn(out_channels, dtype = torch_double())

            for (k_height in c(5,6)) {
              for (k_width in c(5,7)) {

                W <- torch_randn(out_channels, in_channels, k_height, k_width, dtype = torch_double())

                for (stride_dilation in c(1,2,4)) {

                  out_stride <- nnf_conv2d(input_pad, bias = bias, W, stride = stride_dilation)
                  out_size_stride <- dim(out_stride)[3:4]

                  out_dilation <- nnf_conv2d(input_pad, W, bias = bias, dilation = stride_dilation)
                  out_size_dilation <- dim(out_dilation)[3:4]

                  #
                  #------------------------- Stride ----------------------------
                  #

                  conv2d_stride <- conv2d_layer(weight = W,
                                                bias = bias,
                                                dim_in = c(in_channels, in_height, in_width),
                                                dim_out = c(out_channels, out_size_stride),
                                                stride = stride_dilation,
                                                padding = pad,
                                                dilation = 1,
                                                activation_name = "linear",
                                                dtype = "double")
                  # Forward pass
                  out <- conv2d_stride(input)
                  out_ref <- conv2d_stride$update_ref(input_ref)

                  #
                  # Test get_input_multiplier
                  #

                  diff_input <- (input - input_ref)$unsqueeze(5)
                  diff_output <- (out - out_ref)$unsqueeze(5)

                  for (model_out in c(1,2,4)) {
                    mult <- torch_randn(batch_size,out_channels, out_size_stride[1], out_size_stride[2], model_out, dtype = torch_double())
                    y_true <- sum(diff_output * mult, dim = 2:4)

                    # Rescale rule
                    input_mult <- conv2d_stride$get_input_multiplier(mult)
                    y <- sum(input_mult * diff_input, dim = 2:4)
                    expect_equal(dim(input_mult), c(batch_size, in_channels, in_height, in_width, model_out))
                    expect_lt(as_array(mean((y - y_true)^2)), 1e-12)

                    # Reveal-Cancel rule
                    input_mult <- conv2d_stride$get_input_multiplier(mult, rule_name = "reveal_cancel")
                    y <- sum(input_mult * diff_input, dim = 2:4)
                    expect_equal(dim(input_mult), c(batch_size, in_channels, in_height, in_width, model_out))
                    expect_lt(as_array(mean((y - y_true)^2)), 1e-12)
                  }

                  #
                  #------------------------- Dilation --------------------------
                  #

                  conv2d_dilation <- conv2d_layer(weight = W,
                                                  bias = bias,
                                                  dim_in = c(in_channels, in_height, in_width),
                                                  dim_out = c(out_channels, out_size_dilation),
                                                  stride = 1,
                                                  padding = pad,
                                                  dilation = stride_dilation,
                                                  activation_name = "linear",
                                                  dtype = "double")

                  # Forward pass
                  out <- conv2d_dilation(input)
                  out_ref <- conv2d_dilation$update_ref(input_ref)

                  #
                  # Test get_input_multiplier
                  #

                  diff_input <- (input - input_ref)$unsqueeze(5)
                  diff_output <- (out - out_ref)$unsqueeze(5)

                  for (model_out in c(1,2,4)) {
                    mult <- torch_randn(batch_size,out_channels, out_size_dilation[1], out_size_dilation[2], model_out, dtype = torch_double())
                    y_true <- sum(diff_output * mult, dim = 2:4)

                    # Rescale rule
                    input_mult <- conv2d_dilation$get_input_multiplier(mult)
                    y <- sum(input_mult * diff_input, dim = 2:4)
                    expect_equal(dim(input_mult), c(batch_size, in_channels, in_height, in_width, model_out))
                    expect_lt(as_array(mean((y - y_true)^2)), 1e-12)

                    # Reveal-Cancel rule
                    input_mult <- conv2d_dilation$get_input_multiplier(mult, rule_name = "reveal_cancel")
                    y <- sum(input_mult * diff_input, dim = 2:4)
                    expect_equal(dim(input_mult), c(batch_size, in_channels, in_height, in_width, model_out))
                    expect_lt(as_array(mean((y - y_true)^2)), 1e-12)
                  }
                }
              }
            }
          }
        }
      }
    }
  }
})
