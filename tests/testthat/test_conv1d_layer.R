test_that("Test initialization, forward and update_ref", {
  library(torch)

  for (batch_size in c(10)) { #, 50, 100
    for (in_channels in c(1,3,5)) {
      for (in_length in c(50,61,77)) {

        input <- torch_randn(batch_size, in_channels, in_length, dtype = torch_double())

        for (pad in list(c(0,0), c(1,0), c(2,2),c(2,4), c(5,10))) {

          input_pad <- nnf_pad(input, pad = pad)

          for (out_channels in c(1,3,4)) {

            bias <- torch_randn(out_channels, dtype = torch_double())

            for (kernel_length in c(2,5,6)) {

              W <- torch_randn(out_channels, in_channels, kernel_length, dtype = torch_double())

              for (stride_dilation in 1:5) {

                out_stride <- nnf_conv1d(input_pad, bias = bias, W, stride = stride_dilation)
                out_length_stride <- dim(out_stride)[3]

                out_dilation <- nnf_conv1d(input_pad, W, bias = bias, dilation = stride_dilation)
                out_length_dilation <- dim(out_dilation)[3]

                for (act in c("linear", "relu", "tanh", "softplus")) {

                  #
                  #------------------------- Stride ----------------------------
                  #

                  #
                  # Test initialization
                  #
                  conv1d_stride <- conv1d_layer(weight = W,
                                                bias = bias,
                                                dim_in = c(in_channels, in_length),
                                                dim_out = c(out_channels, out_length_stride),
                                                stride = stride_dilation,
                                                padding = pad,
                                                dilation = 1,
                                                activation_name = act,
                                                dtype = "double")

                  #
                  # Test forward
                  #

                  y_true <- conv1d_stride$activation_f(out_stride)
                  y <- conv1d_stride(input)
                  expect_equal(dim(y_true), dim(y))
                  expect_lt(as_array((mean(sum((y_true - y)^2, dim = 2:3)))), 1e-12)

                  #
                  # Test update_ref
                  #

                  y_true <- conv1d_stride$activation_f(out_stride)[1:1,,]
                  y <- conv1d_stride$update_ref(input[1:1,,])
                  expect_equal(dim(y_true), dim(y))
                  expect_lt(as_array((sum((y_true - y)^2, dim = 2:3))), 1e-12)


                  #
                  #------------------------- Dilation --------------------------
                  #

                  #
                  # Test initialization
                  #
                  conv1d_dilation <- conv1d_layer(weight = W,
                                                  bias = bias,
                                                  dim_in = c(in_channels, in_length),
                                                  dim_out = c(out_channels, out_length_dilation),
                                                  stride = 1,
                                                  padding = pad,
                                                  dilation = stride_dilation,
                                                  activation_name = act,
                                                  dtype = "double")

                  #
                  # Test forward
                  #

                  y_true <- conv1d_dilation$activation_f(out_dilation)
                  y <- conv1d_dilation(input)
                  expect_equal(dim(y_true), dim(y))
                  expect_lt(as_array((mean(sum((y_true - y)^2, dim = 2:3)))), 1e-12)

                  #
                  # Test update_ref
                  #

                  y_true <- conv1d_dilation$activation_f(out_dilation)[1:1,,]
                  y <- conv1d_dilation$update_ref(input[1:1,,])
                  expect_equal(dim(y_true), dim(y))
                  expect_lt(as_array((sum((y_true - y)^2, dim = 2:3))), 1e-12)
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

  for (in_channels in c(1,3,5)) {
    for (in_length in c(50,61,77)) {

      input <- torch_randn(batch_size, in_channels, in_length, dtype = torch_double(), requires_grad = TRUE)

      for (pad in list(c(0,0), c(1,0), c(2,2),c(2,4), c(5,10))) {

        input_pad <- nnf_pad(input, pad = pad)

        for (out_channels in c(1,3,4)) {

          bias <- torch_randn(out_channels, dtype = torch_double())

          for (kernel_length in c(2,5,6)) {

            W <- torch_randn(out_channels, in_channels, kernel_length, dtype = torch_double())

            for (stride_dilation in 1:5) {

              out_stride <- nnf_conv1d(input_pad, bias = bias, W, stride = stride_dilation)
              out_length_stride <- dim(out_stride)[3]

              out_dilation <- nnf_conv1d(input_pad, W, bias = bias, dilation = stride_dilation)
              out_length_dilation <- dim(out_dilation)[3]

              #
              #------------------------- Stride ----------------------------
              #

              #
              # Test initialization
              #
              conv1d_stride <- conv1d_layer(weight = W,
                                            bias = bias,
                                            dim_in = c(in_channels, in_length),
                                            dim_out = c(out_channels, out_length_stride),
                                            stride = stride_dilation,
                                            padding = pad,
                                            dilation = 1,
                                            activation_name = "linear",
                                            dtype = "double")

              y <- conv1d_stride(input)

              #
              # Test get_pos_and_neg_outputs
              #
              output <- conv1d_stride$get_pos_and_neg_outputs(input, use_bias = TRUE)
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
                grad <- conv1d_stride$get_gradient(torch_ones(c(dim(out_stride), model_out), dtype = torch_double()), W)
                expect_equal(dim(grad), c(dim(input), model_out))
                expect_lt(as_array(mean((grad -  input$grad$unsqueeze(4))^2)), 1e-12)
              }


              #
              #------------------------- Dilation --------------------------
              #

              #
              # Test initialization
              #
              conv1d_dilation <- conv1d_layer(weight = W,
                                              bias = bias,
                                              dim_in = c(in_channels, in_length),
                                              dim_out = c(out_channels, out_length_dilation),
                                              stride = 1,
                                              padding = pad,
                                              dilation = stride_dilation,
                                              activation_name = "linear",
                                              dtype = "double")

              y <- conv1d_dilation(input)

              #
              # Test get_pos_and_neg_outputs
              #
              output <- conv1d_dilation$get_pos_and_neg_outputs(input, use_bias = TRUE)
              expect_equal(dim(output$pos), dim(y))
              expect_equal(dim(output$neg), dim(y))
              expect_lt(as_array(mean((output$pos + output$neg - y)^2)), 1e-12)

              #
              # Test get_gradient
              #
              input$grad$zero_()
              sum(y)$backward()
              for (model_out in 1:3) {
                grad <- conv1d_dilation$get_gradient(torch_ones(c(dim(out_dilation), model_out), dtype = torch_double()), W)
                expect_equal(dim(grad), c(dim(input), model_out))
                expect_lt(as_array(mean((grad -  input$grad$unsqueeze(4))^2)), 1e-12)
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
  for (in_channels in c(1,3,5)) {
    for (in_length in c(50,61,77)) {

      input <- torch_randn(batch_size, in_channels, in_length, dtype = torch_double())

      for (pad in list(c(0,0), c(1,0), c(2,2), c(3,1), c(2,4))) {

        input_pad <- nnf_pad(input, pad = pad)

        for (out_channels in c(1,3,4)) {

          bias <- torch_zeros(out_channels, dtype = torch_double())

          for (kernel_length in c(5,6,10)) {

            W <- torch_randn(out_channels, in_channels, kernel_length, dtype = torch_double())

            for (stride_dilation in 1:5) {

              out_stride <- nnf_conv1d(input_pad, bias = bias, W, stride = stride_dilation)
              out_length_stride <- dim(out_stride)[3]

              out_dilation <- nnf_conv1d(input_pad, W, bias = bias, dilation = stride_dilation)
              out_length_dilation <- dim(out_dilation)[3]

              #
              #------------------------- Stride ----------------------------
              #

              #
              # Test initialization
              #
              conv1d_stride <- conv1d_layer(weight = W,
                                            bias = bias,
                                            dim_in = c(in_channels, in_length),
                                            dim_out = c(out_channels, out_length_stride),
                                            stride = stride_dilation,
                                            padding = pad,
                                            dilation = 1,
                                            activation_name = "linear",
                                            dtype = "double")
              # Forward pass
              out <- conv1d_stride(input)

              #
              # Test get_input_relevances
              #
              for (model_out in c(1,2,4)) {
                rel <- torch_randn(batch_size,out_channels, out_length_stride, model_out, dtype = torch_double())

                # Simple rule
                rel_lower <- conv1d_stride$get_input_relevances(rel)
                expect_equal(dim(rel_lower), c(batch_size, in_channels, in_length, model_out))
                expect_lt(as_array(mean((sum(rel, dim = 2:3) - sum(rel_lower, dim = 2:3))^2)), 1e-12)

                # Epsilon rule
                rel_lower <- conv1d_stride$get_input_relevances(rel, rule_name = "epsilon")
                expect_equal(dim(rel_lower), c(batch_size, in_channels, in_length, model_out))

                # Alpha-Beta rule
                rel_lower <- conv1d_stride$get_input_relevances(rel, rule_name = "alph_beta")
                expect_equal(dim(rel_lower), c(batch_size, in_channels, in_length, model_out))
              }

              #
              #-------------------------- Dilation -----------------------------
              #

              #
              # Test initialization
              #
              conv1d_dilation <- conv1d_layer(weight = W,
                                            bias = bias,
                                            dim_in = c(in_channels, in_length),
                                            dim_out = c(out_channels, out_length_dilation),
                                            stride = 1,
                                            padding = pad,
                                            dilation = stride_dilation,
                                            activation_name = "linear",
                                            dtype = "double")
              # Forward pass
              out <- conv1d_dilation(input)

              #
              # Test get_input_relevances
              #
              for (model_out in c(1,2,4)) {
                rel <- torch_randn(batch_size,out_channels, out_length_dilation, model_out, dtype = torch_double())

                # Simple rule
                rel_lower <- conv1d_dilation$get_input_relevances(rel)
                expect_equal(dim(rel_lower), c(batch_size, in_channels, in_length, model_out))
                expect_lt(as_array(mean((sum(rel, dim = 2:3) - sum(rel_lower, dim = 2:3))^2)), 1e-12)

                # Epsilon rule
                rel_lower <- conv1d_dilation$get_input_relevances(rel, rule_name = "epsilon")
                expect_equal(dim(rel_lower), c(batch_size, in_channels, in_length, model_out))

                # Alpha-Beta rule
                rel_lower <- conv1d_dilation$get_input_relevances(rel, rule_name = "alph_beta")
                expect_equal(dim(rel_lower), c(batch_size, in_channels, in_length, model_out))
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
  for (in_channels in c(1,3,5)) {
    for (in_length in c(50,61,77)) {

      input <- torch_randn(batch_size, in_channels, in_length, dtype = torch_double())
      input_ref <- torch_randn(1, in_channels, in_length, dtype = torch_double())

      for (pad in list(c(0,0), c(1,0), c(2,2), c(3,1), c(2,4))) {

        input_pad <- nnf_pad(input, pad = pad)

        for (out_channels in c(1,3,4)) {

          bias <- torch_randn(out_channels, dtype = torch_double())

          for (kernel_length in c(5,6,10)) {

            W <- torch_randn(out_channels, in_channels, kernel_length, dtype = torch_double())

            for (stride_dilation in 1:5) {

              out_stride <- nnf_conv1d(input_pad, bias = bias, W, stride = stride_dilation)
              out_length_stride <- dim(out_stride)[3]

              out_dilation <- nnf_conv1d(input_pad, W, bias = bias, dilation = stride_dilation)
              out_length_dilation <- dim(out_dilation)[3]

              #
              #------------------------- Stride ----------------------------
              #

              #
              # Test initialization
              #
              conv1d_stride <- conv1d_layer(weight = W,
                                            bias = bias,
                                            dim_in = c(in_channels, in_length),
                                            dim_out = c(out_channels, out_length_stride),
                                            stride = stride_dilation,
                                            padding = pad,
                                            dilation = 1,
                                            activation_name = "tanh",
                                            dtype = "double")
              # Forward pass
              out <- conv1d_stride(input)
              out_ref <- conv1d_stride$update_ref(input_ref)

              #
              # Test get_input_multiplier
              #

              diff_input <- (input - input_ref)$unsqueeze(4)
              diff_output <- (out - out_ref)$unsqueeze(4)

              for (model_out in c(1,2,4)) {
                mult <- torch_randn(batch_size,out_channels, out_length_stride, model_out, dtype = torch_double())
                y_true <- sum(diff_output * mult, dim = 2:3)

                # Rescale rule
                input_mult <- conv1d_stride$get_input_multiplier(mult)
                y <- sum(input_mult * diff_input, dim = 2:3)
                expect_equal(dim(input_mult), c(batch_size, in_channels, in_length, model_out))
                expect_lt(as_array(mean((y - y_true)^2)), 1e-12)

                # Reveal-Cancel rule
                input_mult <- conv1d_stride$get_input_multiplier(mult, rule_name = "reveal_cancel")
                y <- sum(input_mult * diff_input, dim = 2:3)
                expect_equal(dim(input_mult), c(batch_size, in_channels, in_length, model_out))
                expect_lt(as_array(mean((y - y_true)^2)), 1e-12)
              }

              #
              #-------------------------- Dilation -----------------------------
              #

              #
              # Test initialization
              #
              conv1d_dilation <- conv1d_layer(weight = W,
                                              bias = bias,
                                              dim_in = c(in_channels, in_length),
                                              dim_out = c(out_channels, out_length_dilation),
                                              stride = 1,
                                              padding = pad,
                                              dilation = stride_dilation,
                                              activation_name = "tanh",
                                              dtype = "double")
              # Forward pass
              out <- conv1d_dilation(input)
              out_ref <- conv1d_dilation$update_ref(input_ref)

              #
              # Test get_input_multiplier
              #

              diff_input <- (input - input_ref)$unsqueeze(4)
              diff_output <- (out - out_ref)$unsqueeze(4)

              for (model_out in c(1,2,4)) {
                mult <- torch_randn(batch_size,out_channels, out_length_dilation, model_out, dtype = torch_double())
                y_true <- sum(diff_output * mult, dim = 2:3)

                # Rescale rule
                input_mult <- conv1d_dilation$get_input_multiplier(mult)
                y <- sum(input_mult * diff_input, dim = 2:3)
                expect_equal(dim(input_mult), c(batch_size, in_channels, in_length, model_out))
                expect_lt(as_array(mean((y - y_true)^2)), 1e-12)

                # Reveal-Cancel rule
                input_mult <- conv1d_dilation$get_input_multiplier(mult, rule_name = "reveal_cancel")
                y <- sum(input_mult * diff_input, dim = 2:3)
                expect_equal(dim(input_mult), c(batch_size, in_channels, in_length, model_out))
                expect_lt(as_array(mean((y - y_true)^2)), 1e-12)
              }
            }
          }
        }
      }
    }
  }
})
