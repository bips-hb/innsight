test_that("Test LRP for Dense-Layers",{
  library(torch)
  batch_size = 10
  for(dim_in in c(1,2,10,20)){
    for(dim_out in c(1,3,10,30)){
      for(activation_name in c("relu","tanh","softplus")){

        #
        # Test for dtype = "float"
        #

        W <- torch_randn(dim_out,dim_in)
        b <- torch_randn(dim_out)
        dense <- dense_layer(weight = W, bias = b, activation_name,dtype = "float")

        input <- torch_randn(batch_size, dim_in, dtype = torch_float())
        input_ref <- torch_randn(1, dim_in, dtype = torch_float())

        #
        # Test forward
        #
        y <- dense(input)
        expect_equal(dim(y), c(batch_size, dim_out))

        #
        # Test update_ref
        #
        y <- dense$update_ref(input_ref)
        expect_equal(dim(y), c(1, dim_out))

        #
        # Test get_gradient
        #
        input$requires_grad <- TRUE
        dense(input)
        y <- dense$preactivation
        sum(y)$backward()
        y_true <- input$grad$unsqueeze(3)
        y <- dense$get_gradient(torch_ones(batch_size, dim_out, 1), W)
        expect_lt(as_array(mean((y - y_true)^2)), 1e-8)
        input$requires_grad <- FALSE

        #
        # Test get_pos_and_neg_outputs
        #
        out <- dense$get_pos_and_neg_outputs(input, use_bias = TRUE)
        expect_equal(dim(out$pos), c(batch_size, dim_out))
        expect_equal(dim(out$neg), c(batch_size, dim_out))
        expect_lt(as_array(mean((out$pos + out$neg - dense$preactivation)^2)), 1e-8)

        out <- dense$get_pos_and_neg_outputs(input, use_bias = FALSE)
        expect_equal(dim(out$pos), c(batch_size, dim_out))
        expect_equal(dim(out$neg), c(batch_size, dim_out))
        expect_lt(as_array(mean((out$pos + out$neg - dense$preactivation + b)^2)), 1e-8)

        #
        # get_input_relevances
        #
        dense$b <- dense$b * 0
        dense(input)
        dense$update_ref(input_ref)

        for (model_out in c(1,3)) {

          rel <- torch_randn(batch_size, dim_out, model_out)

          # Simple rule
          y <- dense$get_input_relevances(rel)
          expect_equal(dim(y), c(batch_size, dim_in, model_out))
          rel_in <- sum(y, dim = 2:3)
          rel_out <- sum(rel, dim = 2:3)
          expect_lt(as_array(mean((rel_in - rel_out)^2)), 1e-8)

          # Epsilon rule
          y <- dense$get_input_relevances(rel, rule_name = "epsilon")
          expect_equal(dim(y), c(batch_size, dim_in, model_out))

          # Alpha_Beta rule
          y <- dense$get_input_relevances(rel, rule_name = "alpha_beta")
          expect_equal(dim(y), c(batch_size, dim_in, model_out))
        }

        #
        # Test get_input_multiplier
        #

        for (model_out in c(1,3)) {

          mult <- torch_randn(batch_size, dim_out, model_out)
          diff_input <- (input - input_ref)$unsqueeze(3)
          diff_output <- (dense$output - dense$output_ref)$unsqueeze(3)

          y_true <- sum(mult * diff_output, dim = 2:3)

          # Rescale rule
          mult_in <- dense$get_input_multiplier(mult)
          expect_equal(dim(mult_in), c(batch_size, dim_in, model_out))
          y <- sum(mult_in * diff_input, sum = 2:3)
          expect_lt(as_array(mean((y - y_true)^2)), 1e-8)

          # Reveal Cancel rule
          mult_in <- dense$get_input_multiplier(mult, rule_name = "reveal_cancel")
          expect_equal(dim(mult_in), c(batch_size, dim_in, model_out))
          y <- sum(mult_in * diff_input, sum = 2:3)
          expect_lt(as_array(mean((y - y_true)^2)), 1e-8)
        }
      }
    }
  }
})
