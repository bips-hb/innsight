
#'One-dimensional convolution layer of a Neural Network
#'
#'
#'@description
#'Implementation of a one-dimensional Convolutional Neural Network layer as an R6 class
#'where input, preactivation and output values of the last forward pass are stored
#'(same for a reference input, if this is needed). Applies a torch function for 
#'forwarding an input through a 1d convolution followed by an activation function
#'\eqn{\sigma} to the input data, i.e.
#'\deqn{y= \sigma(\code{nnf_conv1d(x,W,b)})}
#'
#'@export


#' @include Layer.R
#' @export
conv1d_layer <- torch::nn_module(
  classname = "Conv1D_Layer",
  inherit = Layer,

  #
  # weight: [out_channels, in_channels, kernel_size]
  # bias  : [out_channels]
  #
  #'@description 
  #'Create a new instance of this class with given parameters of a one-dimensional 
  #'convolutional layer.
  #'@param weight The weight matrix of dimension \emph{(out_channels, in_channels, kernel_size)}
  #'@param bias The bias vector of dimension \emph{(out_channels)}
  #'@param dim_in The input dimension of the layer: \emph{(in_channels,in_length)}
  #'@param dim_out The output dimensions of the layer: \emph{(out_channels,out_length)}
  #'@param stride The stride used in the convolution, by default \code{stride=1}
  #'@param padding The padding of the layer, by default \code{padding = c(0,0)}, can be an integer or a two-dimensional tuple
  #'@param dilation The dilation of the layer, by default \code{dilation = 1}
  #'@param activation_name The name of the activation function used, by default \code{activation_name = 1}
  #'
  initialize = function(weight,
                        bias,
                        dim_in,
                        dim_out,
                        stride = 1,
                        padding = c(0,0),
                        dilation = 1,
                        activation_name = 'linear', ...) {

    # [in_channels, in_length]
    self$input_dim <- dim_in
    # [out_channels, out_length]
    self$output_dim <- dim_out
    self$in_channels <- dim(weight)[2]
    self$out_channels <- dim(weight)[1]
    self$kernel_size <- dim(weight)[-c(1,2)]

    self$W <- torch_tensor(weight, dtype = torch_float())
    self$b <- torch_tensor(bias, dtype = torch_float())

    self$stride <- stride
    self$padding <- padding
    self$dilation <- dilation

    self$get_activation(activation_name, ...)
  },

  #
  # x : Tensor [batch_size, in_channels, Length_in ]
  #
  #' @title Forward function for conv1d layer
  #' @name forward
  #' 
  #' @description 
  #' The forward function takes an input and forwards it through the layer
  #'
  #'
  #'@param x The input torch tensor of dimensions  \emph{(batch_size, in_channels, in_length )}
  #'@return returns the output of the layer with respect to the given inputs, with dimensions
  #' \emph{(batch_size,out_channels,out_length)}
  forward = function(x) {
    self$input <- x
    #Pad the input
    x <- nnf_pad(x, pad = self$padding)
    
    self$preactivation <- torch::nnf_conv1d(x, self$W, self$b, self$stride,padding=0, self$dilation)
    if (self$activation_name == 'linear') {
        self$output <- self$preactivation
    }
    else {
        self$output <- self$activation_f(self$preactivation)
    }
    self$output
  },

  #
  # x_ref: Tensor of size [in_channels, Length_in]
  #
  #'@title Update the reference value
  #'@name update_ref
  #'@description
  #'This function takes the reference input and runs it through
  #'the layer, updating the the values of input_ref and output_ref
  #'
  #'@param x_ref The new reference input, of dimensions \emph{(in_channels,in_length)}
  #'@return returns the output of the reference input after 
  #'passing through the layer, of dimension \emph{(out_channels,out_length)}
  #'
  update_ref = function(x_ref) {

    self$input_ref <- x_ref
    self$preactivation_ref <- torch::nnf_conv1d(x_ref, self$W, self$b, self$stride, self$padding, self$dilation)
    if (self$activation_name == 'linear') {
        self$output_ref <- self$preactivation_ref
    }
    else {
        self$output_ref <- self$activation_f(self$preactivation_ref)
    }

    self$output_ref
  },

  #'@title LRP relevance method for layer's inputs
  #'@name get_input_relevances
  #'@description
  #'This method uses the output layer relevances and calculates the input layer 
  #'relevances using the specified rule
  #'@param relevance The output relevances, of dimensions \emph{(batch_size, out_channels, out_length, model_out)}
  #' @param rule_name The name of the rule, with which the relevance scores are
  #' calculated. Implemented are \code{"simple"}, \code{"eps"}, \code{"ab"},
  #' \code{"ww"} (default: \code{"simple"}).
  #' @param rule_param The parameter of the selected rule. Note: Only the rules
  #' \code{"eps"} and \code{"ab"} take use of the parameter. Use the default
  #' value \code{NULL} for the default parameters ("eps" : \eqn{0.01}, "ab" : \eqn{0.5}).
  #'
  #'
  get_input_relevances = function(relevance, rule_name = 'simple', rule_param = NULL) {

    if (rule_name == 'simple') {
      z <-  self$preactivation
      # add a small stabilizer
      z <- z + (z==0)*1e-12
          
      
          rel_lower <- self$get_gradient(relevance / z$unsqueeze(4), self$W)
          rel_lower <- torch::torch_mul(rel_lower, self$input$unsqueeze(4))
        
    }
    else if (rule_name == 'epsilon') {
      # set default parameter
      if (is.null(rule_param)) {
        epsilon <- 0.001
      }
      else {
        epsilon <- rule_param
      }

      z <-  self$preactivation
      z <- z + epsilon * torch::torch_sgn(z)
 
      rel_lower <- self$get_gradient(relevance / z$unsqueeze(4), self$W)
      rel_lower <- torch::torch_mul(rel_lower, self$input$unsqueeze(4))
    }
    else if (rule_name == 'alpha_beta') {
      # set default parameter
      if (is.null(rule_param)) {
        alpha <- 0.5
      }
      else {
        alpha <- rule_param
      }

      output_partition <- self$get_pos_and_neg_outputs(self$input, use_bias = TRUE)

     z <- relevance / ( output_partition$pos + (output_partition$pos == 0) * 1e-16 )$unsqueeze(4)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      input <- self$input$unsqueeze(4)
      rel_pos <- torch::torch_mul(t1, (input * (input > 0))) +
                 torch::torch_mul(t2, (input * (input <= 0)))


      z <- relevance / ( output_partition$neg + (output_partition$neg == 0) * 1e-16 )$unsqueeze(4)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- torch::torch_mul(t1, (input * (input <= 0))) +
                 torch::torch_mul(t2, (input * (input > 0)))



    }

    rel_lower
  },
  
  #'@title Get gradient method 
  #'@name get_gradient
  #'@description
  #'This method uses \code{nnf_conv_transpose1d} to multiply the input with the 
  #'gradient of a layer's output with respect to the layer's input. 
  #'
  #'@param input A relevance tensor of dimension \emph{(batch_size, out_channels, out_length, model_out)}
  #'@param weight A weight tensor of dimensions \emph{(out_channels, in_channels, kernel_size)}
  #'
  #'@return 
  #'This returns the 1d transpose of the input using \code{nnf_conv_transpose1d} with params
  #'\code{input}, \code{weight$unsqueeze(4)}
  #'

  get_gradient = function(input, weight) {
    
    # Since we have added the model_out dimension, strides and dilation need to
    # be extended by 1.
    
    stride <- c(self$stride,1)
    
    # dilation is a number or a tuple of length 2
    dilation <- c(self$dilation,1)
    
    out <- torch::nnf_conv_transpose2d(input, weight$unsqueeze(4),
                                       bias = NULL,
                                       stride = stride,
                                       padding = 0,
                                       dilation = dilation)
    
    # If stride is > 1, it could happen that the reconstructed input after
    # padding (out) lost some dimensions, because multiple input shapes are
    # mapped to the same output shape. Therefore, we use padding with zeros to
    # fill in the missing irrelevant input values.
    lost_length <- self$input_dim[2] + self$padding[1] + self$padding[2] - dim(out)[3]

    out <- torch::nnf_pad(out, pad = c(0,0,0, lost_length))
    # Now we have added the missing values such that dim(out) = dim(padded_input)
    
    # Apply the inverse padding to obtain dim(out) = dim(input)
    dim_out <- dim(out)
    
    out[,,(self$padding[1]+1):(dim_out[3]-self$padding[2]),]
    
    
    out
  },

  get_pos_and_neg_outputs = function(input, use_bias = FALSE) {
    output <- NULL

    if (use_bias == TRUE) {
      b_pos <- self$b * (self$b > 0) * 0.5
      b_neg <- self$b * (self$b <= 0) * 0.5
    }
    else {
      b_pos <- NULL
      b_neg <- NULL
    }

    conv1d <- function(x, W, b) {
      out <- torch::nnf_conv1d(x, W,
                              bias = b,
                              stride = self$stride,
                              padding = self$padding,
                              dilation = self$dilation)
      out
    }

    # input (+) x weight (+) and input (-) x weight (-)
    output$pos <- conv1d(input * (input > 0), self$W * (self$W > 0), b_pos) +
                  conv1d(input * (input < 0), self$W * (self$W < 0), b_pos)

    # input (+) x weight (-) and input (-) x weight (+)
    output$neg <- conv1d(input * (input > 0), self$W * (self$W < 0), b_neg) +
                  conv1d(input * (input < 0), self$W * (self$W > 0), b_neg)

    output
  }
)

