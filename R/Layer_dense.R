#'
#'Dense layer of a convolutional neural network
#'
#'
#'Implementation of a dense Neural Network layer as a torch module
#'where input, preactivation and output values of the last forward pass are stored
#'(same for a reference input, if this is needed). Applies a torch function for
#'forwarding an input through a linear function followed by an activation function
#'\eqn{\sigma} to the input data, i.e.
#'\deqn{y= \sigma(\code{nnf_dense(x,W,b)})}
#'
#'@export


#' @include Layer.R
#' @export
dense_layer <- torch::nn_module(
  classname = "Dense_Layer",
  inherit = Layer,

  #
  # weight: [out_features, in_features]
  # bias  : [out_features]
  #
  
  #'@title Initialize the module
  #'@name initialize
  #'@param weight The weight matrix of dimensions \emph{(out_features,in_features)}
  #'@param bias The bias vector of dimension \emph{(out_features)}
  #'@param dim_in the input dimension of the layer
  #'@param dim_out the output dimension of the layer
  #'@param activation_name The name of the activation function used by the layer, by default \code{activation_name = 1}
  #'
  initialize = function(weight, bias, activation_name, dtype = "float") {
    self$input_dim <- dim(weight)[2]
    self$output_dim <- dim(weight)[1]
    self$get_activation(activation_name)

    # Check if weight is already a tensor
    if (!inherits(weight, "torch_tensor")) {
      self$W <- torch::torch_tensor(weight)
    }
    else {
      self$W <- weight
    }
    # Check if bias is already a tensor
    if (!inherits(bias, "torch_tensor")) {
      self$b <- torch::torch_tensor(bias)
    }
    else {
      self$b <- bias
    }

    self$set_dtype(dtype)
  },
  
 #'@title set the data type of weight and bias
 #'@name set_dtype
 #'@description
 #'This function changes the data type of the weight and bias tensor to be either float or double
 #'@param dtype The name of the data type the weight and bias tensor are to be changed into. Can be \emph{"float"} or \emph{"double"}
 #'

  set_dtype = function(dtype) {
    if (dtype == "float") {
      self$W <- self$W$to(torch::torch_float())
      self$b <- self$b$to(torch::torch_float())
    }
    else if (dtype == "double") {
      self$W <- self$W$to(torch::torch_double())
      self$b <- self$b$to(torch::torch_double())
    }
    else {
      stop(sprintf("Unknown argument for 'dtype' : %s . Use 'float' or 'double' instead"))
    }
    self$dtype <- dtype
  },

  #
  # x: [batch_size, in_features]
  #
  
  #' @title Forward function for dense layer
  #' @name forward
  #'
  #' @description
  #' The forward function takes an input and forwards it through the layer
  #'
  #'
  #'@param x The input torch tensor of dimensions  \emph{(batch_size, in_features )}
  #'@return returns the output of the layer with respect to the given inputs, with dimensions
  #' \emph{(batch_size,out_features)}
  forward = function(x) {

    self$input <- x
    self$preactivation <- torch::nnf_linear(x, self$W, self$b)
    self$output <- self$activation_f(self$preactivation)

    self$output
  },

  #
  # x_ref: Tensor of size [1,in_features]
  #
  
  #'@title Update the reference value
  #'@name update_ref
  #'@description
  #'This function takes the reference input and runs it through
  #'the layer, updating the the values of input_ref and output_ref
  #'
  #'@param x_ref The new reference input, of dimensions \emph{(1,in_features)}
  #'@return returns the output of the reference input after
  #'passing through the layer, of dimension \emph{(1,out_features)}
  #'
  update_ref = function(x_ref) {

    self$input_ref <- x_ref
    self$preactivation_ref <- torch::nnf_linear(x_ref, self$W, self$b)
    self$output_ref <- self$activation_f(self$preactivation_ref)

    self$output_ref
  },

  #
  #   rel_output   [batch_size, dim_out, model_out]
  #
  #   output       [batch_size, dim_in, model_out]
  #
  
  #'@title LRP relevance method for layer's inputs
  #'@name get_input_relevances
  #'@description
  #'This method uses the output layer relevances and calculates the input layer
  #'relevances using the specified rule
  #' @param rel_output The output relevances, of dimensions \emph{(batch_size, out_features, model_out)}
  #' @param rule_name The name of the rule, with which the relevance scores are
  #' calculated. Implemented are \code{"simple"}, \code{"epsilon"}, \code{"alpha_beta"}
  #'  (default: \code{"simple"}).
  #' @param rule_param The parameter of the selected rule. Note: Only the rules
  #' \code{"epsilon"} and \code{"alpha_beta"} take use of the parameter. Use the default
  #' value \code{NULL} for the default parameters ("epsilon" : \eqn{0.01}, "alpha_beta" : \eqn{0.5}).
  #'
  #'
  get_input_relevances = function(rel_output, rule_name = 'simple', rule_param = NULL) {

    if (is.null(rule_param)) {
      if (rule_name == "epsilon"){
        rule_param = 0.001
      }
      else if (rule_name == "alpha_beta") {
        rule_param = 0.5
      }
    }

    input <- self$input$unsqueeze(3)
    z <- self$preactivation$unsqueeze(3)

    if(rule_name == "simple"){
      z <- z + (z == 0) * 1e-16

      rel_input <-
        self$get_gradient(rel_output / z, self$W) * input

    }
    else if(rule_name == "epsilon"){
      z <- z + rule_param * torch::torch_sgn(z) + (z == 0) * 1e-16
      rel_input <-
        self$get_gradient(rel_output / z, self$W) * input
    }
    else if(rule_name == "alpha_beta"){

      out_part <- self$get_pos_and_neg_outputs(self$input)

      # Apply the simple rule for each part:
      # - positive part
      z <- rel_output / ( out_part$pos + (out_part$pos == 0) * 1e-16 )$unsqueeze(3)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_pos <- t1 *  (input * (input > 0)) + t2 * (input * (input <= 0))

      # - negative part
      z <- rel_output / ( out_part$neg + (out_part$neg == 0) * 1e-16 )$unsqueeze(3)

      t1 <- self$get_gradient(z, (self$W * (self$W > 0)))
      t2 <- self$get_gradient(z, (self$W * (self$W <= 0)))

      rel_neg <- t1 * (input * (input <= 0)) + t2 * (input * (input > 0))

      # calculate over all relevance for the lower layer
      rel_input <- rel_pos * rule_param + rel_neg * (1 - rule_param)
    }
    #else if(rule_name == "ww"){
    #  relevance <- torch_matmul(( torch_transpose( (torch_square(weights) / torch_sum(torch_transpose(torch_square(weights),1,2),2,keepdim=FALSE)) ,1,2)) , relevance)
    #}


    rel_input
  },

  #
  #   mult_output   [batch_size, dim_out, model_out]
  #
  #   output        [batch_size, dim_in, model_out]
  #
  get_input_multiplier = function(mult_output, rule_name = "rescale") {

    #
    # --------------------- Non-linear part---------------------------
    #
    mult_pos <- mult_output
    mult_neg <- mult_output
    if (self$activation_name != "linear") {
      if (rule_name == "rescale") {
        delta_output <- (self$output - self$output_ref)$unsqueeze(3)
        delta_preact <- (self$preactivation - self$preactivation_ref)$unsqueeze(3)

        nonlin_mult <- delta_output / (delta_preact + 1e-16 * (delta_preact == 0))

        mult_pos <- mult_output * nonlin_mult
        mult_neg <- mult_output * nonlin_mult

      }
      else if (rule_name == "reveal_cancel") {
        act <- self$activation_f
        x <- self$preactivation
        x_ref <- self$preactivation_ref
        delta_x <- self$get_pos_and_neg_outputs(self$input - self$input_ref)

        delta_output_pos <-
          0.5 * (act(x_ref + delta_x$pos) - act(x_ref)) +
          0.5 * (act(x) - act(x_ref + delta_x$neg))

        delta_output_neg <-
          0.5 * (act(x_ref + delta_x$neg) - act(x_ref)) +
          0.5 * (act(x) - act(x_ref + delta_x$pos))

        mult_pos <- mult_output * (delta_output_pos / (delta_x$pos + 1e-16))$unsqueeze(3)
        mult_neg <- mult_output * (delta_output_neg / (delta_x$neg - 1e-16))$unsqueeze(3)
      }
    }

    #
    # -------------- Linear part -----------------------
    #

    # input        [batch_size, dim_in]
    # delta_input  [batch_size, dim_in, 1]
    delta_input <- (self$input - self$input_ref)$unsqueeze(3)

    # mult_input    [batch_size, model_out, dim_in]
    mult_input <-
      self$get_gradient(mult_pos, self$W * (self$W > 0)) * (delta_input > 0) +
      self$get_gradient(mult_pos, self$W * (self$W < 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, self$W * (self$W > 0)) * (delta_input < 0) +
      self$get_gradient(mult_neg, self$W * (self$W < 0)) * (delta_input > 0) +
      self$get_gradient(0.5 * (mult_pos + mult_neg), self$W) * (delta_input == 0)

    mult_input
  },

  #
  #   grad_out   [batch_size, dim_out, model_out]
  #   weight     [dim_out, dim_in]
  #
  #   output  [batch_size, dim_in, model_out]
  #
  
  #'@title Get gradient method
  #'@name get_gradient
  #'@description
  #'This method uses \code{torch::torch_matmul} to multiply the input with the
  #'gradient of a layer's output with respect to the layer's input.
  #'
  #'@param grad_out A tensor of dimension \emph{(batch_size, out_features, model_out)}
  #'@param weight A weight tensor of dimensions \emph{(out_features, in_features)}
  #'
  #'@return
  #'This returns the result of \code{torch:torch_matmul(weight$t(),grad_out)}
  #'
  get_gradient = function(grad_out, weight) {
    grad_in <- torch::torch_matmul(weight$t(), grad_out)

    grad_in
  },
  
  #'@title Get positive and negative parts of the output
  #'@name get_pos_and_neg_outputs
  #'@description
  #'This method uses the \code{torch::nnf_lienar()} method to divide the output of a linear pass of the input
  #'into its positive and negative parts
  #'@param input The input of the dense layer, of dimension \emph{(batch_size, in_features)}
  #'@param use_bias If \code{use_bias == TRUE} the bias is used in the forward passes, if \code{use_bias == FALSE} it is omitted
  #'@return Returns an output object with positive and negative parts of the output stored in \code{output$pos} and \code{output$neg} respectively
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

    W <- self$W

    output$pos <-
      torch::nnf_linear(input * (input > 0), W * (W > 0), bias = b_pos) +
      torch::nnf_linear(input * (input < 0), W * (W < 0), bias = b_pos)

    output$neg <-
      torch::nnf_linear(input * (input > 0), W * (W < 0), bias = b_neg) +
      torch::nnf_linear(input * (input < 0), W * (W > 0), bias = b_neg)

    output
  }
)