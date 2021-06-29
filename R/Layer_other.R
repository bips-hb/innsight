

#' @include Layer.R
#' @export
#' 

#' Torch implementation of a flatten layer
#' @description
#' This \code{torch::nn_module} implements a flatten layer in torch. It takes an 
#' input of dimension \emph{(num_batches, features_1, ..., features_n)} and flattens
#' it to an output of dimensions \emph{(num_batches, features_1 * ... * features_n)}.
flatten_layer <- torch::nn_module(
  classname = "Flatten_Layer",
  inherit = Layer,
  
  #'@title Initialize the flatten layer
  #'@name initialize
  #'@description
  #'This function initializes the attributes of the flatten layer.
  #'@param dim_in The input dimensions of the flatten layer
  #'@param output_dim The output dimensions of the flatten layer
  #'@field channels_first boolean that determines whether to unroll values beginning at the 
  #'last layer with \code{channels_first == TRUE} or at the first layer with \code{channels_first == FALSE}
  initialize = function(dim_in, dim_out) {
    self$input_dim <- dim_in
    self$output_dim <- dim_out
    self$channels_first <- TRUE
  },
  
  #
  # x: [num_batches, features_1, ..., features_n]
  # out: [num_batches, features_1 * ... * features_n]
  #
  #'@title Forward function of flatten layer
  #'@name forward
  #'@description
  #'This function takes the input and forwards it through the layer, updating the layer's preactivation and output
  #'@param x The input of dimension \emph{(num_batches, features_1, ..., features_n)}
  #'@param channels_first Boolean that determines whether to unroll values beginning at the 
  #'last layer with \code{channels_first == TRUE} or at the first layer with \code{channels_first == FALSE}
  #'@return Returns  the output of the forward pass, of dimensions \emph{(num_batches, features_1 * ... * features_n)}
  forward = function(x, channels_first = TRUE) {
    self$input <- x
    if (channels_first == FALSE) {
      x <- torch::torch_movedim(x, 2, -1)
      self$channels_first <- FALSE
    }
    else {
      self$channels_first <- TRUE
    }
    self$preactivation <- torch::torch_flatten(x, start_dim = 2)
    self$output <- self$preactivation
    
    self$output
  },
  
  #
  # x_ref: Tensor of size [features_1, ...,  features_n]
  # out: Tensor of size [features_1 * ... * features_n]
  #
  
  #'@title Updating reference value
  #'@name update_ref
  #'@description
  #'This function updates the reference input and forwards it through the layer, 
  #'updating the output and preactivation.
  #'@param x_ref The reference input to be used
  #'@param channels_first Boolean that determines whether to unroll values beginning at the 
  #'last layer with \code{channels_first == TRUE} or at the first layer with \code{channels_first == FALSE}
  #'@return This function returns the output of the forward pass
  update_ref = function(x_ref, channels_first = TRUE) {
    self$input_ref <- x_ref
    if (channels_first == FALSE) {
      x_ref <- torch::torch_movedim(x_ref, 2, -1)
    }
    self$preactivation_ref <- torch::torch_flatten(x_ref, start_dim = 2)
    self$output_ref <- self$preactivation_ref
    
    self$output_ref
  },
  
  # Arguments:
  #   output       : relevance score from the upper layer to the output, torch Tensor
  #                   : of size [batch_size, dim_out,  model_out]
  #
  #   input          : torch Tensor of size [batch_size, in_channels, * , model_out]
  #
  
  #'@title Reshape the output of the flatten layer to the input dimensions
  #'@name reshape_to_input
  #'@param output The output of the flatten layer
  #'@description
  #'This function reshapes the output of a flatten layer to reverse the flattening
  #'process and recover the original input.
  #'@return 
  #'Returns the original input of the flatten layer
  reshape_to_input = function(output) {
    batch_size <- dim(output)[1]
    model_out <- rev(dim(output))[1]
    
    if (self$channels_first == FALSE) {
      in_channels <- self$input_dim[1]
      in_dim <- c(self$input_dim[-1], in_channels)
      input <- output$reshape(c(batch_size, in_dim, model_out))
      input <- torch_movedim(input, length(self$input_dim) + 1, 2)
    }
    else {
      input <- output$reshape(c(batch_size, self$input_dim, model_out))
    }
    input
  }
)