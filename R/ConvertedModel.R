
###############################################################################
#                           Converted Torch Model
###############################################################################


#' Converted torch-based model
#'
#' This class stores all layers converted to torch in a module which can be
#' used like the original model (but torch-based). In addition, it provides
#' other functions that are useful for interpreting individual predictions or
#' explaining the entire model. This model is part of the class [Converter]
#' and is the core for all the necessary calculations in the methods provided
#' in this package.
#'
#' @param modules_list A list of all accepted layers created by the 'Converter'
#' class during initialization.
#' @param dtype The data type for all the calculations and defined tensors. Use
#' either `'float'` for [torch::torch_float] or `'double'` for
#' [torch::torch_double].
#'
#' @section Public fields:
#' \describe{
#'   \item{`modules_list`}{A list of all accepted layers created by the
#'   'Converter' class during initialization.}
#'   \item{`dtype`}{The datatype for all the calculations and defined
#'   tensors. Either `'float'` for [torch::torch_float] or `'double'` for
#'   [torch::torch_double]}.
#' }
#'
ConvertedModel <- nn_module(
  classname = "ConvertedModel",
  modules_list = NULL,
  graph = NULL,
  input_nodes = NULL,
  output_nodes = NULL,
  output_order = NULL,
  dtype = NULL,
  initialize = function(modules_list, graph, input_nodes, output_nodes, dtype = "float") {
    self$modules_list <- modules_list
    self$graph <- graph
    self$input_nodes <- input_nodes
    self$output_nodes <- output_nodes
    self$dtype <- dtype

    # Calculate output order
    last_step <- graph[[length(graph)]]
    graph_output_nodes <- last_step$current_nodes
    graph_output_nodes[[last_step$used_idx]] <- last_step$used_node
    graph_output_nodes <- unlist(graph_output_nodes)
    self$output_order <- match(output_nodes, graph_output_nodes)
  },

  ### -------------------------forward and update------------------------------
  #'
  #' @section Method `forward()`:
  #'
  #' The forward method of the whole model, i.e. it calculates the output
  #' \eqn{y=f(x)} of a given input \eqn{x}. In doing so, all intermediate
  #' values are stored in the individual torch modules from `modules_list`.
  #'
  #' ## Usage
  #' `self(x, channels_first = TRUE)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x`}{The input torch tensor of dimensions
  #'   \emph{(batch_size, dim_in)}.}
  #'   \item{`channels_first`}{If the input tensor `x` is given in the format
  #'   'channels first' use `TRUE`. Otherwise, if the channels are last,
  #'   use `FALSE` and the input will be transformed into the format 'channels
  #'   first'. Default: `TRUE`.}
  #' }
  #'
  #' ## Return
  #' Returns the output of the model with respect to the given inputs with
  #' dimensions \emph{(batch_size, dim_out)}.
  #'
  forward = function(x, channels_first = TRUE, save_input = FALSE,
                     save_preactivation = FALSE, save_output = FALSE,
                     save_last_layer = FALSE) {
    # Convert the input to a list
    if (!is.list(x)) {
      x <- list(x)
    }

    # If the channels are last, we have to move the channel axis to the first
    # position after the batch dimension
    if (channels_first == FALSE) {
      x <- lapply(x, function(x_i) torch_movedim(x_i, source = -1, destination = 2))
    }

    # This is the main part of the forward pass.
    # We move the graph forward step by step whereby each step correspond to
    # one layer
    for (step in self$graph) {
      # get the necessary inputs for the current layer
      input <- x[step$used_idx]
      # if this layer has only one input (and not a list of inputs), we need
      # to convert it back to an tensor instead of a list
      if (length(input) == 1) {
        input <- input[[1]]
      }
      # If the current layer is an output layer and we want to save the
      # intermediate values of the last layer (independent of the treatment of
      # the previous layers)
      if (save_last_layer & any(step$used_node %in% self$output_nodes)) {
        out <- self$modules_list[[step$used_node]](
          input,
          channels_first = channels_first,
          save_input = save_input,
          save_preactivation = TRUE,
          save_output = TRUE)
      }
      # Otherwise we use the normal forward pass
      else {
        out <- self$modules_list[[step$used_node]](
          input,
          channels_first = channels_first,
          save_input = save_input,
          save_preactivation = save_preactivation,
          save_output = save_output)
      }

      # Remove the used inputs from our original input
      x <- x[-step$used_idx]
      # Add the layer output to the input x, hence we can proceed propagation
      # in the next step
      x <- append(x, rep(list(out), step$times),
                  after = min(step$used_idx) - 1)
    }

    # Make sure that we have the correct order of the output
    x[self$output_order]
  },

  #'
  #' @section Method `update_ref()`:
  #'
  #' This method updates the stored intermediate values in each module from the
  #' list `modules_list` when the reference input `x_ref` has changed.
  #'
  #' ## Usage
  #' `self$update_ref(x_ref, channels_first = TRUE)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x_ref`}{Reference input of the model of dimensions
  #'   \emph{(1, dim_in)}.}
  #'   \item{`channels_first`}{If the reference input tensor `x` is given in
  #'   the format 'channels first' use `TRUE`. Otherwise, if the channels are
  #'   last, use `FALSE` and the input will be transformed into the format
  #'   'channels first'. Default: `TRUE`.}
  #' }
  #'
  #' ## Return
  #' Returns the output of the reference input with dimension
  #' \emph{(1, dim_out)} after passing through the model.
  #'
  update_ref = function(x_ref, channels_first = TRUE, save_input = FALSE,
                        save_preactivation = FALSE, save_output = FALSE,
                        save_last_layer = FALSE) {
    # Convert the input to a list
    if (!is.list(x_ref)) {
      x_ref <- list(x_ref)
    }

    # If the channels are last, we have to move the channel axis to the first
    # position after the batch dimension
    if (channels_first == FALSE) {
      x_ref <- lapply(x_ref, function(x_i) torch_movedim(x_i, source = -1, destination = 2))
    }

    # This is the main part of the forward pass.
    # We move the graph forward step by step whereby each step correspond to
    # one layer
    for (step in self$graph) {
      # get the necessary inputs for the current layer
      input <- x_ref[step$used_idx]
      # if this layer has only one input (and not a list of inputs), we need
      # to convert it back to an tensor instead of a list
      if (length(input) == 1) {
        input <- input[[1]]
      }
      # If the current layer is an output layer and we want to save the
      # intermediate values of the last layer (independent of the treatment of
      # the previous layers)
      if (save_last_layer & any(step$used_node %in% self$output_nodes)) {
        out <- self$modules_list[[step$used_node]]$update_ref(
          input,
          channels_first = channels_first,
          save_input = save_input,
          save_preactivation = TRUE,
          save_output = TRUE)
      }
      # Otherwise we use the normal forward pass
      else {
        out <- self$modules_list[[step$used_node]]$update_ref(
          input,
          channels_first = channels_first,
          save_input = save_input,
          save_preactivation = save_preactivation,
          save_output = save_output)
      }

      # Remove the used inputs from our original input
      x_ref <- x_ref[-step$used_idx]
      # Add the layer output to the input x, hence we can proceed propagation
      # in the next step
      x_ref <- append(x_ref, rep(list(out), step$times),
                      after = min(step$used_idx) - 1)
    }

    # Make sure that we have the correct order of the output
    x_ref[self$output_order]
  },

  #'
  #' @section Method `set_dtype()`:
  #'
  #' This method changes the data type for all the layers in `modules_list`.
  #' Use either `'float'` for [torch::torch_float] or `'double'` for
  #' [torch::torch_double].
  #'
  #' ## Usage
  #' `self$set_dtype(dtype)`
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`dtype`}{The data type for all the calculations and defined
  #'   tensors.}
  #' }
  #'
  set_dtype = function(dtype) {
    for (module in self$modules_list) {
      module$set_dtype(dtype)
    }
    self$dtype <- dtype
  },

  reset = function() {
    for (module in self$modules_list) {
      module$reset()
    }
  }
)
