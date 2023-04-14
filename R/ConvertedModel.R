
###############################################################################
#                           Converted Torch Model
###############################################################################


#' Converted torch-based model
#'
#' This class stores all layers converted to `torch` in a module which can be
#' used like the original model (but `torch`-based). In addition, it provides
#' other functions that are useful for interpreting individual predictions or
#' explaining the entire model. This model is part of the class [`Converter`]
#' and is the core for all the necessary calculations in the methods provided
#' in this package.
#'
#' @param modules_list (`list`)\cr
#' A list of all accepted layers created by the [`Converter`]
#' class during initialization.\cr
#' @param dtype (`character(1)`)\cr
#' The data type for all the calculations and defined tensors. Use
#' either `'float'` for [`torch::torch_float`] or `'double'` for
#' [`torch::torch_double`].\cr
#' @param graph (`list`)\cr
#' The `graph` argument gives a way to pass an input through
#' the model, which is especially relevant for non-sequential architectures.
#' It can be seen as a list of steps in which order the layers from
#' `modules_list` must be applied. The list contains the following elements:
#' - `$current_nodes`\cr
#' This list describes the current position and the number
#' of the respective intermediate values when passing through the model.
#' For example, `list(1,3,3)` means that in this step one output from the
#' first layer and two from the third layer (the numbers correspond to the
#' list indices from the `modules_list` argument) are available for
#' the calculation of the current layer with index `used_node`.
#' - `$used_node`\cr
#' The index of the layer from the `modules_list` argument
#' which will be applied in this step.
#' - `$used_idx`\cr
#' The indices of the outputs from `current_nodes`, which are
#' used as inputs of the current layer (`used_node`).
#' - `$times`\cr
#' The frequency of the output value, i.e., is the output used
#' more than once as an input for subsequent layers?\cr
#' @param input_nodes (`numeric`)\cr
#' A vector of layer indices describing the input layers,
#' i.e., they are used as the starting point for the calculations.\cr
#' @param output_nodes (`numeric`)\cr
#' A vector of layer indices describing the indices
#' of the output layers.\cr
#'
ConvertedModel <- nn_module(
  classname = "ConvertedModel",
  modules_list = NULL,
  graph = NULL,
  input_nodes = NULL,
  output_nodes = NULL,
  output_order = NULL,
  dtype = NULL,
  initialize = function(modules_list, graph, input_nodes, output_nodes,
                        dtype = "float") {
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
  #' @section Method `forward()`:
  #'
  #' The forward method of the whole model, i.e., it calculates the output
  #' \eqn{y=f(x)} of a given input \eqn{x}. In doing so, all intermediate
  #' values are stored in the individual torch modules from `modules_list`.
  #'
  #' ## Usage
  #' ```
  #' self(x,
  #'      channels_first = TRUE,
  #'      save_input = FALSE,
  #'      save_preactivation = FALSE,
  #'      save_output = FAlSE,
  #'      save_last_layer = FALSE)
  #' ```
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x`}{The input torch tensor for this model.}
  #'   \item{`channels_first`}{If the input tensor `x` is given in the format
  #'   'channels first', use `TRUE`. Otherwise, if the channels are last,
  #'   use `FALSE` and the input will be transformed into the format 'channels
  #'   first'. Default: `TRUE`.}
  #'   \item{`save_input`}{Logical value whether the inputs from each layer
  #'   are to be saved or not. Default: `FALSE`.}
  #'   \item{`save_preactivation`}{Logical value whether the preactivations
  #'   from each layer are to be saved or not. Default: `FALSE`.}
  #'   \item{`save_output`}{Logical value whether the outputs from each layer
  #'   are to be saved or not. Default: `FALSE`.}
  #'   \item{`save_last_layer`}{Logical value whether the inputs,
  #'   preactivations and outputs from the last layer are to be saved or not.
  #'   Default: `FALSE`.}
  #' }
  #'
  #' ## Return
  #' Returns a list of the output values of the model with respect to the
  #' given inputs.
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
      x <- lapply(x,
                  function(z) torch_movedim(z, source = -1, destination = 2))
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
      } else {
        # Otherwise we use the normal forward pass
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
        after = min(step$used_idx) - 1
      )
    }

    # Make sure that we have the correct order of the output
    x[self$output_order]
  },

  #' @section Method `update_ref()`:
  #'
  #' This method updates the intermediate values in each module from the
  #' list `modules_list` for the reference input `x_ref` and returns the
  #' output from it in the same way as in the `forward` method.
  #'
  #' ## Usage
  #' ```
  #' self$update_ref(x_ref,
  #'                 channels_first = TRUE,
  #'                 save_input = FALSE,
  #'                 save_preactivation = FALSE,
  #'                 save_output = FAlSE,
  #'                 save_last_layer = FALSE)
  #' ```
  #'
  #' ## Arguments
  #' \describe{
  #'   \item{`x_ref`}{Reference input of the model.}
  #'   \item{`channels_first`}{If the tensor `x_ref` is given in the format
  #'   'channels first' use `TRUE`. Otherwise, if the channels are last,
  #'   use `FALSE` and the input will be transformed into the format 'channels
  #'   first'. Default: `TRUE`.}
  #'   \item{`save_input`}{Logical value whether the inputs from each layer
  #'   are to be saved or not. Default: `FALSE`.}
  #'   \item{`save_preactivation`}{Logical value whether the preactivations
  #'   from each layer are to be saved or not. Default: `FALSE`.}
  #'   \item{`save_output`}{Logical value whether the outputs from each layer
  #'   are to be saved or not. Default: `FALSE`.}
  #'   \item{`save_last_layer`}{Logical value whether the inputs,
  #'   preactivations and outputs from the last layer are to be saved or not.
  #'   Default: `FALSE`.}
  #' }
  #'
  #' ## Return
  #' Returns a list of the output values of the model with respect to the
  #' given reference input.
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
      x_ref <- lapply(
        x_ref,
        function(z) torch_movedim(z, source = -1, destination = 2))
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
          save_output = TRUE
        )
      } else {
        # Otherwise we use the normal forward pass
        out <- self$modules_list[[step$used_node]]$update_ref(
          input,
          channels_first = channels_first,
          save_input = save_input,
          save_preactivation = save_preactivation,
          save_output = save_output
        )
      }

      # Remove the used inputs from our original input
      x_ref <- x_ref[-step$used_idx]
      # Add the layer output to the input x, hence we can proceed propagation
      # in the next step
      x_ref <- append(x_ref, rep(list(out), step$times),
        after = min(step$used_idx) - 1
      )
    }

    # Make sure that we have the correct order of the output
    x_ref[self$output_order]
  },

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
