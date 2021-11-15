#' Converter of an artificial Neural Network
#'
#' @description
#' This class analyzes a passed neural network and stores its internal
#' structure and the individual layers by converting the entire network into a
#' \code{\link[torch]{nn_module}}. With the help of this converter, many
#' methods for interpreting the behavior of neural networks are provided, which
#' give a better understanding of the whole model or individual predictions.
#' You can use models from the following libraries:
#' * `torch` (\code{\link[torch]{nn_sequential}})
#' * \code{\link[keras]{keras}} (\code{\link[keras]{keras_model}},
#' \code{\link[keras]{keras_model_sequential}}),
#' * \code{\link[neuralnet]{neuralnet}}
#'
#' Furthermore, a model can be passed as a list (see details for more
#' information).
#'
#' @details
#'
#' In order to better understand and analyze the prediction of a neural
#' network, the preactivation or other information of the individual layers,
#' which are not stored in an ordinary forward pass, are often required. For
#' this reason, a given neural network is converted into a torch-based neural
#' network, which provides all the necessary information for an interpretation.
#' The converted torch model is stored in the field `model` and is an instance
#' of \code{\link[innsight:ConvertedModel]{innsight::ConvertedModel}}.
#' However, before the torch model is created, all relevant details of the
#' passed model are extracted into a named list. This list can be saved
#' in complete form in the `model_dict` field with the argument
#' `save_model_as_list`, but this may consume a lot of memory for large
#' networks and is not done by default. Also, this named list can again be
#' used as a passed model for the class `Converter`, which will be described
#' in more detail in the section 'Implemented Libraries'.
#'
#' ## Implemented Methods
#' An object of the Converter class can be applied to the
#' following methods:
#'   * Layerwise Relevance Propagation ([LRP]), Bach et al. (2015)
#'   * Deep Learning Important Feartures ([DeepLift]), Shrikumar et al. (2017)
#'   * [SmoothGrad] including 'SmoothGrad x Input', Smilkov et al. (2017)
#'   * Vanilla [Gradient] including 'Gradient x Input'
#'   * [ConnectionWeights], Olden et al. (2004)
#'
#'
#' ## Implemented Libraries
#' The converter is implemented for models from the libraries
#' \code{\link[torch]{nn_sequential}},
#' \code{\link[neuralnet]{neuralnet}} and \code{\link[keras]{keras}}. But you
#' can also write a wrapper for other libraries because a model can be passed
#' as a named list with the following components:
#'
#' * **`$input_dim`**\cr
#' An integer vector with the model input dimension, e.g. for
#' a dense layer with 5 input features use `c(5)` or for  a 1D-convolutional
#' layer with signal length 50 and 4 channels use `c(4,50)`.
#'
#' * **`$input_names`** (optional)\cr
#' A list with the names for each input dimension, e.g. for
#' a dense layer with 3 input features use `list(c("X1", "X2", "X3"))` or for a
#' 1D-convolutional layer with signal length 5 and 2 channels use
#' `list(c("C1", "C2"), c("L1","L2","L3","L4","L5"))`. By default (`NULL`)
#' the names are generated.
#'
#' * **`$output_dim`** (optional)\cr
#' An integer vector with the model output dimension analogous to `$input_dim`.
#' This value does not need to be specified. But if it is set, the calculated
#' value will be compared with it to avoid errors during converting.
#'
#' * **`$output_names`** (optional)\cr
#' A list with the names for each output dimension analogous to `$input_names`.
#' By default (`NULL`) the names are generated.
#'
#' * **`$layers`**\cr
#' A list with the respective layers of the model. Each layer is represented as
#' another list that requires the following entries depending on the type:
#'   * **Dense Layer:**
#'      * **`$type`**: `'Dense'`
#'      * **`$weight`**: The weight matrix of the dense layer with shape
#'      (`dim_out`, `dim_in`).
#'      * **`$bias`**: The bias vector of the dense layer with length
#'      `dim_out`.
#'      * **`activation_name`**: The name of the activation function for this
#'      dense layer, e.g. `'relu'`, `'tanh'` or `'softmax'`.
#'      * **`dim_in`** (optional): The input dimension of this layer. This
#'      value is not necessary, but helpful to check the format of the weight
#'      matrix.
#'      * **`dim_out`** (optional): The output dimension of this layer. This
#'      value is not necessary, but helpful to check the format of the weight
#'      matrix.
#'
#'   * **Convolutional Layers:**
#'      * **`$type`**: `'Conv1D'` or `'Conv2D'`
#'      * **`$weight`**: The weight array of the convolutional layer with shape
#'      (`out_channels`, `in_channels`, `kernel_length`) for 1D or
#'      (`out_channels`, `in_channels`, `kernel_height`, `kernel_width`) for
#'      2D.
#'      * **`$bias`**: The bias vector of the layer with length `out_channels`.
#'      * **`$activation_name`**: The name of the activation function for this
#'      layer, e.g. `'relu'`, `'tanh'` or `'softmax'`.
#'      * **`$dim_in`** (optional): The input dimension of this layer according
#'      to the format (`in_channels`, `in_length`) for 1D or
#'      (`in_channels`, `in_height`, `in_width`) for 2D.
#'      * **`$dim_out`** (optional): The output dimension of this layer
#'      according to the format (`out_channels`, `out_length`) for 1D or
#'      (`out_channels`, `out_height`, `out_width`) for 2D.
#'      * **`$stride`** (optional): The stride of the convolution (single
#'      integer for 1D and tuple of two integers for 2D). If this value is not
#'      specified, the default values (1D: `1` and 2D: `c(1,1)`) are used.
#'      * **`$padding`** (optional): Zero-padding added to the sides of the
#'      input before convolution. For 1D-convolution a tuple of the form
#'      (`pad_left`, `pad_right`) and for 2D-convolution
#'      (`pad_left`, `pad_right`, `pad_top`, `pad_bottom`) is required. If this
#'      value is not specified, the default values (1D: `c(0,0)` and 2D:
#'      `c(0,0,0,0)`) are used.
#'      * **`$dilation`** (optional): Spacing between kernel elements (single
#'      integer for 1D and tuple of two integers for 2D). If this value is
#'      not specified, the default values (1D: `1` and 2D: `c(1,1)`) are used.
#'
#'   * **Pooling Layers:**
#'     * **`$type`**: `'MaxPooling1D'`, `'MaxPooling2D'`, `'AveragePooling1D'`
#'     or `'AveragePooling2D'`
#'     * **`$kernel_size`**: The size of the pooling window as an integer
#'     value for 1D-pooling and an tuple of two integers for 2D-pooling.
#'     * **`$strides`** (optional): The stride of the pooling window (single
#'      integer for 1D and tuple of two integers for 2D). If this value is not
#'      specified (`NULL`), the value of `kernel_size` will be used.
#'     * **`dim_in`** (optional): The input dimension of this layer. This
#'      value is not necessary, but helpful to check the correctness of the
#'      converted model.
#'     * **`dim_out`** (optional): The output dimension of this layer. This
#'      value is not necessary, but helpful to check the correctness of the
#'      converted model.
#'
#'   * **Flatten Layer:**
#'      * **`$type`**: `'Flatten'`
#'      * **`$dim_in`** (optional): The input dimension of this layer
#'      without the batch dimension.
#'      * **`$dim_out`** (optional): The output dimension of this layer
#'      without the batch dimension.
#'
#' **Note:** This package works internally only with the data format 'channels
#' first', i.e. all input dimensions and weight matrices must be adapted
#' accordingly.
#'
#'
#' @field model The converted neural network based on the torch module
#' [ConvertedModel].
#' @field model_dict The model stored in a named list (see details for more
#' information). By default, the entry `model_dict$layers` is deleted
#' because it may require a lot of memory for large networks. However, with
#' the argument `save_model_as_list` this can be saved anyway.
#'
#'
#' @examplesIf torch::torch_is_installed()
#' #----------------------- Example 1: Torch ----------------------------------
#' library(torch)
#'
#' model <- nn_sequential(
#'   nn_linear(5, 10),
#'   nn_relu(),
#'   nn_linear(10, 2, bias = FALSE),
#'   nn_softmax(dim = 2)
#' )
#' data <- torch_randn(25, 5)
#'
#' # Convert the model (for torch models is 'input_dim' required!)
#' converter <- Converter$new(model, input_dim = c(5))
#'
#' # Get the converted model
#' converted_model <- converter$model
#'
#' # Test it with the original model
#' mean(abs(converted_model(data) - model(data)))
#'
#'
#' #----------------------- Example 2: Neuralnet ------------------------------
#' library(neuralnet)
#' data(iris)
#'
#' # Train a neural network
#' nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width,
#'   iris,
#'   linear.output = FALSE,
#'   hidden = c(3, 2), act.fct = "tanh", rep = 1
#' )
#'
#' # Convert the model
#' converter <- Converter$new(nn)
#'
#' # Print all the layers
#' converter$model$modules_list
#'
#' #----------------------- Example 3: Keras ----------------------------------
#' library(keras)
#'
#' if (is_keras_available()) {
#'   # Define a keras model
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_conv_2d(
#'       input_shape = c(32, 32, 3), kernel_size = 8, filters = 8,
#'       activation = "relu", padding = "same"
#'     ) %>%
#'     layer_conv_2d(
#'       kernel_size = 8, filters = 4,
#'       activation = "tanh", padding = "same"
#'     ) %>%
#'     layer_conv_2d(
#'       kernel_size = 4, filters = 2,
#'       activation = "relu", padding = "same"
#'     ) %>%
#'     layer_flatten() %>%
#'     layer_dense(units = 64, activation = "relu") %>%
#'     layer_dense(units = 1, activation = "sigmoid")
#'
#'   # Convert this model
#'   converter <- Converter$new(model)
#'
#'   # Print the converted model as a named list
#'   str(converter$model_dict)
#' }
#'
#' #----------------------- Example 4: List  ----------------------------------
#'
#' # Define a model
#'
#' model <- NULL
#' model$input_dim <- 5
#' model$input_names <- list(c("Feat1", "Feat2", "Feat3", "Feat4", "Feat5"))
#' model$output_dim <- 2
#' model$output_names <- list(c("Cat", "no-Cat"))
#' model$layers$Layer_1 <-
#'   list(
#'     type = "Dense",
#'     weight = matrix(rnorm(5 * 20), 20, 5),
#'     bias = rnorm(20),
#'     activation_name = "tanh",
#'     dim_in = 5,
#'     dim_out = 20
#'   )
#' model$layers$Layer_2 <-
#'   list(
#'     type = "Dense",
#'     weight = matrix(rnorm(20 * 2), 2, 20),
#'     bias = rnorm(2),
#'     activation_name = "softmax"#,
#'     #dim_in = 20, # These values are optional, but
#'     #dim_out = 2  # useful for internal checks
#'   )
#'
#' # Convert the model
#' converter <- Converter$new(model)
#'
#' # Get the model as a torch::nn_module
#' torch_model <- converter$model
#'
#' # You can use it as a normal torch model
#' x <- torch::torch_randn(3, 5)
#' torch_model(x)
#'
#' @references
#' * J. D. Olden et al. (2004) \emph{An accurate comparison of methods for
#'  quantifying variable importance in artificial neural networks using
#'  simulated data.} Ecological Modelling 178, p. 389–397
#' * S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#'  classifier decisions by layer-wise relevance propagation.} PLoS ONE 10,
#'  p. 1-46
#' * A. Shrikumar et al. (2017) \emph{Learning important features through
#' propagating activation differences.}  ICML 2017, p. 4844-4866
#' * D. Smilkov et al. (2017) \emph{SmoothGrad: removing noise by adding noise.}
#' CoRR, abs/1706.03825
#'
#' @export
#'
Converter <- R6Class("Converter",
  public = list(
    model = NULL,
    model_dict = NULL,

    ### -----------------------------Initialize--------------------------------
    #' @description
    #' Create a new Converter for a given neural network.
    #'
    #' @param model A trained neural network for classification or regression
    #' tasks to be interpreted. Only models from the following types or
    #' packages are allowed: \code{\link[torch]{nn_sequential}},
    #' \code{\link[keras]{keras_model}},
    #' \code{\link[keras]{keras_model_sequential}},
    #' \code{\link[neuralnet]{neuralnet}} or a named list (see details).
    #' @param input_dim An integer vector with the model input dimension
    #' excluding the batch dimension, e.g. for a dense layer with `5` input
    #' features use `c(5)` or for a 1D convolutional layer with signal
    #' length `50` and `4` channels use `c(4, 50)`. \cr
    #' **Note:** This argument is only necessary for `torch::nn_sequential`,
    #' for all others it is automatically extracted from the passed model.
    #' In addition, the input dimension `input_dim` has to be in the format
    #' channels first.
    #' @param input_names (Optional) A list with the names for each input
    #' dimension, e.g. for a dense layer with `3` input features use
    #' `list(c("X1", "X2", "X3"))` or for a 1D convolutional layer with
    #' signal length `5` and `2` channels use
    #' `list(c("C1", "C2"), c("L1","L2","L3","L4","L5"))`.\cr
    #' **Note:** This argument is optional and otherwise the names are
    #' generated automatically. But if this argument is set, all found
    #' input names in the passed model will be disregarded.
    #' @param output_names (Optional) A list with the names for the output,
    #' e.g. for a model with `3` outputs use `list(c("Y1", "Y2", "Y3"))`.\cr
    #' **Note:** This argument is optional and otherwise the names are
    #' generated automatically. But if this argument is set, all found
    #' output names in the passed model will be disregarded.
    #' @param save_model_as_list This logical value specifies whether the
    #' passed model should be stored as a list (as it is described in the
    #' details also as an alternative input for a model). This list can take
    #' a lot of memory for large networks, so by default the model is not
    #' stored as a list (`FALSE`).
    #' @param dtype The data type for the calculations. Use
    #' either `'float'` for [torch::torch_float] or `'double'` for
    #' [torch::torch_double].
    #'
    #' @return A new instance of the R6 class \code{'Converter'}.
    #'

    initialize = function(model, input_dim = NULL, input_names = NULL,
                          output_names = NULL, dtype = "float",
                          save_model_as_list = FALSE) {
      assertChoice(dtype, c("float", "double"))
      assertLogical(save_model_as_list)

      # Analyze the passed model and store its internal structure in a list of
      # layers
      if (inherits(model, "nn")) {
        model_dict <- convert_neuralnet_model(model)
      } else if (inherits(model, c(
        "keras.engine.sequential.Sequential",
        "keras.engine.functional.Functional",
        "keras.engine.training.Model"
      ))) {
        model_dict <- convert_keras_model(model)
      } else if (is.list(model)) {
        model_dict <- model
      } else if (inherits(model, "nn_module") && is_nn_module(model)) {
        if (inherits(model, "nn_sequential")) {
          if (!testNumeric(input_dim, lower = 1)) {
            stop("For a 'torch' model you have to specify the argument 'input_dim'!")
          }
          model_dict <- convert_torch_sequential(model)
          model_dict$input_dim <- input_dim
        } else {
          stop("At the moment only sequential models are allowed!")
        }
      } else {
        stop(sprintf(
          "Unknown model of class \"%s\".",
          paste0(class(model), collapse = "\", \"")
        ))
      }

      if (!is.null(input_names)) {
        model_dict$input_names <- input_names
      }
      if (!is.null(output_names)) {
        model_dict$output_names <- output_names
      }

      private$create_model_from_dict(model_dict, dtype, save_model_as_list)
    }
  ),
  private = list(
    create_model_from_dict = function(model_dict, dtype = "float",
                                      save_model_as_list = FALSE) {
      modules_list <- NULL
      assertIntegerish(model_dict$input_dim, min.len = 1, max.len = 3)
      input <- torch_randn(c(2, model_dict$input_dim))

      assertChoice("layers", names(model_dict))
      for (i in seq_along(model_dict$layers)) {
        assertString(model_dict$layers[[i]]$type)
        type <- model_dict$layers[[i]]$type
        dim_in <- model_dict$layers[[i]]$dim_in
        dim_out <- model_dict$layers[[i]]$dim_out

        #
        # ---------------------- Flatten Layer -------------------------------
        #
        if (type == "Flatten") {
          # Create Layer
          layer <- flatten_layer(dim_in, dim_out)

          # Test Layer and register dim_in and dim_out if needed
          # Input dimension
          calculated_dim_in <- dim(input)[-1]
          if (!is.null(dim_in)) {
            assertSetEqual(dim_in, calculated_dim_in,
              ordered = TRUE,
              .var.name = paste0("model_dict$layers[[",i, "]]$dim_in"))
          } else {
            layer$input_dim <- calculated_dim_in
            model_dict$layers[[i]]$dim_in <- calculated_dim_in
          }
          # Layer works properly
          tryCatch(input <- layer(input, TRUE, FALSE, FALSE))
          # Output dimension
          calculated_dim_out <- dim(input)[-1]
          if (!is.null(dim_out)) {
            assertSetEqual(dim_out, calculated_dim_out,
                        ordered = TRUE,
                        .var.name = paste0("model_dict$layers[[",i, "]]$dim_out"))
          } else {
            layer$output_dim <- calculated_dim_out
            model_dict$layers[[i]]$dim_out <- calculated_dim_out
          }

          modules_list[[paste0("Flatten_", i)]] <- layer
        }
        #
        # ------------------------ Dense Layer -------------------------------
        #
        else if (type == "Dense") {
          # Check for required keys
          assertSubset(
            c("weight", "bias", "activation_name"),
            names(model_dict$layers[[i]])
          )
          assertArray(model_dict$layers[[i]]$weight, mode = "numeric", d = 2)
          assertVector(model_dict$layers[[i]]$bias, len = dim_out)
          assertString(model_dict$layers[[i]]$activation_name)

          weight <- model_dict$layers[[i]]$weight
          bias <- model_dict$layers[[i]]$bias
          activation_name <- model_dict$layers[[i]]$activation_name

          # Create the dense layer
          layer <- dense_layer(weight,
                               bias,
                               activation_name,
                               dim_in,
                               dim_out,
                               dtype = dtype)

          # Test Layer and register dim_in and dim_out if needed
          # Input dimension
          calculated_dim_in <- dim(input)[-1]
          if (!is.null(dim_in)) {
            assertSetEqual(dim_in, calculated_dim_in,
                        ordered = TRUE,
                        .var.name = paste0("model_dict$layers[[",i, "]]$dim_in"))
          } else {
            layer$input_dim <- calculated_dim_in
            model_dict$layers[[i]]$dim_in <- calculated_dim_in
          }

          # Layer works properly
          tryCatch(
            input <- layer(input, FALSE, FALSE, FALSE),
            error =
              function(e) {
                e$message <-
                  paste0("Could not create dense layer from list entry ",
                      "'model_dict$layers[[",i, "]]'. Maybe you have a wrong ",
                      "dimension order of your weight matrix. Remember: The ",
                      "weight matrix for a dense layer has to be stored as ",
                      "an array with shape (dim_out, dim_in)!\n\n",
                      "Original message:\n", e)
                stop(e)
              }
          )

          # Output dimension
          calculated_dim_out <- dim(input)[-1]
          if (!is.null(dim_out)) {
            assertSetEqual(dim_out, calculated_dim_out,
                        ordered = TRUE,
                        .var.name = paste0("model_dict$layers[[",i, "]]$dim_out"))
          } else {
            layer$output_dim <- calculated_dim_out
            model_dict$layers[[i]]$dim_out <- calculated_dim_out
          }

          modules_list[[paste0("Dense_", i)]] <- layer

        }
        #
        # ------------------------ Conv1D Layer -------------------------------
        #
        else if (type == "Conv1D") {
          # Check for required keys
          assertSubset(
            c("weight", "bias", "activation_name"),
            names(model_dict$layers[[i]])
          )
          assertArray(model_dict$layers[[i]]$weight, mode = "numeric", d = 3)
          assertVector(model_dict$layers[[i]]$bias, len = dim_out[1])
          assertString(model_dict$layers[[i]]$activation_name)
          assertInt(model_dict$layers[[i]]$stride, null.ok = TRUE)
          assertInt(model_dict$layers[[i]]$dilation, null.ok = TRUE)
          assertNumeric(model_dict$layers[[i]]$padding,
                        null.ok = TRUE, lower = 0
          )

          weight <- model_dict$layers[[i]]$weight
          bias <- model_dict$layers[[i]]$bias
          activation_name <- model_dict$layers[[i]]$activation_name
          stride <- model_dict$layers[[i]]$stride
          dilation <- model_dict$layers[[i]]$dilation
          padding <- model_dict$layers[[i]]$padding

          if (is.null(stride)) stride <- 1
          if (is.null(dilation)) dilation <- 1
          if (is.null(padding)) padding <- c(0, 0)

          if (length(padding) == 1) {
            padding <- rep(padding, 2)
          } else if (length(padding) != 2) {
            stop(paste0(
              "Expected a padding vector in 'model_dict$layers[[",i,"]] ",
              "of length:\n", "   - 1: same padding for each side\n",
              "   - 2: first value: padding for left side; second value: padding for right side\n",
              "But your length: ", length(padding))
            )
          }

          model_dict$layers[[i]]$stride <- stride
          model_dict$layers[[i]]$stride <- padding
          model_dict$layers[[i]]$stride <- dilation


          layer <- conv1d_layer(weight, bias, dim_in, dim_out, stride,
                                padding, dilation, activation_name,
                                dtype = dtype
          )

          # Test Layer and register dim_in and dim_out if needed
          # Input dimension
          calculated_dim_in <- dim(input)[-1]
          if (!is.null(dim_in)) {
            assertSetEqual(dim_in, calculated_dim_in,
                        ordered = TRUE,
                        .var.name = paste0("model_dict$layers[[",i, "]]$dim_in"))
          } else {
            layer$input_dim <- calculated_dim_in
            model_dict$layers[[i]]$dim_in <- calculated_dim_in
          }

          # Layer works properly
          tryCatch(
            input <- layer(input, FALSE, FALSE, FALSE),
            error =
              function(e) {
                e$message <-
                  paste0("Could not create 1d convolutional layer from list entry ",
                         "'model_dict$layers[[",i, "]]'. Maybe you have a wrong ",
                         "dimension order of your weight matrix. Remember: The ",
                         "weight matrix for this layer has to be stored as ",
                         "an array with shape (out_channels, in_channels, ",
                         "kernel_length)!\n\n", "Original message:\n", e$message)
                stop(e)
              }
          )

          # Output dimension
          calculated_dim_out <- dim(input)[-1]
          if (!is.null(dim_out)) {
            assertSetEqual(dim_out, calculated_dim_out,
                        ordered = TRUE,
                        .var.name = paste0("model_dict$layers[[",i, "]]$dim_out"))
          } else {
            layer$output_dim <- calculated_dim_out
            model_dict$layers[[i]]$dim_out <- calculated_dim_out
          }

          modules_list[[paste0("Conv1D_", i)]] <- layer

        }
        #
        # ------------------------ Conv2D Layer -------------------------------
        #
        else if (type == "Conv2D") {
          assertSubset(
            c("weight", "bias", "activation_name"),
            names(model_dict$layers[[i]])
          )
          assertArray(model_dict$layers[[i]]$weight, mode = "numeric", d = 4)
          assertVector(model_dict$layers[[i]]$bias, len = dim_out[1])
          assertString(model_dict$layers[[i]]$activation_name)
          assertNumeric(model_dict$layers[[i]]$stride,
                        null.ok = TRUE, lower = 1
          )
          assertNumeric(model_dict$layers[[i]]$dilation,
                        null.ok = TRUE, lower = 1
          )
          assertNumeric(model_dict$layers[[i]]$padding,
                        null.ok = TRUE, lower = 0
          )

          weight <- model_dict$layers[[i]]$weight
          bias <- model_dict$layers[[i]]$bias
          activation_name <- model_dict$layers[[i]]$activation_name
          stride <- model_dict$layers[[i]]$stride
          dilation <- model_dict$layers[[i]]$dilation
          padding <- model_dict$layers[[i]]$padding

          if (is.null(stride)) stride <- c(1, 1)
          if (is.null(dilation)) dilation <- c(1, 1)
          if (is.null(padding)) padding <- c(0, 0, 0, 0)

          if (length(padding) == 1) {
            padding <- rep(padding, 4)
          } else if (length(padding) == 2) {
            padding <- rep(padding, each = 2)
          } else if (length(padding) != 4) {
            stop(paste0(
              "Expected a padding vector in 'model_dict$layers[[",i,"]] ",
              "of length:\n", "   - 1: same padding on each side\n",
              "   - 2: first value: pad_left and pad_right; second value: pad_top and pad_bottom\n",
              "   - 4: (pad_left, pad_right, pad_top, pad_bottom)\n",
              "But your length: ", length(padding))
            )
          }
          if (length(stride) == 1) {
            stride <- rep(stride, 2)
          } else if (length(stride) != 2) {
            stop(paste0(
              "Expected a stride vector in 'model_dict$layers[[",i,"]] ",
              "of length:\n", "   - 1: same stride for image heigth and width\n",
              "   - 2: first value: strides for height; second value: strides for width\n",
              "But your length: ", length(stride))
            )
          }
          if (length(dilation) == 1) {
            dilation <- rep(dilation, 2)
          } else if (length(dilation) != 2) {
            stop(paste0(
              "Expected a dilation vector in 'model_dict$layers[[",i,"]] ",
              "of length:\n", "   - 1: same dilation for image heigth and width\n",
              "   - 2: first value: dilation for height; second value: dilation for width\n",
              "But your length: ", length(dilation))
            )
          }

          model_dict$layers[[i]]$stride <- stride
          model_dict$layers[[i]]$stride <- padding
          model_dict$layers[[i]]$stride <- dilation

          layer <- conv2d_layer(weight, bias, dim_in, dim_out, stride,
                                padding, dilation, activation_name,
                                dtype = dtype
          )

          # Test Layer and register dim_in and dim_out if needed
          # Input dimension
          calculated_dim_in <- dim(input)[-1]
          if (!is.null(dim_in)) {
            assertSetEqual(dim_in, calculated_dim_in,
                        ordered = TRUE,
                        .var.name = paste0("model_dict$layers[[",i, "]]$dim_in"))
          } else {
            layer$input_dim <- calculated_dim_in
            model_dict$layers[[i]]$dim_in <- calculated_dim_in
          }

          # Layer works properly
          tryCatch(
            input <- layer(input, FALSE, FALSE, FALSE),
            error =
              function(e) {
                e$message <-
                  paste0("Could not create 2d convolutional layer from list entry ",
                         "'model_dict$layers[[",i, "]]'. Maybe you have a wrong ",
                         "dimension order of your weight matrix. Remember: The ",
                         "weight matrix for this layer has to be stored as ",
                         "an array with shape (out_channels, in_channels, ",
                         "kernel_height, kernel_width)!\n\n",
                         "Original message:\n", e$message)
                stop(e)
              }
          )

          # Output dimension
          calculated_dim_out <- dim(input)[-1]
          if (!is.null(dim_out)) {
            assertSetEqual(dim_out, calculated_dim_out,
                        ordered = TRUE,
                        .var.name = paste0("model_dict$layers[[",i, "]]$dim_out"))
          } else {
            layer$output_dim <- calculated_dim_out
            model_dict$layers[[i]]$dim_out <- calculated_dim_out
          }

          modules_list[[paste0("Conv2D_", i)]] <- layer
        }
        #
        # ------------------------ Pooling Layers -------------------------------
        #
        else if (type %in% c("MaxPooling1D", "MaxPooling2D",
                             "AveragePooling1D", "AveragePooling2D")) {
          # Check for required keys
          assertSubset(c("kernel_size"), names(model_dict$layers[[i]]))
          assertVector(model_dict$layers[[i]]$kernel_size, min.len = 1, max.len = 2)

          kernel_size <- model_dict$layers[[i]]$kernel_size
          strides <- model_dict$layers[[i]]$strides

          if (is.null(strides)) strides <- kernel_size

          if (type == "MaxPooling1D") {
            assertSetEqual(length(kernel_size), 1)
            assertSetEqual(length(strides), 1)

            layer <- max_pool1d_layer(kernel_size, dim_in, dim_out, strides)
          } else if (type == "AveragePooling1D") {
            assertSetEqual(length(kernel_size), 1)
            assertSetEqual(length(strides), 1)

            layer <- avg_pool1d_layer(kernel_size, dim_in, dim_out, strides)
          } else if (type == "MaxPooling2D") {
            assertSetEqual(length(kernel_size), 2)
            assertSetEqual(length(strides), 2)

            layer <- max_pool2d_layer(kernel_size, dim_in, dim_out, strides)
          } else if (type == "AveragePooling2D") {
            assertSetEqual(length(kernel_size), 2)
            assertSetEqual(length(strides), 2)

            layer <- avg_pool2d_layer(kernel_size, dim_in, dim_out, strides)
          }

          # Test Layer and register dim_in and dim_out if needed
          # Input dimension
          calculated_dim_in <- dim(input)[-1]
          if (!is.null(dim_in)) {
            assertSetEqual(dim_in, calculated_dim_in,
                           ordered = TRUE,
                           .var.name = paste0("model_dict$layers[[",i, "]]$dim_in"))
          } else {
            layer$input_dim <- calculated_dim_in
            model_dict$layers[[i]]$dim_in <- calculated_dim_in
          }

          # Layer works properly
          tryCatch(input <- layer(input, FALSE, FALSE, FALSE))

          # Output dimension
          calculated_dim_out <- dim(input)[-1]
          if (!is.null(dim_out)) {
            assertSetEqual(dim_out, calculated_dim_out,
                           ordered = TRUE,
                           .var.name = paste0("model_dict$layers[[",i, "]]$dim_out"))
          } else {
            layer$output_dim <- calculated_dim_out
            model_dict$layers[[i]]$dim_out <- calculated_dim_out
          }

          modules_list[[paste0(type, "_", i)]] <- layer

        } else {
          stop(sprintf("Unknown layer type '%s' in model dictionary. Only the
                       types 'Dense', 'Conv1D', 'Conv2D', 'Flatten', 'AveragePooling1D',
                       'AveragePooling2D', 'MaxPooling1D' and 'MaxPooling2D'
                       are allowed!", type))
        }
      }

      # Check output dimension
      if (is.null(model_dict$output_dim)) {
        model_dict$output_dim <- dim(input)[-1]
      } else {
        assertSetEqual(model_dict$output_dim, dim(input)[-1],
                    ordered = TRUE,
                    .var.name = paste0("model_dict$output_dim"))
      }

      # Check for classification output
      if (length(model_dict$output_dim) != 1) {
        stop(paste0(
          "This package only allows models with classification or regression ",
          "output, i.e. the model output dimension has to be one. ",
          "But your model has an output dimension of '",
          length(model_dict$output_dim), "'!"))
      }

      # Check input names
      if (is.null(model_dict$input_names)) {
        model_dict$input_names <- private$get_input_names(model_dict)
      } else {
        input_names <- model_dict$input_names
        assertList(input_names)
        assertSetEqual(sapply(input_names, length), model_dict$input_dim,
                       ordered = TRUE)
      }

      # Check output names
      if (is.null(model_dict$output_names)) {
        model_dict$output_names <- private$get_output_names(model_dict)
      } else {
        output_names <- model_dict$output_names
        assertList(output_names)
        assertSetEqual(sapply(output_names, length), model_dict$output_dim,
                       ordered = TRUE)
      }

      # Save model_dict and create torch model
      if (!save_model_as_list) {
        model_dict$layers <- NULL
      }
      self$model_dict <- model_dict
      self$model <- ConvertedModel(modules_list, dtype = dtype)
      self$model$reset()

    },
    get_input_names = function(model_dict) {
      input_dim <- model_dict$input_dim
      if (length(input_dim) == 1) {
        short_names <- c("X")
      } else if (length(input_dim) == 2) {
        short_names <- c("C", "L")
      } else if (length(input_dim) == 3) {
        short_names <- c("C", "H", "W")
      } else {
        stop(sprintf("Too many input dimensions. This package only allows
                     model inputs with '1', '2' or '3' dimensions and not
                     '%s'", length(imput_dim)))
      }
      input_names <-
        mapply(function(x, y) paste0(rep(y, times = x), 1:x),
          input_dim,
          short_names,
          SIMPLIFY = FALSE
        )
      input_names
    },
    get_output_names = function(model_dict) {
      output_names <-
          lapply(
            model_dict$output_dim,
            function(x) paste0(rep("Y", times = x), 1:x)
          )
      output_names
    }
  )
)



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
  dtype = NULL,
  initialize = function(modules_list, dtype = "float") {
    self$modules_list <- modules_list
    self$dtype <- dtype
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
    if (channels_first == FALSE) {
      x <- torch_movedim(x, -1, 2)
    }

    num_modules <- length(self$modules_list)
    i <- 1

    for (module in self$modules_list) {
      if (save_last_layer & i == num_modules) {
        save_preactivation <- TRUE
        save_output <- TRUE
      }
      if ("Flatten_Layer" %in% module$.classes) {
        x <- module(x, channels_first, save_input, save_output)
      } else {
        x <- module(x, save_input, save_preactivation, save_output)
      }

      i <- i + 1
    }
    x
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
                        save_last_preactivation = FALSE) {
    if (channels_first == FALSE) {
      x_ref <- torch_movedim(x_ref, -1, 2)
    }

    num_modules <- length(self$modules_list)
    i <- 1

    for (module in self$modules_list) {
      if (save_last_preactivation & i == num_modules) {
        save_preactivation <- TRUE
      }
      if ("Flatten_Layer" %in% module$.classes) {
        x_ref <- module(x_ref, channels_first, save_input, save_output)
      } else {
        x_ref <- module$update_ref(x_ref, save_input, save_preactivation,
                                   save_output)
      }
      i <- i + 1
    }
    x_ref
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
      if (!("Flatten_Layer" %in% module$.classes)) {
        module$set_dtype(dtype)
      }
    }
    self$dtype <- dtype
  },

  reset = function() {
    for (module in self$modules_list) {
      module$reset()
    }
  }
)
