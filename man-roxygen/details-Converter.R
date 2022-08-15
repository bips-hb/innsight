#' @details
#'
#' In order to better understand and analyze the prediction of a neural
#' network, the preactivation or other information of the individual layers,
#' which are not stored in an ordinary forward pass, are often required. For
#' this reason, a given neural network is converted into a torch-based neural
#' network, which provides all the necessary information for an interpretation.
#' The converted torch model is stored in the field `model` and is an instance
#' of \code{\link[innsight:ConvertedModel]{ConvertedModel}}.
#' However, before the torch model is created, all relevant details of the
#' passed model are extracted into a named list. This list can be saved
#' in complete form in the `model_as_list` field with the argument
#' `save_model_as_list`, but this may consume a lot of memory for large
#' networks and is not done by default. Also, this named list can again be
#' used as a passed model for the class `Converter`, which will be described
#' in more detail in the section 'Implemented Libraries'.
#'
#' ## Implemented Methods
#'
#' An object of the Converter class can be applied to the
#' following methods:
#'   * Layerwise Relevance Propagation ([LRP]), Bach et al. (2015)
#'   * Deep Learning Important Features ([DeepLift]), Shrikumar et al. (2017)
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
