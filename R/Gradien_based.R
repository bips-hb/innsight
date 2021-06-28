
Gradient_Based <- R6::R6Class(
  classname = "Gradient_Based",
  public = list(

    data = NULL,
    model = NULL,
    times_input = NULL,

    result = NULL,

    initialize = function() {

    },

    set_dtype = function(dtype) {
      #
      # toDo
      #
      # Set dtype of data and all the model params
      #
    },

    change_data = function(data) {
      #
      # toDo
      #
      # change the data and re-run the method
      #
    }


  ),

  private = list(
    #
    # data    : torch Tensor [*, dim_in]
    #
    # output  : Gradients [*, dim_in]
    calculate_gradients = function(input) {
      grad <- NULL

      #
      # toDo
      #

      grad
    },

    run = function() {

    }
  )
)

Gradient <- R6::R6Class(
  classname = "Gradient",
  inherit = "Gradient_Based",
  public = list(

    initialize = function(analyzer, data,
                          dtype = "float",
                          times_input = TRUE,
                          ignore_last_act = TRUE) {
      #
      # toDo
      #
      # store and check arguments, transform to torch Tensor and calculate
      # normal gradients (use private method 'run') of the given data. Save the
      # result in the attribute 'result'
      #
    }
  ),

  private = list(
    run = function() {
      #
      # toDo
      #
      # Main method. Use the parent-method 'calculate_gradient'. In this
      # case, this is more or less already the solution
      #

    }
  )
)

SmoothGrad <- R6::R6Class(
  classname = "SmoothGrad",
  inherit = "Gradient_Based",
  public = list(

    initialize = function(analyzer, data,
                          n = 50,
                          noise_level = 0.1,
                          dtype = "float",
                          times_input = TRUE,
                          ignore_last_act = TRUE) {
      #
      # toDo
      #
      # store and check relevant arguments, transform to torch Tensor and calculate
      # smoothed gradients (use private method 'run') of the given data. Save the
      # result in the attribute 'result'
      #
    }
  ),

  private = list(
    run = function() {
      #
      # toDo
      #
      # Main method. Apply SmoothGrad.
      #

    }
  )
)

