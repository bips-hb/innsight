

DeepLift <- R6::R6Class(
  classname = "DeepLift",

  public = list(

    rule_name = NULL,
    x_ref = NULL,


    contribution = NULL,

    initialize = function(analyzer, data,
                          rule_name = "rescale",
                          x_ref = NULL,
                          dtype = "float") {

      checkmate::assertClass(analyzer, "Analyzer")

      #
      # toDo
      #
      # check and store arguments and transform to torch Tensors. Use the private
      # method 'run' for calculating the contributions and save the result in 'contribution'
      #
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
    },

    change_rule = function(rule_name, rule_param) {
      #
      # toDo
      #
      # change the rule and re-run the method
      #
    }
  ),

  private = list(

    run = function() {
      #
      # toDo
      #
      # Apply method DeepLift
      #
    }
  )
)
