

DeepLift <- R6::R6Class(
  classname = "DeepLift",

  public = list(

    rule_name = NULL,
    x_ref = NULL,
    dtype = "float",
    analyzer = NULL,
    data = NULL,


    contribution = NULL,

    initialize = function(analyzer, data,
                          rule_name = "rescale",
                          x_ref = NULL,
                          dtype = "float") {

      checkmate::assertClass(analyzer, "Analyzer")
      
      self$analyzer <- analyzer
      self$rule_name <- rule_name
      self$data <- data
      
      # Get the reference input
      # NULL: create an input vector of zeros
      
      
      self$update_ref(x_ref)
      self$x_ref <- x_ref
      self$dtype <- dtype

      
      
      self$set_dtype(dtype)

      
     private$run()
      

      #
      # toDo
      #
      # check and store arguments and transform to torch Tensors. Use the private
      # method 'run' for calculating the contributions and save the result in 'contribution'
      #
    },
    
    
    update_ref = function(x_ref, channels_first = TRUE) {
      
      if(!inherits(x_ref,"torch_tensor")){
        if(is.null(x_ref)){
          x_ref <- torch::torch_unsqueeze(torch::torch_zeros(self$analyzer$input_dim),1)

        }else{
          x_ref <- torch::torch_tensor(as.array(x_ref), dtype = torch::torch_float())
        }
      }
      #x_ref <- torch::torch_tensor(as.array(x_ref), dtype = torch::torch_float())
      if (channels_first == FALSE) {
        x_ref <- torch::torch_movedim(x_ref, -1,2)
      }
      out_ref <- self$analyzer$model$update_ref(x_ref, channels_first)
      self$analyzer$input_last_ref <- x_ref
      
      torch::as_array(out_ref)
    },

      
      set_dtype = function(dtype) {
        
        
        if (dtype == "float") {
          
          for(i in 1:length(self$analyzer$model$modules_list)){


            
            layer <- self$analyzer$model$modules_list[[i]]

            if("Flatten_Layer" %in% layer$'.classes' == FALSE){
              
              self$analyzer$model$modules_list[[i]]$W <- layer$W$to(torch::torch_float())
              self$analyzer$model$modules_list[[i]]$b <- layer$b$to(torch::torch_float())
              
            }
          }
        }
        else if (dtype == "double") {
          for(i in 1:length(self$analyzer$model$modules_list)){
            
            layer <- self$analyzer$model$modules_list[[i]]
            if("Flatten_Layer" %in% layer$'.classes' == FALSE){
              
              
              self$analyzer$model$modules_list[[i]]$W <- layer$W$to(torch::torch_double())
              self$analyzer$model$modules_list[[i]]$b <- layer$b$to(torch::torch_double())
              
            }
          }
        }
        else {
          stop(sprintf("Unknown argument for 'dtype' : %s . Use 'float' or 'double' instead"))
        }
        
        
        self$dtype <- dtype
        
        #
        # toDo
        #
        # Set dtype of data and all the model params
        #
      #
      # toDo
      #
      # Set dtype of data and all the model params
      #
    },

    change_data = function(data) {
      
      self$data <- data
      private$run()
      #
      # toDo
      #
      # change the data and re-run the method
      #
    },
    
    change_ref = function(x_ref,channels_first = TRUE){
      
      self$update_ref(x_ref,channels_first)
      self$x_ref <- x_ref
      private$run()
    },

    change_rule = function(rule_name, rule_param) {
      
      self$rule_name <- rule_name
      
      private$run()
      #
      # toDo
      #
      # change the rule and re-run the method
      #
    }
  ),

  private = list(

    run = function() {

      
      output_dim <- self$analyzer$output_dim
      batch_size <- dim(self$data)[1]
      
      
      rev_layers <- rev(self$analyzer$model$modules_list)
      last_layer <- rev_layers[[1]]

      
      mul <- torch::torch_diag_embed(last_layer$output)

      


      
      
      # other layers
      for (layer in rev_layers) {
        if("Flatten_Layer" %in% layer$'.classes'){
          mul <- layer$reshape_to_input(mul)
          
        }else{
          
          
          
          
          mul <- layer$get_input_multiplier(mul,self$rule_name)

          
        }
        
      }

      mul
      
      
      
      #
      # toDo
      #
      # Apply method DeepLift
      #
    }
  )
)
