LRP <- R6::R6Class(
  classname = "LRP",
  
  public = list(
    
    rule_name = NULL,
    rule_param = NULL,
    data = NULL,
    analyzer = NULL,
    
    relevances = NULL,
    dtype = "float",
    
    initialize = function(analyzer, data,
                          rule_name = "simple",
                          rule_param = NULL,
                          dtype = "float") {
      
      
      checkmate::assertClass(analyzer, "Analyzer")
      self$rule_name <- rule_name
      self$rule_param <- rule_param
      self$analyzer <- analyzer
      self$data <- data
      self$dtype <- dtype
      
      
      #Bit weird , should we first set self$dtype <- dtype?
      self$set_dtype(dtype)
      
      self$analyzer$forward(data,channels_first = FALSE)
      
      private$run()
      
      #
      # toDo
      #
      # check and store arguments and transform to torch Tensors. Use the private
      # method 'run' for calculating the relevances and save the result in 'relevances'
      #
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
    
    change_rule = function(rule_name, rule_param) {
      
      self$rule_name <- rule_name
      self$rule_param <- rule_param
      
      private$run()
      
      #
      # toDo
      #
      # change the rule and re-run the method
      #
    }
  ),
  
  private = list(
    
    #'@title Get relevances relative to model input
    #'@name run
    #'@description
    #'This function passes the relevance through each layer of the network and outputs the 
    #'relevance relative to the inputs
    #'@return
    #'Returns the input relevances, of dimension \emph{(batch_size,model_input,model_output)}
    
    run = function() {
      


      output_dim <- self$analyzer$output_dim
      batch_size <- dim(self$data)[1]
      
      
      rev_layers <- rev(self$analyzer$model$modules_list)
      last_layer <- rev_layers[[1]]
      
      out_last <- last_layer$output
      print("rel")
      print(last_layer$output)



        rel <- torch_diag_embed(last_layer$output)
        print(rel)

        
        # other layers
      for (layer in rev_layers) {
        if("Flatten_Layer" %in% layer$'.classes'){
          rel <- layer$reshape_to_input(rel)
          
        }else{
          

          
  
        rel <- layer$get_input_relevances(rel,self$rule_name,self$rule_param)
        l <- length(layer$input_dim)
        s <- rel$sum(2:(1+l))
        #print(dim(s))

       # print(dim(out_last))
        print(s-out_last)

        }

      }

      rel
      sum_rel <- rel$sum(c(2,3))

     print(sum_rel-out_last)


      #
      # toDo
      #
      # Apply method LRP
      #
      #}
    }
  )
)
