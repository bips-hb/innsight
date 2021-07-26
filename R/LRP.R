#' Layer-wise relevance propagation
#'
#' Implementation of layer-wise backpropagation of relevances as an R6 class where the input relevances relative to the output of
#' the model are stored.
#' @section Attributes:
#' \describe{
#' 
#'   \item{`self$rule_name`}{ rule_name The name of the rule to be applied, currently supports \emph{"simple"}, \emph{"epsilon"} and \emph{"alpha_beta} }
#'   \item{`self$rule_param`}{The value of the rule parameter, only for epsilon and alpha_beta rules.}
#'   \item{`self$data`}{The inputs whose relevances are to be evaluated}
#'   \item{`self$analyzer`}{The R6 Analalyzer class instance that contains the model to be analyzed}
#'   \item{`self$data`}{The relevances of the output with respect to the input, of dimensions \emph{(batch_size,model_dim_in,model_dim_out)}}
#'   \item{`self$dtype`}{The data type to be used for the weight matrices and biases in the model}
#'}
#'
#'@export
#'
#'

# Layer-wise Relevance Propagation (LRP)
# "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation"
#       by S. Bach et al. (2015)
#

#' @title Layer-wise Relevance Propagation (LRP) method
#' @name LRP
#'
#' @description
#' This is an implementation of the \emph{Layer-wise Relevance Propagation (LRP)}
#' algorithm introduced by Bach et al. (2015). It's a local method for
#' interpreting a single element of the dataset and returns the relevance scores for
#' each input feature. The basic idea of this method is to decompose the
#' prediction score of the model with respect to the input features, i.e.
#' \deqn{f(x) = \sum_i R(x_i).}
#' Because of the bias vector, this decomposition is generally an approximation.
#' There exist several propagation rules to determine the relevance scores. In this
#' package are implemented: \code{\link{linear_simple_rule}},
#' \code{\link{linear_eps_rule}}, \code{\link{linear_ab_rule}}
#'
#' @param analyzer An instance of the R6 class \code{\link{Analyzer}}.
#' @param data The input to the model, of dimensions \emph{(batch_size,model_dim_in)}, where \emph{model_dim_in}
#' refers to the dimensions of an input to the model
#' @param rule_name The name of the rule, with which the relevance scores are
#' calculated. Implemented are \code{"simple"}, \code{"epsilon"}, \code{"alpha_beta"}
#' (default: \code{"simple"}).
#' @param rule_param The parameter of the selected rule. Note: Only the rules
#' \code{"epsilon"} and \code{"alpha_beta"} make use of the parameter. Use the default
#' value \code{NULL} for the default parameters ("epsilon" : \eqn{0.01}, "alpha_beta" : \eqn{0.5}).
#' @param dtype The datatype to be used for the weights and biases of the layers in the model's neural network, supported are \emph{"float"}
#' and \emph{"double"}
#' @references
#' S. Bach et al. (2015) \emph{On pixel-wise explanations for non-linear
#' classifier decisions by layer-wise relevance propagation.} PLoS ONE 10, p. 1-46
#'
#' @export


LRP <- R6::R6Class(
  classname = "LRP",
  
  public = list(
    #'@field relevances The input relevances, of dimension \emph{(batch_size,model_dim_in,model_dim_out)}
    
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
    },
    
    plot_relevances = function(i = NULL,j = NULL, rank = FALSE, scale = FALSE, ...){
      
      output_dim <- self$analyzer$output_dim
      batch_size <- dim(self$data)[1]
      rel <- torch_squeeze(self$relevances)
      rel_array <- as.array(rel)

      
      if (self$rule_name %in% c("epsilon", "alpha_beta")) {
        subtitle = sprintf("%s-Rule (%s)", rule_name, rule_param)
      } else {
        subtitle = sprintf("%s-Rule", self$rule_name)
      }
      
      if(!is.null(j)){
        aperm_array <- as.array(torch_tensor(aperm(rel_array,length(dim(rel_array)):1))[j,])
        rel_array <- aperm(aperm_array,length(dim(aperm_array)):1)

        rel <- torch_tensor(rel_array)
        output_dim <- 1
        names_out <- "Y"
      }
        
      print("before i")
      if(!is.null(i)){
        rel <- rel[i,]
        rel_array <- as.array(rel)

        batch_size <- 1
      }

      if("Dense_Layer" %in% self$analyzer$model$modules_list[[1]]$'.classes'){
        print("in dense")
        dim_in <- dim(self$data)[2]
        
        if(is.null(i)){names_in <- paste0("X",1:dim(rel_array)[[2]])}else{names_in <- paste0("X",1:length(rel_array))}
        if(is.null(j)){names_out <- paste0("Y",1:dim(rel_array)[[3]])}

        x <- rel_array
          
          features = rep(names_in, output_dim*batch_size)
          
          Class = rep(names_out, each = dim_in, times = batch_size)
          if (rank) {
            x[] <- apply(x, 3, function(z) apply(z,2, rank))
            y_min <- 1
            y_max <- dim(x)[1]
          } else if (scale) {
            y_min <- stats::quantile(x, 0.05)
            y_max <- stats::quantile(x, 0.95)
          } else {
            y_min <- min(x)
            y_max <- max(x)
          }
          Relevance = as.vector(x)
          Features <- factor(features, levels = names_in)
          ggplot2::ggplot(data.frame(Features, Class, Relevance),
                          mapping = ggplot2::aes(x = Features, y = Relevance, fill = Class), ...) +
            ggplot2::geom_boxplot(alpha = 0.6) +
            ggplot2::scale_fill_viridis_d() +
            ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
            ggplot2::ggtitle("Feature Importance with Layerwise Relevance Propagation", subtitle = subtitle)
        
      }else if("Conv2D_Layer" %in% self$analyzer$model$modules_list[[1]]$".classes"){
        print("in conv2d")
        if((!is.null(i) | batch_size == 1) && (!is.null(j) | output_dim == 1)){
          if(length(dim(rel_array)) == 3){
            
            rel <- torch_sum(rel,1)
            rel_array <- as.matrix(rel)
            print("before image")
            print(dim(as.data.frame(rel_array)))
            image(rel_array,col=grey(seq(0, 1, length = 256)))
            x <- 1:32
            y <- x
            #plot(rel_array)
            
          }
          
        }
      }
        


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



        rel <- torch_diag_embed(last_layer$output)

        
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
        #print(s-out_last)

        }

      }
        
        
      self$relevances <- rel
      rel

     # sum_rel <- rel$sum(c(2,3))

     #print(sum_rel-out_last)


      #
      # toDo
      #
      # Apply method LRP
      #
      #}
    }
  )
)
