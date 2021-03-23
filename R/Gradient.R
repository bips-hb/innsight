

#' @title Gradient method
#' @name func_gradient
#'
#' @description
#' This method computes the gradients of the outputs with respect to the input
#' variables, i.e. for all input variable \eqn{i} and output class \eqn{j}
#' \deqn{\frac{\partial f(x)_j}{\partial x_i}.}
#'
#' @param layers List of layers of type \code{\link{Dense_Layer}}.
#' @param out_class If the given model is a classification model, this
#' parameter can be used to determine which class the gradients should be
#' calculated for. Use the default value \code{NULL} to return the gradients
#' for all classes.
#'
#' @return If \code{out_class} is \code{NULL} it returns a matrix of shape \emph{(in, out)},
#' which contains the gradients for each input variable to the
#' output predictions. Otherwise returns a vector of the gradient
#' for each input variable for the given output class.
#'
#' @export
#'

func_gradient <- function(layers, out_class = NULL) {
    last_layer <- layers[[length(layers)]]
    act_dev <- get_deveritive_activation(last_layer$activation_name)

    gradient <- act_dev(last_layer$preactivation) %*% t(last_layer$weights)

    for (layer in rev(layers)[-1]) {
        act_dev <- get_deveritive_activation(layer$activation_name)
        gradient <- gradient %*% act_dev(layer$preactivation) %*% t(layer$weights)
    }
    gradient <- t(gradient)
    rownames(gradient) <- paste0(rep("X", nrow(gradient)), 1:nrow(gradient))
    colnames(gradient) <- paste0(rep("Y", ncol(gradient)), 1:ncol(gradient))
    if (is.null(out_class)) {
        return(gradient)
    } else {
        if ( !(out_class %in% 1:ncol(gradient) ) ) {
            stop(sprintf("Parameter 'out_class' has to be an integer value between 1 and %s! Your value: %s",
                         ncol(gradient), out_class ))
        } else return(gradient[, out_class])
    }
}
