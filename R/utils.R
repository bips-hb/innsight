

###############################################################################
#                 Stop, warning and message methods
###############################################################################

stopf <- function(..., call = NULL) {
  stop(simpleError(paste0(...), call = call))
}

warningf <- function(..., call = NULL) {
  warning(simpleWarning(paste0(...), call = call))
}

messagef <- function(...) {
  message(paste0(...))
}
