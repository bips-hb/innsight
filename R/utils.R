#' @importFrom cli cli_abort cli_warn cli_inform

###############################################################################
#                 Stop, warning and message methods
###############################################################################

stopf <- function(..., use_paste = TRUE) {
  if (use_paste) {
    cli_abort(paste0(...))
  } else {
    cli_abort(...)
  }
}

warningf <- function(...) {
  cli_warn(paste0(...))
}

messagef <- function(...) {
  cli_inform(paste0(...))
}

cli_check <- function(check, varname) {
  fail <- all(!unlist(lapply(check, function(x) isTRUE(x) | x == "TRUE")))
  if (fail) {
    # remove curly braces
    check <- gsub("','", "', '", gsub("[}]", "]", gsub("[{]", "[", check)))

    if (length(check) == 1) {
      stopf("Assertion on {.arg ", varname, "} failed: ", check)
    } else {
      check <- paste0("{.arg ", varname, "}: ", check)
      names(check) <- rep("*", length(check))
      check <- c("Assertion failed. One of the following must apply:", check)
      stopf(check, use_paste = FALSE)
    }
  }
}
