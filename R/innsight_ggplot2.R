#' @importFrom methods is new show
NULL

#' S4-Class for ggplot2-based plots
#'
#' The S4 class `innsight_ggplot2` visualizes the results of the methods
#' provided from the package `innsight` using [ggplot2]. In addition, it
#' allows easier analysis of the results and modification of the
#' visualization by basic generic functions. The individual slots are for
#' internal use only and should not be modified.
#'
#' @slot grobs The individual ggplot2 objects arranged as a matrix (see
#' details for more information)
#' @slot multiplot A logical value indicating whether there are multiple
#' input layers and therefore correspondingly individual ggplot2 objects
#' instead of one single object.
#' @slot output_strips A list containing the labels and themes of the strips
#' for the output nodes. This slot is only relevant if `multiplot` is `TRUE`.
#' @slot col_dims A list of the length of `output_strips` assigning to
#' each strip the column index of `grobs` of the associated strip.
#' @slot boxplot A logical value indicating whether the result of individual
#' data points or a boxplot over multiple instances is displayed.
#'
#' @details
#'
#' This S4 class is a simple extension of a [ggplot2] object that enables
#' a more detailed analysis of the results and a way to visualize the results
#' of models with multiple input layers (e.g., images and tabular data).
#' The distinction between one and multiple input layers decides the behavior
#' of this class, and this information is stored in the slot `multiplot`.
#'
#' ## One input layer (`multiplot = FALSE`)
#'
#' If the model passed to a method from the innsight package has only one
#' input layer, the S4 class `innsight_ggplot2` is just a wrapper of a
#' single ggplot2 object. This object is stored as a 1x1 matrix in
#' the slot `grobs` and the slots `output_strips` and `col_dims` contain
#' only empty lists because no second line of stripes describing the input
#' layer is needed.
#' Although it is an object of the class `innsight_ggplot2`,
#' the generic function [+.innsight_ggplot2] provides a ggplot2-typical usage
#' to modify the representation. The graphical objects are simply forwarded to
#' the ggplot2 object in `grobs` and added using [ggplot2::+.gg]. In addition,
#' some generic functions are implemented to visualize or examine
#' individual aspects of the overall plot in more detail. All available
#' generic functions are listed below:
#'
#' - \code{\link[=+.innsight_ggplot2]{+}}
#' - \code{\link[=plot.innsight_ggplot2]{plot}},
#' \code{\link[=print.innsight_ggplot2]{print}} and
#' \code{\link[=show.innsight_ggplot2]{show}}
#' (all behave the same)
#' - \code{\link[=[.innsight_ggplot2]{[}}
#' - \code{\link[=[[.innsight_ggplot2]{[[}}
#'
#' *Note:* In this case, the generic function `[<-` is not implemented
#' because there is only one ggplot2 object and not multiple ones.
#'
#' ## Multiple input layers (`multiplot = TRUE`)
#'
#' If the passed model has multiple input layers, a ggplot2 object is
#' created for each data point, input layer and output node and then stored
#' as a matrix in the slot `grobs`. During visualization, these are combined
#' using the function [`gridExtra::arrangeGrob`] and corresponding strips for
#' the output layer/node names are added at the top. The labels, column
#' indices and theme for the extra row of strips are stored in the slots
#' `output_strips` and `col_dims`. The strips for the input
#' layer and the data points (if not boxplot) are created using
#' [ggplot2::facet_grid] in the individual ggplot2 objects of the grob matrix.
#' An example structure is shown below:
#'
#' ```
#' |      Output 1: Node 1      |      Output 1: Node 3      |
#' |   Input 1   |   Input 2    |   Input 1   |   Input 2    |
#' |---------------------------------------------------------|-------------
#' |             |              |             |              |
#' | grobs[1,1]  |  grobs[1,2]  | grobs[1,3]  | grobs[1,4]   | data point 1
#' |             |              |             |              |
#' |---------------------------------------------------------|-------------
#' |             |              |             |              |
#' | grobs[2,1]  |  grobs[2,2]  | grobs[2,3]  | grobs[2,4]   | data point 2
#' |             |              |             |              |
#' ```
#'
#' Similar to the other case, generic functions are implemented to add
#' graphical objects from ggplot2, create the whole plot or select only
#' specific rows/columns. The difference, however, is that each entry in
#' each row and column is a separate ggplot2 object and can be modified
#' individually. For example, adds `+ ggplot2::xlab("X")` the x-axis label
#' "X" to all objects and not only to those in the last row. The generic
#' function \code{\link[=[<-.innsight_ggplot2]{[<-}} allows you to replace
#' a selection of objects in `grobs` and thus, for example, to change
#' the x-axis title only in the bottom row. All available
#' generic functions are listed below:
#'
#' - \code{\link[=+.innsight_ggplot2]{+}}
#' - \code{\link[=plot.innsight_ggplot2]{plot}},
#' \code{\link[=print.innsight_ggplot2]{print}} and
#' \code{\link[=show.innsight_ggplot2]{show}}
#' (all behave the same)
#' - \code{\link[=[.innsight_ggplot2]{[}}
#' - \code{\link[=[[.innsight_ggplot2]{[[}}
#' - \code{\link[=[<-.innsight_ggplot2]{[<-}}
#'
#' *Note:* Since this is not a standard visualization, the suggested packages
#' `'grid'`, `'gridExtra'` and `'gtable'` must be installed.
#'
#'
#' @name innsight_ggplot2
#' @rdname innsight_ggplot2-class
setClass("innsight_ggplot2", slots = list(
  grobs = "matrix", multiplot = "logical", output_strips = "list",
  col_dims = "list", boxplot = "logical"
))


#' Generic print, plot and show for `innsight_ggplot2`
#'
#' The class [innsight_ggplot2] provides the generic visualization functions
#' \code{\link{print}}, \code{\link{plot}} and \code{\link{show}}, which all
#' behave the same in this case. They create the plot of the results
#' (see [innsight_ggplot2] for details) and return it invisibly.
#'
#' @param x An instance of the S4 class [innsight_ggplot2].
#' @param object An instance of the S4 class [innsight_ggplot2].
#' @param ... Further arguments passed to the base function `print` if
#' `x@multiplot` is `FALSE`. Otherwise, if `x@multiplot` is `TRUE`, the
#' arguments are passed to [gridExtra::arrangeGrob].
#' @param y unused argument
#'
#' @return For multiple plots (`x@multiplot = TRUE`), a [gtable::gtable] and
#' otherwise a [ggplot2::ggplot] object is returned invisibly.
#'
#' @seealso [`innsight_ggplot2`],
#' [`+.innsight_ggplot2`],
#' \code{\link{[.innsight_ggplot2}},
#' \code{\link{[[.innsight_ggplot2}},
#' \code{\link{[<-.innsight_ggplot2}}
#'
#' @rdname innsight_ggplot2-print
#' @aliases print,innsight_ggplot2-method print.innsight_ggplot2
#' @export
setMethod(
  "print", list(x = "innsight_ggplot2"),
  function(x, ...) {
    if (x@multiplot) {
      # Arrange grobs
      mat <- matrix(seq_along(x@grobs), nrow = nrow(x@grobs))
      fig <- do.call(
        gridExtra::arrangeGrob,
        list(grobs = x@grobs, layout_matrix = mat, ...)
      )

      # Add strips for the output layer
      fig <- add_strips(fig, x)
      plot(fig)
    } else { # plotting only a single ggplot
      fig <- x@grobs[[1, 1]]
      print(fig, ...)
    }

    invisible(fig)
  }
)

#' @rdname innsight_ggplot2-print
#' @aliases show,innsight_ggplot2-method show.innsight_ggplot2
#' @export
setMethod(
  "show", list(object = "innsight_ggplot2"),
  function(object) {
    print(object)
  }
)

#' @rdname innsight_ggplot2-print
#' @aliases plot,innsight_ggplot2-method plot.innsight_ggplot2
#' @export
setMethod(
  "plot", list(x = "innsight_ggplot2"),
  function(x, y, ...) {
    print(x, ...)
  }
)

#' Generic add function for `innsight_ggplot2`
#'
#' This generic add function allows to treat an instance of [`innsight_ggplot2`]
#' as an ordinary plot object of [`ggplot2`]. For example geoms, themes and
#' scales can be added as usual (see [`ggplot2::+.gg`] for more information).\cr \cr
#' **Note:** If `e1` represents a multiplot (i.e., `e1@mulitplot = TRUE`),
#' `e2` is added to each individual plot. If only specific plots need to be
#' changed, the generic assignment function should be used (see
#' [innsight_ggplot2] for details).
#'
#' @param e1 An instance of the S4 class [`innsight_ggplot2`].
#' @param e2 An object of class [`ggplot2::ggplot`] or a [`ggplot2::theme`].
#'
#' @seealso [`innsight_ggplot2`],
#' [`print.innsight_ggplot2`],
#' \code{\link{[.innsight_ggplot2}},
#' \code{\link{[[.innsight_ggplot2}},
#' \code{\link{[<-.innsight_ggplot2}}
#'
#' @rdname innsight_ggplot2-plus
#' @aliases +,innsight_ggplot2,ANY-method +.innsight_ggplot2
#' @export
setMethod(
  "+", list(e1 = "innsight_ggplot2"),
  function(e1, e2) {
    if (e1@multiplot) {
      if (inherits(e2, "theme")) {
        # Update theme
        e1@output_strips$theme[names(e2)] <- e2[names(e2)]
        # Set ticks
        grobs <- set_theme(e1@grobs, e2, labels = FALSE)
      } else {
        # apply the given layer 'e2' to all entries
        grobs <- apply(e1@grobs, 1:2, function(grob) grob[[1]] + e2)
      }
    } else {
      grobs <- matrix(list(e1@grobs[[1, 1]] + e2))
    }

    new("innsight_ggplot2",
      grobs = grobs,
      multiplot = e1@multiplot,
      output_strips = e1@output_strips,
      col_dims = e1@col_dims,
      boxplot = e1@boxplot
    )
  }
)

#' Indexing plots of `innsight_ggplot2`
#'
#' The S4 class [`innsight_ggplot2`] visualizes the results in the form of
#' a matrix, with the output nodes (and also the input layers) in the columns
#' and the selected data points in the rows. With these basic generic indexing
#' functions, the plots of individual rows and columns can be accessed,
#' modified and the overall plot can be adjusted accordingly.
#'
#' @param x An instance of the s4 class [`innsight_ggplot2`].
#' @param i The numeric (or missing) index for the rows.
#' @param j The numeric (or missing) index for the columns.
#' @param value Another instance of the S4 class `innsight_ggplot2` but of
#' shape `i` x `j`.
#' @param drop unused argument
#' @param ... other unused arguments
#'
#' @return
#' * `[.innsight_ggplot2`: Selects only the plots from the `i`-th row and
#' `j`-th column and returns them as a new instance of [`innsight_ggplot2`].
#' * `[[.innsight_ggplot2`: Selects only the single plot in the `i`-th row and
#' `j`-th column and returns it an [ggplot2::ggplot] object.
#' * `[<-.innsight_ggplot2`: Replaces the plots from the `i`-th row and `j`-th
#' column with those from `value` and returns the modified instance of
#' [`innsight_ggplot2`].
#'
#' @seealso [`innsight_ggplot2`], [`print.innsight_ggplot2`],
#' [`+.innsight_ggplot2`]
#'
#' @rdname innsight_ggplot2-indexing
#' @aliases [,innsight_ggplot2-method [.innsight_ggplot2
#' @export
setMethod(
  "[", list(x = "innsight_ggplot2"),
  function(x, i, j, ..., drop = TRUE) {
    #----- Multiplot -----------------------------------------------------------
    if (x@multiplot) {
      # Check indices and set defaults (if necessary)
      if (missing(i)) i <- seq_len(nrow(x@grobs))
      if (missing(j)) j <- seq_len(ncol(x@grobs))
      cli_check(checkIntegerish(i, lower = 1, upper = nrow(x@grobs)), "i")
      cli_check(checkIntegerish(j, lower = 1, upper = ncol(x@grobs)), "j")

      # Get only selected grobs
      grobs <- x@grobs[i, j, drop = FALSE]

      # Set the facets
      grobs <- set_facets(grobs, x@boxplot)

      # Set labels, ticks and theme
      grobs <- set_theme(grobs, x@output_strips$theme,
        orig_grobs = x@grobs, i = i, j = j, labels = TRUE
      )

      # Update the arguments 'output_strips' and 'col_dims'
      res <- update_coldims_and_outstrips(x@output_strips, x@col_dims, j)
      col_dims <- res$col_dims
      output_strips <- res$output_strips
    } else {
      #----- Single plot ------------------------------------------------------
      # Get the saved facet vars in the ggplot2 object
      facet_rows <- x@grobs[[1, 1]]$facet$params$rows
      facet_cols <- x@grobs[[1, 1]]$facet$params$cols
      facet_rows <- if (length(facet_rows) > 0) {
        quo_name(facet_rows[[1]])
      } else {
        NULL
      }
      facet_cols <- if (length(facet_cols) > 0) {
        quo_name(facet_cols[[1]])
      } else {
        NULL
      }

      # Get the data for the plot
      data <- x@grobs[[1, 1]]$data

      # Get indices of the data with the facet names regarding the selected
      # row and column indices (i and j)
      idx_row <- get_facet_data_idx(facet_rows, i, data)
      idx_col <- get_facet_data_idx(facet_cols, j, data)

      # Get the subset of the data corresponding to the selected row and cols
      data <- data[idx_row & idx_col, ]

      # Update the plot with the new data
      grobs <- matrix(list(x@grobs[[1, 1]] %+% data))
      output_strips <- x@output_strips
      col_dims <- x@col_dims
    }

    new("innsight_ggplot2",
      grobs = grobs,
      multiplot = x@multiplot,
      boxplot = x@boxplot,
      col_dims = col_dims,
      output_strips = output_strips
    )
  }
)

#' @rdname innsight_ggplot2-indexing
#' @aliases [[,innsight_ggplot2-method [[.innsight_ggplot2
#' @export
setMethod(
  "[[", list(x = "innsight_ggplot2"),
  function(x, i, j, ...) {
    # Check indices
    if (x@multiplot) {
      upper_row <- nrow(x@grobs)
      upper_col <- ncol(x@grobs)
    } else {
      facet_rows <- x@grobs[[1, 1]]$facet$params$rows
      facet_cols <- x@grobs[[1, 1]]$facet$params$cols
      if (length(facet_rows) == 0) {
        upper_row <- 1
      } else {
        facet_rows <- quo_name(facet_rows[[1]])
        upper_row <- length(unique(x@grobs[[1, 1]]$data[[facet_rows]]))
      }
      if (length(facet_cols) == 0) {
        upper_col <- 1
      } else {
        facet_cols <- quo_name(facet_cols[[1]])
        upper_col <- length(unique(x@grobs[[1, 1]]$data[[facet_cols]]))
      }
    }
    cli_check(checkInt(i, lower = 1, upper = upper_row), "i")
    cli_check(checkInt(j, lower = 1, upper = upper_col), "j")

    x[i, j]@grobs[[1, 1]]
  }
)

#' @rdname innsight_ggplot2-indexing
#' @aliases [<-,innsight_ggplot2-method [<-.innsight_ggplot2
#' @export
setMethod(
  "[<-", list(x = "innsight_ggplot2"),
  function(x, i, j, ..., value) {
    if (!x@multiplot) {
      stopf("The method '[<-' is not implemented for single plots!")
    }

    # If missing, set defaults
    if (missing(i)) i <- seq_len(nrow(x@grobs))
    if (missing(j)) j <- seq_len(ncol(x@grobs))

    # Check indices
    cli_check(checkIntegerish(i, lower = 1, upper = nrow(x@grobs)), "i")
    cli_check(checkIntegerish(j, lower = 1, upper = ncol(x@grobs)), "j")

    if (is(value, "innsight_ggplot2")) {
      cli_check(checkTRUE(identical(c(length(i), length(j)), dim(value@grobs))),
                "identical(c(length(i), length(j)), dim(value@grobs))")

      # Remove facets of 'value'
      grobs_value <-
        apply(value@grobs, 1:2, function(grob) grob[[1]] + facet_grid())

      # Insert grobs from value
      grobs <- x@grobs
      grobs[i, j] <- grobs_value

      # Update facets
      grobs <- set_facets(grobs, x@boxplot)
    } else {
      warningf("Ignoring unknown object of class(es): ",
        paste(class(value), collapse = ", ")
      )
      grobs <- x@grobs
    }

    new("innsight_ggplot2",
      grobs = grobs,
      multiplot = x@multiplot, output_strips = x@output_strips,
      col_dims = x@col_dims, boxplot = x@boxplot
    )
  }
)


###############################################################################
#                              Utility functions
###############################################################################
generate_strips <- function(output_strips) {

  # Render strips based on ggplot2
  strips <- render_strips(
    x = output_strips$labels,
    labeller = label_value,
    theme = output_strips$theme
  )$x$top
  # get background color
  col <- output_strips$theme$plot.background$colour
  # Create rectangle for background
  rect <- grid::rectGrob(gp = grid::gpar(fill = col, col = col))
  for (i in seq_along(strips)) {
    l <- ifelse(i == 1, 20, 10)
    r <- ifelse(i == length(strips), 20, 10)
    # Add padding
    strips[[i]] <-
      gtable::gtable_add_padding(
        strips[[i]],
        grid::unit(c(0, r, 0, l), "points")
      )
    # Add background
    strips[[i]] <- gtable::gtable_add_grob(strips[[i]], rect,
      t = 1, l = 1, z = -Inf,
      b = nrow(strips[[i]]),
      r = ncol(strips[[i]])
    )
  }

  strips
}

set_theme <- function(grobs, theme, i = NULL, j = NULL, orig_grobs = NULL,
                      labels = FALSE) {
  # Otherwise all elements will be inherit from blank elements
  attr(theme, "complete") <- FALSE

  # For each entry in the matrix out of grobs..
  for (row in seq_len(nrow(grobs))) {
    for (col in seq_len(ncol(grobs))) {
      current_theme <- theme
      keys <- names(current_theme)
      xlabel <- NULL
      ylabel <- NULL

      # The theme should not effect all entries, only those laying in the
      # left column and bottom row
      if (col == 1 && row < nrow(grobs)) {
        idx <- keys[startsWith(keys, "axis.text.x") |
          startsWith(keys, "axis.ticks.x")]
        if (labels) {
          ylabel <- orig_grobs[[i[row], 1]]$labels$y
        }
      } else if (col > 1 && row == nrow(grobs)) {
        idx <- keys[startsWith(keys, "axis.text.y") |
          startsWith(keys, "axis.ticks.y")]
        if (labels) {
          xlabel <- orig_grobs[[nrow(orig_grobs), j[col]]]$labels$x
        }
      } else if (col > 1 && row < nrow(grobs)) {
        idx <- keys[startsWith(keys, "axis.text.x") |
          startsWith(keys, "axis.ticks.x") |
          startsWith(keys, "axis.text.y") |
          startsWith(keys, "axis.ticks.y")]
      } else {
        idx <- NULL
        if (labels) {
          ylabel <- orig_grobs[[i[row], 1]]$labels$y
          xlabel <- orig_grobs[[nrow(orig_grobs), j[col]]]$labels$x
        }
      }

      # Remove corresponding entries of the theme
      current_theme[idx] <- NULL
      # Set theme and labels
      if (labels) {
        grobs[[row, col]] <- grobs[[row, col]] + current_theme +
          xlab(xlabel) + ylab(ylabel)
      } else {
        grobs[[row, col]] <- grobs[[row, col]] + current_theme
      }
    }
  }

  grobs
}

add_strips <- function(gtab, object) {
  # Generate strips
  strips <- generate_strips(object@output_strips)

  # Get height of strips and add rows in the gtable
  h <- grid::unit(grid::convertHeight(
    grid::grobHeight(strips[[1]]), "points"
  ), units = "points")

  # Add row for strips
  gtab <- gtable::gtable_add_rows(gtab, h, pos = 0)

  # Add strips to the gtable
  l <- 0
  r <- 0
  for (i in seq_along(strips)) {
    l <- r + 1
    r <- l + object@col_dims[[i]] - 1
    gtab <- gtable::gtable_add_grob(gtab, strips[[i]],
      t = 1, l = l, r = r, z = -Inf
    )
  }

  gtab
}

set_facets <- function(grobs, boxplot = FALSE) {
  num_col <- ncol(grobs)

  # Set facet top
  for (col in seq_len(num_col - 1)) {
    grobs[[1, col]] <- grobs[[1, col]] +
      facet_grid(cols = vars(.data$model_input))
  }

  if (boxplot) {
    # Set facet corner top right
    grobs[[1, num_col]] <-
      grobs[[1, num_col]] +
      facet_grid(cols = vars(.data$model_input))
  } else {
    # Set facets right
    for (row in seq_len(nrow(grobs))[-1]) {
      grobs[[row, num_col]] <- grobs[[row, num_col]] +
        facet_grid(rows = vars(.data$data))
    }
    # Set facet corner top right
    grobs[[1, num_col]] <-
      grobs[[1, num_col]] +
      facet_grid(rows = vars(.data$data), cols = vars(.data$model_input))
  }

  grobs
}

update_coldims_and_outstrips <- function(output_strips, col_dims, col_idx) {
  # Create a list of the same length as 'col_dims', but in each entry are
  # the indices of the corresponding grobs of the matrix
  list_idx <- NULL
  value <- 0
  for (k in col_dims) {
    list_idx <- append(list_idx, list(seq_len(k) + value))
    value <- value + k
  }

  ## Update argument 'output_strips'
  # Get indices of 'list_idx' with at least one match with the selected
  # column indices ('col_idx')
  idx <- unlist(lapply(list_idx, function(k) any(col_idx %in% k)))
  # Update the corresponding labels of 'output_strips'
  output_strips$labels <- output_strips$labels[idx, , drop = FALSE]

  ## Update argument 'col_dims'
  # For each list entry in 'list_idx', get the total number of matches
  # with the selected column indices ('col_idx')
  col_dims <- lapply(list_idx, function(k) sum(col_idx %in% k))
  # Remove entries with zero matches
  col_dims <- col_dims[unlist(lapply(col_dims, function(k) k != 0))]

  list(output_strips = output_strips, col_dims = col_dims)
}

get_facet_data_idx <- function(facet_name, idx, data) {
  if (is.null(facet_name)) {
    if (!missing(idx)) {
      cli_check(checkInt(idx, lower = 1, upper = 1), "idx")
    }
    res_idx <- TRUE
  } else {
    levels_facet <- unique(data[[facet_name]])
    if (!missing(idx)) {
      cli_check(checkIntegerish(idx, lower = 1, upper = length(levels_facet)),
                "idx")
      levels_facet <- levels_facet[idx]
    }
    res_idx <- data[[facet_name]] %in% levels_facet
  }

  res_idx
}
