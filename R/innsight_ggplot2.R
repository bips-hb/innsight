
setClass("innsight_ggplot2", slots = list(
  grobs = "matrix", multiplot = "logical", output_strips = "list",
  col_dims = "list", boxplot = "logical"
))

setMethod(
  "print", list(x = "innsight_ggplot2"),
  function(x, ...) {
    if (x@multiplot) {

      # Arrange grobs
      mat <- matrix(seq_along(x@grobs), nrow = nrow(x@grobs))
      fig <- do.call(arrangeGrob,
                      list(grobs = x@grobs, layout_matrix = mat, ...))

      # Add strips for the output layer
      fig <- add_strips(fig, x)
      plot(fig)
    }
    else { # plotting only a single ggplot
      fig <- x@grobs[[1,1]]
      print(fig, ...)
    }

    invisible(fig)
  }
)

setMethod(
  "show", list(object = "innsight_ggplot2"),
  function(object) {
    print(object)
  }
)

setMethod(
  "plot", list(x = "innsight_ggplot2"),
  function(x, ...) {
    print(x, ...)
  }
)
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
      grobs <- matrix(list(e1@grobs[[1,1]] + e2))
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

setMethod(
  "[", list(x = "innsight_ggplot2"),
  function(x, i, j, ..., drop) {
    #----- Multiplot -----------------------------------------------------------
    if (x@multiplot) {
      # Check indices and set defaults (if necessary)
      if (missing(i)) i <- seq_len(nrow(x@grobs))
      if (missing(j)) j <- seq_len(ncol(x@grobs))
      assertIntegerish(i, lower = 1, upper = nrow(x@grobs))
      assertIntegerish(j, lower = 1, upper = ncol(x@grobs))

      # Get only selected grobs
      grobs <- x@grobs[i, j, drop = FALSE]

      # Set the facets
      grobs <- set_facets(grobs, x@boxplot)

      # Set labels, ticks and theme
      grobs <- set_theme(grobs, x@output_strips$theme,
                         orig_grobs = x@grobs, i = i, j = j, labels = TRUE)

      # Update the arguments 'output_strips' and 'col_dims'
      res <- update_coldims_and_outputstrips(x@output_strips, x@col_dims, j)
      col_dims <- res$col_dims
      output_strips <- res$output_strips
    }
    #----- Single plot ---------------------------------------------------------
    else {
      # Get the saved facet vars in the ggplot2 object
      facet_rows <- x@grobs[[1,1]]$facet$params$rows
      facet_cols <- x@grobs[[1,1]]$facet$params$cols
      facet_rows <- if (length(facet_rows) > 0)
        quo_name(facet_rows[[1]]) else NULL
      facet_cols <- if (length(facet_cols) > 0)
        quo_name(facet_cols[[1]]) else NULL

      # Get the data for the plot
      data <- x@grobs[[1,1]]$data

      # Get indices of the data with the facet names regarding the selected
      # row and column indices (i and j)
      idx_row <- get_facet_data_idx(facet_rows, i, data)
      idx_col <- get_facet_data_idx(facet_cols, j, data)

      # Get the subset of the data corresponding to the selected row and cols
      data <- data[idx_row & idx_col, ]

      # Update the plot with the new data
      grobs <- matrix(list(x@grobs[[1,1]] %+% data))
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

setMethod(
  "[[", list(x = "innsight_ggplot2"),
  function(x, i, j, ..., drop) {
    # Check indices
    if (x@multiplot) {
      upper_row <- nrow(x@grobs)
      upper_col <- ncol(x@grobs)
    } else {
      facet_rows <- quo_name(x@grobs[[1,1]]$facet$params$rows[[1]])
      facet_cols <- quo_name(x@grobs[[1,1]]$facet$params$cols[[1]])
      upper_row <- length(unique(x@grobs[[1,1]]$data[[facet_rows]]))
      upper_col <- length(unique(x@grobs[[1,1]]$data[[facet_cols]]))
    }
    assertInt(i, lower = 1, upper = upper_row)
    assertInt(j, lower = 1, upper = upper_col)

    x[i,j]@grobs[[1,1]]
  }
)


setMethod(
  "[<-", list(x = "innsight_ggplot2"),
  function(x, i, j, ..., value) {
    if (!x@multiplot) {
      stop("The method '[<-' is not implemented for single plots!",
           call. = FALSE)
    }

    # If missing, set defaults
    if (missing(i)) i <- seq_len(nrow(x@grobs))
    if (missing(j)) j <- seq_len(ncol(x@grobs))

    # Check indices
    assertIntegerish(i, lower = 1, upper = nrow(x@grobs))
    assertIntegerish(j, lower = 1, upper = ncol(x@grobs))

    if (is(value, "innsight_ggplot2")) {
      assertTRUE(identical(c(length(i), length(j)), dim(value@grobs)))

      # Remove facets of 'value'
      grobs_value <-
        apply(value@grobs, 1:2, function(grob) grob[[1]] + facet_grid())

      # Insert grobs from value
      grobs <- x@grobs
      grobs[i,j] <- grobs_value

      # Update facets
      grobs <- set_facets(grobs, x@boxplot)
    } else {
      message(paste0(
        "Ignoring unknown object of class(es): ",
        paste(class(value), collapse = ", ")
      ))
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
  rect <- rectGrob(gp = gpar(fill = col, col = col))
  for (i in seq_along(strips)) {
    l <- ifelse(i == 1, 20, 10)
    r <- ifelse(i == length(strips), 20, 10)
    # Add padding
    strips[[i]] <-
      gtable_add_padding(strips[[i]], unit(c(0, r, 0, l), "points"))
    # Add background
    strips[[i]] <- gtable_add_grob(strips[[i]], rect,
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
      if (col == 1 & row < nrow(grobs)) {
        idx <- keys[startsWith(keys, "axis.text.x") |
                      startsWith(keys, "axis.ticks.x")]
        if (labels) {
          ylabel <- orig_grobs[[i[row], 1]]$labels$y
        }
      } else if (col > 1 & row == nrow(grobs)) {
        idx <- keys[startsWith(keys, "axis.text.y") |
                      startsWith(keys, "axis.ticks.y")]
        if (labels) {
          xlabel <- orig_grobs[[nrow(orig_grobs), j[col]]]$labels$x
        }
      } else if (col > 1 & row < nrow(grobs)) {
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
  h <- unit(convertHeight(grobHeight(strips[[1]]), "points"), units = "points")

  # Add row for strips
  gtab <- gtable_add_rows(gtab, h, pos = 0)

  # Add strips to the gtable
  l <- 0
  r <- 0
  for (i in seq_along(strips)) {
    l <- r + 1
    r <- l + object@col_dims[[i]] - 1
    gtab <- gtable_add_grob(gtab, strips[[i]], t = 1, l = l, r = r, z = -Inf)
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

update_coldims_and_outputstrips <- function(output_strips, col_dims, col_idx) {
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
      assertInt(idx, lower = 1, upper = 1)
    }
    res_idx <- TRUE
  } else {
    levels_facet <- unique(data[[facet_name]])
    if (!missing(idx)) {
      assertIntegerish(idx, lower = 1, upper = length(levels_facet))
      levels_facet <- levels_facet[idx]
    }
    res_idx <- data[[facet_name]] %in% levels_facet
  }

  res_idx
}
