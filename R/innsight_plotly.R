
#' S4 class for plotly-based plots
#'
#' The S4 class `innsight_plotly` visualizes the results of the methods
#' provided from the package `innsight` using [plotly](https://plotly.com/r/).
#' In addition, it allows easier analysis of the results and modification of
#' the visualization by basic generic functions. The individual slots are for
#' internal use only and should not be modified.
#'
#' @slot plots The individual plotly objects arranged as a matrix (see
#' details for more information).
#' @slot shapes A list of two lists with the names `shapes_strips` and
#' `shapes_other`. The list `shapes_strips` contains the shapes for the
#' strips and may not be manipulated. The other list `shapes_other` contains
#' a matrix of the same size as `plots` and each entry contains the shapes
#' of the corresponding plot.
#' @slot annotations A list of two lists with the names `annotations_strips`
#' and `annotations_other`. The list `annotations_strips` contains the
#' annotations for the strips and may not be manipulated. The other list
#' `annotations_other` contains a matrix of the same size as `plots` and
#' each entry contains the annotations of the corresponding plot.
#' @slot multiplot A logical value indicating whether there are multiple
#' input layers and therefore correspondingly individual ggplot2 objects
#' instead of one single object.
#' @slot layout This list contains all global layout options, e.g. update
#' buttons, sliders, margins etc. (see [plotly::layout] for more details).
#' @slot col_dims A list to assign a label to the columns for the output
#' strips.
#'
#' @details
#' This S4 class is a simple extension of a plotly object that enables
#' a more detailed analysis of the results and a way to visualize the results
#' of models with multiple input layers (e.g., images and tabular data).
#'
#' The overall plot is created in the following order:
#'
#' 1. The corresponding shapes and annotations of the slots `annotations`
#' and `shapes` are added to each plot in `plots`. This also adds the strips
#' at the top for the output node (or input layer) and, if necessary, on the
#' right side for the data point.
#' 2. Subsequently, all individual plots are combined into one plot with
#' the help of the function [plotly::subplot].
#' 3. Lastly, the global elements from the `layout` slot are added and if there
#' are multiple input layers (`multiplot = TRUE`), another output strip is
#' added for the columns.
#'
#' An example structure of the plot with multiple input layers is shown below:
#' ```
#' |      Output 1: Node 1      |      Output 1: Node 3      |
#' |   Input 1   |   Input 2    |   Input 1   |   Input 2    |
#' |---------------------------------------------------------|-------------
#' |             |              |             |              |
#' | plots[1,1]  |  plots[1,2]  | plots[1,3]  | plots[1,4]   | data point 1
#' |             |              |             |              |
#' |---------------------------------------------------------|-------------
#' |             |              |             |              |
#' | plots[2,1]  |  plots[2,2]  | plots[2,3]  | plots[2,4]   | data point 2
#' |             |              |             |              |
#' ```
#'
#' Additionally, some generic functions are implemented to visualize individual
#' aspects of the overall plot or to examine them in more detail. All available
#' generic functions are listed below:
#'
#' - \code{\link[=plot.innsight_plotly]{plot}},
#' \code{\link[=print.innsight_plotly]{print}} and
#' \code{\link[=show.innsight_plotly]{show}}
#' (all behave the same)
#' - \code{\link[=[.innsight_plotly]{[}}
#' - \code{\link[=[[.innsight_plotly]{[[}}
#'
#' @name innsight_plotly
#' @rdname innsight_plotly-class
setClass("innsight_plotly", slots = list(
  plots = "matrix", shapes = "list", annotations = "list",
  multiplot = "logical",
  layout = "list", col_dims = "list"
))

#' Generic print, plot and show for `innsight_plotly`
#'
#' The class [`innsight_plotly`] provides the generic visualization functions
#' \code{\link{print}}, \code{\link{plot}} and \code{\link{show}}, which all
#' behave the same in this case. They create a plot of the results using
#' [`plotly::subplot`] (see [`innsight_plotly`] for details) and return it
#' invisibly.
#'
#' @param x An instance of the S4 class [`innsight_plotly`].
#' @param shareX A logical value whether the x-axis should be shared among
#' the subplots.
#' @param object An instance of the S4 class [`innsight_plotly`].
#' @param y unused argument
#' @param ... Further arguments passed to [`plotly::subplot`].
#'
#' @rdname innsight_plotly-print
#' @aliases print,innsight_plotly-method print.innsight_plotly
#' @export
setMethod(
  "print", list(x = "innsight_plotly"),
  function(x, shareX = TRUE, ...) {
    # Add annotations and shapes
    plots <- x@plots
    for (i in seq_len(nrow(plots))) {
      for (j in seq_len(ncol(plots))) {
        plots[[i, j]] <- plots[[i, j]] %>%
          plotly::layout(
            shapes = c(
              x@shapes$shapes_strips[[i, j]],
              x@shapes$shapes_other[[i, j]]
            ),
            annotations = c(
              x@annotations$annotations_strips[[i, j]],
              x@annotations$annotation_other[[i, j]]
            )
          )
      }
    }

    # Create plot
    fig <- plotly::subplot(t(plots),
      nrows = nrow(plots),
      shareX = shareX, ...
    )

    # Add global layout
    args <- x@layout
    args$p <- fig
    fig <- do.call(plotly::layout, args)

    # Add additional strips and labels for the output layer
    if (x@multiplot) {
      fig <- add_outputlayer_strips(fig, x@col_dims)
    }

    # Show the result
    print(fig)

    invisible(fig)
  }
)


#' @rdname innsight_plotly-print
#' @aliases show,innsight_plotly-method show.innsight_plotly
#' @export
setMethod(
  "show", list(object = "innsight_plotly"),
  function(object) {
    print(object)
  }
)

#' @rdname innsight_plotly-print
#' @aliases plot,innsight_plotly-method plot.innsight_plotly
#' @export
setMethod(
  "plot", list(x = "innsight_plotly"),
  function(x, ...) {
    print(x, ...)
  }
)

#' Indexing plots of `innsight_plotly`
#'
#' The S4 class [`innsight_plotly`] visualizes the results as a matrix of
#' plots based on [`plotly::plot_ly`]. The output nodes (and also input layers)
#' are displayed in the columns and the selected data points in the rows. With
#' these basic generic indexing functions, the plots of individual rows and
#' columns can be accessed.
#'
#' @param x An instance of the S4 class [`innsight_plotly`].
#' @param i The numeric (or missing) index for the rows.
#' @param j The numeric (or missing) index for the columns.
#' @param drop unused argument
#' @param ... other unused arguments
#'
#' @return
#' - `[.innsight_plotly`: Selects the plots from the i-th rows and j-th
#' columns and returns them as a new instance of `innsight_plotly`.
#' - `[[.innisght_plotly`: Selects only the single plot in the i-th row and
#' j-th column and returns it as a plotly object.
#'
#' @seealso [`innsight_plotly`], [`print.innsight_plotly`],
#' [`plot.innsight_plotly`], [`show.innsight_plotly`]
#'
#' @rdname innsight_plotly-indexing
#' @aliases [,innsight_plotly-method [.innsight_plotly
#' @export
setMethod(
  "[", list(x = "innsight_plotly"),
  function(x, i, j, ..., drop) {
    if (missing(i)) i <- seq_len(nrow(x@plots))
    if (missing(j)) j <- seq_len(ncol(x@plots))
    i <- sort(i)
    j <- sort(j)

    plots <- x@plots[i, j, drop = FALSE]

    #----- Adjust shapes and annotations --------------------------------------
    shapes_strips <- x@shapes$shapes_strips[i, j, drop = FALSE]
    annot_strips <- x@annotations$annotations_strips[i, j, drop = FALSE]

    # Set shapes and annotations top
    if (!(1 %in% i)) {
      for (k in seq_len(ncol(shapes_strips))) {
        old_strip <- x@shapes$shapes_strips[[1, j[k]]][1] # first entry is always strip top
        old_annot <- x@annotations$annotations_strips[[1, j[k]]][1]

        shapes_strips[[1, k]] <- append(shapes_strips[[1, k]], old_strip, 0)
        annot_strips[[1, k]] <- append(annot_strips[[1, k]], old_annot, 0)
      }
    }
    # Set shapes and annotations right
    num_cols_old <- ncol(x@plots)
    is_strip_right <- length(x@shapes$shapes_strips[[1, num_cols_old]]) > 1
    num_cols <- ncol(shapes_strips)
    if (is_strip_right & !(num_cols_old %in% j)) {
      for (k in seq_len(nrow(shapes_strips))) {
        old_strip <- x@shapes$shapes_strips[[i[k], num_cols_old]]
        old_strip <- old_strip[length(old_strip)] # last entry is always strip right
        old_annot <- x@annotations$annotations_strips[[i[k], num_cols_old]]
        old_annot <- old_annot[length(old_annot)]

        shapes_strips[[k, num_cols]] <-
          append(shapes_strips[[k, num_cols]], old_strip)
        annot_strips[[k, num_cols]] <-
          append(annot_strips[[k, num_cols]], old_annot)
      }
    }

    shapes <- list(
      shapes_strips = shapes_strips,
      shapes_other = x@shapes$shapes_other[i, j, drop = FALSE]
    )
    annot <- list(
      annotations_strips = annot_strips,
      annotations_other =
        x@annotations$annotations_other[i, j, drop = FALSE]
    )



    #----- Adjust buttons -----------------------------------------------------
    plot <- plotly::subplot(t(plots), nrows = nrow(plots))
    trace_types <- unlist(lapply(plot$x$data, function(i) i$type))
    trace_names <- unlist(lapply(plot$x$data, function(i) i$name))
    layout <- x@layout

    # Adjust slider
    if (is.null(trace_names)) {
      trace_idx <- NULL
    } else {
      trace_idx <- which(startsWith(trace_names, "image_"))
    }

    if (length(trace_idx) > 0) {
      trace_names_selection <- trace_names[trace_idx]
      for (idx in seq_len(9)) {
        arg_visible <- rep(FALSE, length(trace_names_selection))
        arg_visible[trace_names_selection == paste0("image_", idx)] <- TRUE
        layout$sliders[[1]]$steps[[idx]]$args <-
          list(list(visible = arg_visible), as.list(trace_idx - 1))
      }
    } else {
      layout$sliders <- NULL
    }

    # Adjust buttons
    remove_idx <- c()
    for (idx in seq_along(layout$updatemenus)) {
      name <- layout$updatemenus[[idx]]$name

      if (name == "button_heatmap_contour") {
        res <- update_button(layout, idx,
          trace_idx = which(trace_types == "heatmap"),
          annot_name = "annot_heatmap_contour"
        )
      } else if (name == "button_box_violin") {
        res <- update_button(layout, idx,
          trace_idx = which(trace_types == "box"),
          annot_name = "annot_box_violin"
        )
      } else if (name == "button_datapoints") {
        res <- update_button(layout, idx,
          trace_idx = which(startsWith(trace_names, "selected_points")),
          annot_name = "annot_datapoint"
        )
      } else if (name == "button_colorscale") {
        res <- update_button(layout, idx,
          trace_idx = which(trace_types %in% c("heatmap", "bar")),
          annot_name = "annot_colorscale"
        )
      }

      layout <- res$layout
      remove_idx <- c(remove_idx, res$remove_idx)
    }

    # Remove unused buttons
    if (length(remove_idx) > 0) {
      layout$updatemenus <- layout$updatemenus[-remove_idx]
    }

    # Adjust col_dims
    idx <- x@col_dims$col_idx[j]
    labels <- x@col_dims$col_label[unique(idx)]
    idx <- rep(seq_along(unique(idx)), as.vector(table(idx)))
    col_dims <- list(col_idx = idx, col_label = labels)


    new("innsight_plotly",
      plots = plots, shapes = shapes,
      annotations = annot, multiplot = x@multiplot,
      layout = layout, col_dims = col_dims
    )
  }
)

#' @rdname innsight_plotly-indexing
#' @aliases [[,innsight_plotly-method [[.innsight_plotly
#' @export
setMethod(
  "[[", list(x = "innsight_plotly"),
  function(x, i, j, ..., drop) {
    cli_check(checkInt(i, lower = 1, upper = nrow(x@plots)), "i")
    cli_check(checkInt(j, lower = 1, upper = ncol(x@plots)), "j")

    print(x[i, j])
  }
)
