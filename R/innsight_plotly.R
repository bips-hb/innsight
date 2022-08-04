

setClass("innsight_plotly", slots = list(
  plots = "matrix", shapes = "list", annotations = "list", multiplot = "logical",
  boxplot = "logical", layout = "list", col_dims = "list"))

setMethod(
  "print", list(x = "innsight_plotly"),
  function(x, shareX = TRUE, ...) {
    # Add annotations and shapes
    plots <- x@plots
    for (i in seq_len(nrow(plots))) {
      for (j in seq_len(ncol(plots))) {
        plots[[i,j]] <- plots[[i,j]] %>%
          plotly::layout(
            shapes = c(
              x@shapes$shapes_strips[[i,j]],
              x@shapes$shapes_other[[i,j]]
            ),
            annotations = c(
              x@annotations$annotations_strips[[i,j]],
              x@annotations$annotation_other[[i,j]])
          )
      }
    }

    # Create plot
    fig <- plotly::subplot(t(plots), nrows = nrow(plots),
                           shareX = shareX, ...)

    # Add global layout
    args <- x@layout
    args$p <- fig
    fig <- do.call(plotly::layout, args)

    # Add additional strips and labels for the output layer
    if (x@multiplot) {
      fig <- add_outputlayer_strips(fig, x@col_dims)
    }

    fig
  }
)


setMethod(
  "show", list(object = "innsight_plotly"),
  function(object) {
    print(object)
  }
)



setMethod(
  "plot", list(x = "innsight_plotly"),
  function(x, ...) {
    print(x, ...)
  }
)


setMethod(
  "[", list(x = "innsight_plotly"),
  function(x, i, j, ..., drop) {
    if (missing(i)) i <- seq_len(nrow(x@plots))
    if (missing(j)) j <- seq_len(ncol(x@plots))
    i <- sort(i)
    j <- sort(j)

    plots <- x@plots[i,j, drop = FALSE]

    #----- Adjust shapes and annotations ---------------------------------------
    shapes_strips <- x@shapes$shapes_strips[i,j, drop = FALSE]
    annot_strips <- x@annotations$annotations_strips[i,j, drop = FALSE]

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

    shapes <- list(shapes_strips = shapes_strips,
                   shapes_other = x@shapes$shapes_other[i,j, drop = FALSE])
    annot <- list(annotations_strips = annot_strips,
                  annotations_other =
                    x@annotations$annotations_other[i,j, drop = FALSE])



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
                             annot_name = "annot_heatmap_contour")
      } else if (name == "button_box_violin") {
        res <- update_button(layout, idx,
                             trace_idx =  which(trace_types == "box"),
                             annot_name = "annot_box_violin")
      } else if (name == "button_datapoints") {
        res <- update_button(layout, idx,
                             trace_idx =  which(startsWith(trace_names, "selected_points")),
                             annot_name = "annot_datapoint")
      } else if (name == "button_colorscale") {
        res <- update_button(layout, idx,
                             trace_idx =  which(trace_types %in% c("heatmap", "bar")),
                             annot_name = "annot_colorscale")
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


    new("innsight_plotly", plots = plots, shapes = shapes,
        annotations = annot, multiplot = x$multiplot,
        boxplot = x$boxplot, layout = layout, col_dims = col_dims)
  }
)


setMethod(
  "[[", list(x = "innsight_plotly"),
  function(x, i, j, ..., drop) {
    assertInt(i, lower = 1, upper = nrow(x@plots))
    assertInt(j, lower = 1, upper = ncol(x@plots))

    print(x[i,j])
  }
)




