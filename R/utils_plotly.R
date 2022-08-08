#' @importFrom stats quantile aggregate
#' @importFrom grDevices boxplot.stats
NULL

###############################################################################
#                       Main function for plotly
###############################################################################

create_plotly <- function(result_df, value_name = "Relevance",
                          include_data = TRUE, boxplot = FALSE, data_idx = NULL) {
  if (!requireNamespace("plotly", quietly = FALSE)) {
    stop(
      "Please install the 'plotly' package if you want to create an ",
      "interactive plot."
    )
  }

  #---- Pre-processing ---------------------------------------------------------
  # Do we have multiple input layers?
  multiplot <- if (length(unique(result_df$model_input)) > 1) TRUE else FALSE

  # Add name of the output layer to the output node
  if (length(levels(result_df$model_output)) > 1) {
    result_df$output_node <- paste0(
      as.character(result_df$model_output), ": ",
      as.character(result_df$output_node)
    )
  }

  # Create normalized fill value
  if (boxplot) {
    result_df <- result_df %>%
      plotly::group_by(.data$output_node) %>%
      plotly::mutate(fill = .data$value / max(abs(.data$value))) %>%
      plotly::group_by(.data$output_node, .data$model_input)
  } else {
    result_df <- result_df %>%
      plotly::group_by(.data$data, .data$output_node) %>%
      plotly::mutate(fill = .data$value / max(abs(.data$value))) %>%
      plotly::group_by(.data$data, .data$output_node, .data$model_input)
  }

  #---- Generate plots --------------------------------------------------------
  # Get number of plot rows and columns
  num_rows <- if (boxplot) 1 else length(unique(result_df$data))
  num_cols <- length(unique(result_df$output_node)) *
    length(unique(result_df$model_input))

  # Generate plots and plot data.frame
  showscale_env <- new.env()
  showscale_env$SHOWSCALE <- TRUE

  plot_df <- result_df %>%
    plotly::do(
      plots = create_single_plotly(.data,
        boxplot = boxplot,
        value_name = value_name,
        data_idx = data_idx,
        env = showscale_env
      ),
      dim = unique(.data$input_dimension)
    )
  plot_df$dim <- unlist(plot_df$dim)
  plots <- matrix(plot_df$plots,
    ncol = num_cols,
    nrow = num_rows,
    byrow = TRUE
  )

  #----- Generate basic shapes and annotations --------------------------------
  shapes_and_annot <- get_shape_and_annotations(
    plot_df = plot_df, nrows = num_rows, ncols = num_cols,
    multiplot = multiplot, input_names = unique(result_df$model_input),
    output_names = unique(result_df$output_node),
    data_names = levels(result_df$data),
    facet_right = include_data & !boxplot
  )

  # Create global layout
  if ("individual_data" %in% colnames(result_df)) {
    data_names <- unique(result_df$data[result_df$individual_data])
  } else {
    data_names <- NULL
  }
  layouts <- create_layout(plot_df, data_idx, data_names,boxplot, value_name)

  # Create column dims and labels
  col_dims <- list(
    col_idx = rep(seq_along(unique(result_df$output_node)),
      each = length(unique(result_df$model_input))
    ),
    col_label = unique(result_df$output_node)
  )

  new("innsight_plotly",
    plots = plots, shapes = shapes_and_annot$shapes,
    annotations = shapes_and_annot$annotations, multiplot = multiplot,
    layout = layouts, col_dims = col_dims
  )
}


###############################################################################
#                         Shapes and Annotations
###############################################################################

get_shape_and_annotations <- function(plot_df, nrows, ncols, multiplot,
                                      input_names, output_names, data_names,
                                      facet_right) {

  # Create zero line for y axis
  zeroline <- list(list(
    type = "line", y0 = 0, y1 = 0, x0 = 0, x1 = 1,
    line = list(color = "black"), xref = "paper"
  ))

  # Initialize shapes and annotation lists
  shapes_other <- lapply(plot_df$dim, function(dim) if (dim < 3) zeroline)
  shapes_other <- matrix(shapes_other,
    byrow = TRUE, nrow = nrows,
    ncol = ncols
  )
  shapes_strips <- matrix(list(), nrow = nrows, ncol = ncols)
  annotations_other <- matrix(list(), nrow = nrows, ncol = ncols)
  annotations_strips <- matrix(list(), nrow = nrows, ncol = ncols)

  # Get facet texts
  # For multiplots, we use the input layer as the facet text top. Otherwise,
  # the output_node is used
  if (multiplot) {
    text_top <- rep_len(input_names, length.out = ncols)
  } else {
    text_top <- as.character(output_names)
  }
  text_right <- as.character(data_names)

  # Add strips and annotations top row
  for (i in seq_len(ncols)) {
    shapes_strips[[1, i]] <- append(
      x = shapes_strips[[1, i]],
      values = list(get_strip("top")),
      after = 0
    )
    annotations_strips[[1, i]] <- append(
      x = annotations_strips[[1, i]],
      values = list(get_strip_annotation("top", text_top[i])),
      after = 0
    )
  }

  # Add strips and annotations last column
  if (facet_right) {
    for (i in seq_len(nrows)) {
      shapes_strips[[i, ncols]] <- append(
        x = shapes_strips[[i, ncols]],
        values = list(get_strip("right"))
      )
      annotations_strips[[i, ncols]] <- append(
        x = annotations_strips[[i, ncols]],
        values = list(get_strip_annotation("right", text_right[i]))
      )
    }
  }

  list(
    shapes = list(
      shapes_strips = shapes_strips,
      shapes_other = shapes_other
    ),
    annotations = list(
      annotations_strips = annotations_strips,
      annotations_other = annotations_other
    )
  )
}

###############################################################################
#                               Global Layout
###############################################################################

create_layout <- function(plot_df, data_idx, data_levels, boxplot, value_name) {
  sliders <- list()
  buttons <- list()
  annots <- list()
  yshift <- 0

  nrows <- if (boxplot) 1 else length(unique(plot_df$data))

  # Create subplot and get the types of the traces
  tmp_subplot <- plotly::subplot(plot_df$plots, nrows = nrows)
  trace_types <- unlist(lapply(tmp_subplot$x$data, function(x) x$type))
  trace_names <- unlist(lapply(tmp_subplot$x$data, function(x) x$name))

  # Create button for colorscale
  if (any(trace_types %in% c("bar", "heatmap"))) {
    button_and_annot <- button_colorscale(trace_types, 1)
    buttons <- append(buttons, list(button_and_annot$button))
    annots <- append(annots, list(button_and_annot$annotation))
  }

  # Buttons with annotations for tabular and 1D data
  if (any(trace_types == "box")) {
    # Create button for boxplot/violin (only for 1D and 2D data)
    button_and_annot <- button_box_violin(trace_types, 0.5)
    buttons <- append(buttons, list(button_and_annot$button))
    annots <- append(annots, list(button_and_annot$annotation))
    yshift <- 60

    # Create button for individual data points
    button_and_annot <- button_datapoints(trace_names, data_levels, data_idx, 1, 60)
    buttons <- append(buttons, list(button_and_annot$button))
    annots <- append(annots, list(button_and_annot$annotation))
  }

  if (any(plot_df$dim == 3)) {
    if (boxplot) {
      # Create slider for quantiles
      sliders <- append(sliders, list(slider_quantiles_image(trace_names)))
    }

    # Create button for 2D plot type
    button_and_annot <- button_heatmap_contour(trace_types, 0.5, yshift)
    buttons <- append(buttons, list(button_and_annot$button))
    annots <- append(annots, list(button_and_annot$annotation))
  }

  colaxis <- get_default_colorbar(value_name)

  list(
    margin = list(t = 50, r = 50),
    showlegend = FALSE,
    updatemenus = buttons,
    annotations = annots,
    sliders = sliders
  )
}

###############################################################################
#                            Buttons and Sliders
###############################################################################

#----- Colorscale -------------------------------------------------------------
# only relevant for heatmaps and bar plots
button_colorscale <- function(trace_types, y = 1) {
  trace_idx <- which(trace_types %in% c("heatmap", "bar"))

  get_color_button <- function(name) {
    if (endsWith(name, "(rev.)")) {
      reversed <- TRUE
      colorscale <- strsplit(name, " ")[[1]][1]
    } else {
      reversed <- FALSE
      colorscale <- name
    }

    if (identical(colorscale, "Default")) {
      colorscale <-
        '[[0, "rgb(0,0,255)"], [0.5, "rgb(255,255,255)"], [1, "rgb(255,0,0)"]]'
    }

    list(
      label = name,
      method = "restyle",
      args = list(list(
        colorscale = colorscale,
        reversescale = reversed,
        marker.reversescale = reversed,
        marker.colorscale = colorscale
      ), as.list(trace_idx - 1))
    )
  }
  colorscales <- c(
    "Default", "Blackbody", "Bluered", "Blues", "Cividis", "Earth", "Electric",
    "Greens", "Greys", "Hot", "Jet", "Picnic", "Portland", "Rainbow", "RdBu",
    "Reds", "Viridis", "YlGnBu", "YlOrRd")
  colorscales <- c(colorscales, paste0(colorscales, " (rev.)"))

  button <- list(
    y = y,
    x = -0.05,
    active = 0,
    name = "button_colorscale",
    showactive = TRUE,
    buttons = lapply(colorscales, get_color_button)
  )

  list(
    button = button,
    annotation = get_button_annot("Colorscale", y, "annot_colorscale")
  )
}

#----- Boxplot/Violin ---------------------------------------------------------
# only relevant for boxplots
button_box_violin <- function(trace_types, y = 0.5) {
  trace_idx <- which(trace_types == "box")

  button <- list(
    y = y,
    active = 0,
    showactive = TRUE,
    name = "button_box_violin",
    buttons = list(
      list(
        method = "restyle",
        args = list(list(type = "box"), as.list(trace_idx - 1)),
        label = "Boxplot"
      ),
      list(
        method = "restyle",
        args = list(list(type = "violin"), as.list(trace_idx - 1)),
        label = "Violin"
      )
    )
  )

  list(
    button = button,
    annotation = get_button_annot("1D Plot Type", y, "annot_box_violin")
  )
}

#----- Individual datapoints --------------------------------------------------
# only relevant for boxplots
button_datapoints <- function(trace_names, data_levels, data_idx, y = 0.7, yshift = 0) {
  create_button <- function(idx, trace_names, data_levels) {
    trace_idx <- which(startsWith(trace_names, "selected_points"))
    arg_visible <- rep(FALSE, length(trace_idx))

    if (idx != 0) {
      name_idx <- trace_names[trace_idx] == paste0("selected_points_", idx)
      arg_visible[name_idx] <- TRUE
      label <- data_levels[idx]
    } else {
      label <- "None"
    }

    list(
      method = "restyle",
      args = list(list(visible = arg_visible), as.list(trace_idx - 1)),
      label = label
    )
  }

  button <- list(
    y = y,
    active = if (is.null(data_idx)) 0 else which(data_levels == paste0("data_", data_idx)),
    showactive = TRUE,
    pad = list(t = yshift),
    name = "button_datapoints",
    buttons = lapply(c(0, seq_along(data_levels)),
      FUN = create_button,
      trace_names = trace_names,
      data_levels = data_levels
    )
  )

  list(
    button = button,
    annotation = get_button_annot("Data point", y, "annot_datapoint", yshift)
  )
}

#----- Slider for quantiles ----------------------------------------------------
# only for images if boxplot == TRUE
slider_quantiles_image <- function(trace_names) {
  create_slider_step <- function(idx, trace_names, labels) {
    trace_idx <- which(startsWith(trace_names, "image_"))
    trace_names_selection <- trace_names[trace_idx]
    arg_visible <- rep(FALSE, length(trace_names_selection))
    arg_visible[trace_names_selection == paste0("image_", idx)] <- TRUE

    list(
      method = "restyle",
      label = labels[idx],
      args = list(list(visible = arg_visible), as.list(trace_idx - 1))
    )
  }

  labels <- c(
    "min", "q12.5%", "q25%", "q37.5", "median", "q62.5%", "q75%",
    "q87.5%", "max"
  )

  list(
    active = 4,
    steps = lapply(seq_len(9),
      FUN = create_slider_step,
      trace_names = trace_names, labels = labels
    )
  )
}

#----- Heatmap/Contour plot ---------------------------------------------------
# only relevant for heatmaps
button_heatmap_contour <- function(trace_types, y = 0.4, yshift = 0) {
  trace_idx <- which(trace_types == "heatmap")
  button <- list(
    y = y,
    active = 0,
    showactive = TRUE,
    name = "button_heatmap_contour",
    pad = list(t = yshift),
    buttons = list(
      list(
        method = "restyle",
        args = list(list(type = "heatmap"), as.list(trace_idx - 1)),
        label = "Heatmap"
      ),
      list(
        method = "restyle",
        args = list(list(type = "contour"), as.list(trace_idx - 1)),
        label = "Contour"
      )
    )
  )

  list(
    button = button,
    annotation = get_button_annot("2D Plot Type", y, "annot_heatmap_contour", yshift)
  )
}

###############################################################################
#                        Create single Plotly-Plots
###############################################################################

create_single_plotly <- function(data, boxplot = FALSE, data_idx = NULL,
                                 value_name = "Relevance", env = NULL) {
  input_dim <- unique(data$input_dimension)

  if (input_dim == 3) {
    if (boxplot) {
      plot <- plotly_boxplot_image(data, value_name, env$SHOWSCALE)
    } else {
      plot <- plotly_image(data, value_name, env$SHOWSCALE)
    }
    axis <- list(
      showticklabels = FALSE,
      showline = FALSE,
      zeroline = FALSE,
      ticks = "", title = "")
    plot <- plot %>%
      plotly::layout(xaxis = axis, yaxis = axis)
    env$SHOWSCALE <- FALSE
  } else {
    if (boxplot) {
      plot <- plotly_boxplot(data, value_name, data_idx)
    } else {
      plot <- plotly_bar(data, value_name)
    }

    axis <- list(
      title = "",
      showgrid = TRUE,
      gridcolor = "#ffffff"
    )
    plot <- plot %>%
      plotly::layout(
        plot_bgcolor = "#ebebeb",
        xaxis = axis,
        yaxis = axis
      )
  }

  plot
}


plotly_bar <- function(data, scale_title) {

  hovertext <- get_hovertext(data, scale_title, title_br = TRUE)
  colaxis <- get_default_colorbar(scale_title)

  if (all(data$input_dimension == 2)) {
    data$feature <- as.numeric(data$feature)
  } else {
    data$feature <- factor(data$feature, levels = unique(data$feature))
  }


  plotly::plot_ly() %>%
    plotly::add_trace(
      x = data$feature, y = data$value,
      type = "bar", marker = list(
        colorscale = colaxis$colorscale, color = data$fill, cmin = -1,
        cmid = 0, cmax = 1, colorbar = colaxis$colorbar, showscale = FALSE
      ),
      text = hovertext, textposition = "none",
      hoverlabel = list(align = "left"),
      hovertemplate = paste("%{text}")
    )
}

plotly_image <- function(data, scale_title = "Relevance", showscale = FALSE) {
  height <- length(unique(data$feature))
  width <- length(unique(data$feature_2))

  hovertext <- get_hovertext(data, scale_title, title_br = TRUE, matrix = TRUE)
  colaxis <- get_default_colorbar(scale_title)

  plotly::plot_ly() %>%
    plotly::add_trace(
      z = matrix(data$fill, ncol = width, nrow = height), zmid = 0, zmax = 1,
      zmin = -1, type = "heatmap", hoverinfo = "text", text = hovertext,
      colorscale = colaxis$colorscale, colorbar = colaxis$colorbar,
      showscale = showscale, hovertemplate = paste("%{text}"),
      hoverlabel = list(bgcolor = "rgb(80,80,80)", align = "left")
    )
}


plotly_boxplot <- function(data, scale_title, data_idx) {
  data$feature <- factor(data$feature, levels = unique(data$feature))

  ref_data <- data[data$individual_data, ]
  data <- data[data$boxplot_data, ]

  outliers <- data.frame()
  for (feat in unique(data$feature)) {
    dat <- data[data$feature == feat, ]
    stats <- boxplot.stats(dat$value, do.conf = FALSE)
    outliers <- rbind(outliers, dat[which(dat$value %in% stats$out), ])
  }

  p <- plotly::plot_ly() %>%
    plotly::add_trace(
      type = "box", data = data, x = ~ as.numeric(feature), y = ~value,
      hoveron = "boxes+kde+violin", name = "boxes_violins",
      hoverinfo = "y", marker = list(opacity = 0),
      hoverlabel = list(bgcolor = "rgb(80,80,80)", align = "left"),
      fillcolor = "rgba(40,40,40, 0.75)",
      line = list(color = "rgb(50,50,50)", size = 1)
    )

  if (all(data$input_dimension == 1)) {
    p <- p %>%
      plotly::layout(
        xaxis = list(
          tickvals = unique(as.numeric(data$feature)),
          ticktext = levels(data$feature)
        )
      )
  }

  # Add individual points
  i <- 1
  data_idx <- if (is.null(data_idx)) 0 else data_idx
  for (data_name in unique(ref_data$data)) {
    visible <- if (data_name == paste0("data_",data_idx)) TRUE else FALSE
    single_data <- ref_data[ref_data$data == data_name, ]
    hovertext <- get_hovertext(single_data, scale_title, title_br = FALSE)
    p <- p %>%
      plotly::add_segments(
        data = single_data, x = ~ (as.numeric(feature) - 0.375),
        y = ~value, xend = ~ (as.numeric(feature) + 0.375),
        yend = ~value, color = I("red"), name = paste0("selected_points_", i),
        text = hovertext,
        visible = visible, hoverlabel = list(align = "left"),
        hovertemplate = paste0("<b>Selected Point</b><br><br>%{text}")
      )
    i <- i + 1
  }

  if (nrow(outliers) > 0) {
    hovertext <- get_hovertext(outliers, scale_title, title_br = FALSE)
    p <- p %>%
      plotly::add_markers(
        name = "outliers",
        x = as.numeric(outliers$feature), y = outliers$value,
        text = hovertext,
        hovertemplate = paste("<b>Outlier</b><br><br>%{text}"),
        marker = list(color = "rgb(50,50,50)", opacity = 0.8, size = 2),
        hoverlabel = list(bgcolor = "rgb(80,80,80)", align = "left")
      )
  }

  p
}

plotly_boxplot_image <- function(data, scale_title = "Relevance", showscale = FALSE) {

  height <- length(unique(data$feature))
  width <- length(unique(data$feature_2))

  data <- data[data$boxplot_data, ]

  data$feature <- factor(data$feature, levels = unique(data$feature))
  data$feature_2 <- factor(data$feature_2, levels = unique(data$feature_2))
  colaxis <- get_default_colorbar(scale_title)

  value_stats <- aggregate(data$value,
    by = list(feature = data$feature, feature_2 = data$feature_2),
    FUN = function(x) quantile(x, seq(0, 1, 0.125))
  )
  fill_stats <- aggregate(data$fill,
    by = list(feature = data$feature, feature_2 = data$feature_2),
    FUN = function(x) quantile(x, seq(0, 1, 0.125))
  )

  dats <- lapply(1:9, function(i)
    data.frame(value_stats[,1:2], model_input = data$model_input[1],
               output_node = data$output_node[1],
               model_output = data$model_output[1],
               input_dimension = data$input_dimension[1],
               value = value_stats$x[, i], fill = fill_stats$x[, i]))

  labels <- paste0("Percentile (", colnames(fill_stats$x), ")")
  labels[c(1, 5, 9)] <- c("Minimum", "Median", "Maximum")

  p <- plotly::plot_ly()

  for (i in seq_len(9)) {
    hovertext <- get_hovertext(dats[[i]], scale_title, FALSE, TRUE, FALSE)

    p <- p %>%
      plotly::add_trace(
        z = matrix(dats[[i]]$fill, nrow = height, ncol = width),
        type = "heatmap", name = paste0("image_", i),
        zmin = -1, zmid = 0, zmax = 1,
        colorbar = colaxis$colorbar, colorscale = colaxis$colorscale,
        text = hovertext, showscale = showscale,
        hovertemplate = paste("<b>", labels[i], "</b><br><br>%{text}"),
        hoverlabel = list(bgcolor = "rgb(80,80,80)", align = "left"),
        visible = if (i == 5) TRUE else FALSE
      )
  }

  p
}


###############################################################################
#                             Utils
###############################################################################


get_strip <- function(side, x0 = 0, x1 = NULL, y0 = 0, y1 = NULL) {
  if (is.null(x1)) {
    x1 <- if (identical(side, "top")) 1 else 22
  }
  if (is.null(y1)) {
    y1 <- if (identical(side, "top")) 22 else 1
  }

  list(
    type = "rect",
    x0 = x0,
    x1 = x1,
    xref = "paper",
    xanchor = if (identical(side, "top")) NULL else 1,
    xsizemode = if (identical(side, "top")) "scaled" else "pixel",
    y0 = y0,
    y1 = y1,
    yref = "paper",
    yanchor = if (identical(side, "top")) 1 else NULL,
    ysizemode = if (identical(side, "top")) "pixel" else "scaled",
    fillcolor = "rgba(204,204,204,1)",
    line = list(color = "transparent")
  )
}

get_strip_annotation <- function(side, text, x = NULL, yshift = 0) {
  if (is.null(x)) {
    x <- if (identical(side, "top")) 0.5 else 1
  }

  list(
    text = text,
    x = x,
    y = if (identical(side, "top")) 1 else 0.5,
    xref = "paper",
    yref = "paper",
    yshift = yshift,
    showarrow = FALSE,
    xanchor = if (identical(side, "top")) "center" else "left",
    yanchor = if (identical(side, "top")) "bottom" else "middle",
    textangle = if (identical(side, "top")) 0 else 90,
    font = list(face = "bold")
  )
}

get_default_colorbar <- function(title) {
  colorbar <- list(
    lenmode = "fraction", len = 0.8, y = 0.5, yanchor = "middle",
    x = 1.02, xanchor = "left", xpad = 22,
    title = list(text = paste0("<b>", title, "</b><br>(normalized)"))
  )
  colorscale <- list(list(0, "blue"), list(0.5, "white"), list(1, "red"))

  list(colorbar = colorbar, colorscale = colorscale)
}

get_hovertext <- function(data, scale_title, title_br = TRUE, matrix = FALSE,
                          include_data = TRUE) {
  dim <- unique(data$input_dimension)[1]
  if (dim == 1) {
    feat_title <- "Feature: "
  } else if (dim == 2) {
    feat_title <- "Length:  "
  } else {
    feat_title <- "Height:   "
  }

  hovertext <- paste0(
    "<b>", scale_title, ":</b>  ", signif(data$value, 4), "<br>",
    if (title_br) "<br>" else "",
    "<b>Input:</b>     ", data$model_input, "<br>",
    "<b>", feat_title, "</b>", data$feature,
    if (dim == 3) paste0("<br><b>Width:</b>    ", data$feature_2) else "",
    "<extra>",
    if (include_data) paste0(data$data, "<br>") else "",
    data$model_output, "<br>",
    sub(".*: ", "", data$output_node), "</extra>"
  )

  if (matrix) {
    hovertext <- matrix(hovertext,
      nrow = length(unique(data$feature)),
      ncol = length(unique(data$feature_2))
    )
  }

  hovertext
}

get_button_annot <- function(text, y, name, yshift = 0) {
  list(
    text = paste0("<b>", text, "</b>"),
    font = list(size = 15),
    y = y,
    name = name,
    yanchor = "bottom",
    yshift = -yshift,
    showarrow = FALSE,
    x = -0.05,
    xanchor = "right",
    xref = "paper",
    yref = "paper"
  )
}

add_outputlayer_strips <- function(plot, col_dims) {
  # Get shapes and annotations in the plotly plot
  shapes <- plot$x$layout$shapes
  annot <- plot$x$layout$annotations

  # Now add for each output layer the corresponding strip
  for (i in seq_along(col_dims$col_label)) {
    # Get the indices of the relevant x axis and create the corresponding names
    # (plotly creats xaxis in increasing order, i.e. xaxis, xaxis2, xaxis3,...)
    idx <- which(col_dims$col_idx == i)
    axis_names <- ifelse(idx == 1, "xaxis", paste0("xaxis", idx))
    # Get the axis domains
    domain <- unlist(lapply(plot$x$layout[axis_names], function(ax) ax$domain))
    # Add shape and annotation for the output layer strip
    new_shape <- get_strip("top", x0 = min(domain), x1 = max(domain),
                           y0 = 25, y1 = 47)
    new_annot <- get_strip_annotation("top", col_dims$col_label[i],
                                      x = 0.5 * (max(domain) + min(domain)),
                                      yshift = 25)
    shapes <- c(shapes, list(new_shape))
    annot <- c(annot, list(new_annot))
  }

  plot$x$layout$shapes <- shapes
  plot$x$layout$annotations <- annot
  plot$x$layout$margin$t <- plot$x$layout$margin$t + 50

  plot
}

update_button <- function(layout, idx, trace_idx, annot_name) {
  buttons <- layout$updatemenus[[idx]]$buttons
  remove_idx <- NULL
  if (length(trace_idx) == 0) {
    annot_names <- unlist(lapply(layout$annotations, function(x) x$name))
    layout$annotations[which(annot_names == annot_name)] <- NULL
    remove_idx <- idx
  } else {
    for (b_idx in seq_along(buttons)) {
      buttons[[b_idx]]$args[[2]] <- as.list(trace_idx - 1)
    }
    layout$updatemenus[[idx]]$buttons <- buttons
  }

  list(layout = layout, remove_idx = remove_idx)
}
