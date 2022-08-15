#' @importFrom stats ave median

###############################################################################
#                             Plot function
###############################################################################

#----- Main plot function -----------------------------------------------------
create_ggplot <- function(result_df, value_name = "Relevance",
                      include_data = TRUE, boxplot = FALSE, data_idx = NULL) {

  num_inputs <- length(unique(result_df$model_input))
  num_outputs <- length(unique(result_df$model_output))

  # regular plots for neural networks with one input and one output layer
  if (num_inputs == 1 && num_outputs == 1) {
    facet_rows <- if (include_data) "data" else NULL

    if (all(result_df$input_dimension == 3)) {
      p <- plot_image(result_df, value_name,
                      facet_rows = facet_rows,
                      facet_cols = "output_node",
                      boxplot = boxplot)
    } else {
      p <- plot_bar(result_df, value_name,
                    facet_rows = facet_rows,
                    facet_cols = "output_node",
                    boxplot = boxplot,
                    data_idx = data_idx)
    }

    p <- new("innsight_ggplot2",
             grobs = matrix(list(p)),
             output_strips = list(),
             col_dims = list(),
             boxplot = boxplot,
             multiplot = FALSE)
  } else {
    # This is for models with multiple input and/or output layers
    p <- plot_extended(result_df, value_name, include_data, boxplot, data_idx)
  }

  p
}

#----- Plot function for 1D and 2D --------------------------------------------
plot_bar <- function(result_df, value_name = "value", facet_rows = NULL,
                     facet_cols = NULL, calc_fill = TRUE, xticks = TRUE,
                     yticks = TRUE, boxplot = FALSE, data_idx = NULL) {

  if (boxplot) {
    facet_rows <- NULL
  }

  # normalize result for all data points, if FALSE then 'result_df' needs
  # the column 'fill'
  if (calc_fill && !boxplot) {
    result_df$fill <- result_df$value /
      ave(result_df$value, as.character(result_df$data),
          as.character(result_df$output_node),
          FUN = function(x) max(abs(x)))
  }

  # Depending on the input dimension, create labels, hovertext and x_scale
  if (all(result_df$input_dimension == 2)) {
    x_label <- "Signal Length"
    result_df$feature <- as.numeric(result_df$feature)
    x_scale <- scale_x_continuous(expand = c(0, 0))
  } else {
    x_label <- "Feature"
    x_scale <- scale_x_discrete(guide = guide_axis(check.overlap = TRUE))
  }

  # Define facets
  facet_rows <- if (is.null(facet_rows)) NULL else vars(.data[[facet_rows]])
  facet_cols <- if (is.null(facet_cols)) NULL else vars(.data[[facet_cols]])
  facet <- facet_grid(cols = facet_cols, rows = facet_rows, scales = "free_y")

  # Create plot/boxplot
  if (boxplot) {
    ref_data <- result_df[result_df$individual_data, ]
    ref_line <- geom_segment(data = ref_data,
      aes(x = as.numeric(.data$feature) - 0.35,
          xend = as.numeric(.data$feature) + 0.35,
          y = .data$value, yend = .data$value, group = .data$feature),
      col = "red", size = 1)

    result_df <- result_df[result_df$boxplot_data, ]
    geom <- geom_boxplot(aes(group = .data$feature), fill = "gray", alpha = 0.8,
                         show.legend = FALSE, width = 0.7, outlier.size = 1)
    scale_fill <- NULL
  } else {
    geom <- geom_bar(aes(fill = .data$fill), stat = "identity",
                     show.legend = FALSE)
    scale_fill <- scale_fill_gradient2(low = "blue", mid = "white",
                                       high = "red",
                                       midpoint = 0, limits = c(-1, 1))
  }
  p <- ggplot(result_df, aes(x = .data$feature, y = .data$value)) +
    geom +
    facet +
    geom_hline(yintercept = 0) +
    xlab(x_label) +
    ylab(value_name) +
    x_scale +
    scale_y_continuous(labels = get_format)

  # Add reference datapoint
  if (boxplot && !is.null(data_idx)) {
    p <- p + ref_line
  }
  if (!is.null(scale_fill)) p <- p + scale_fill

  # Remove ticks and labels
  if (!xticks) {
    p <- p + xlab(NULL) +
      theme(axis.ticks.x = element_blank(),
            axis.text.x = element_blank())
  }
  if (!yticks) {
    p <- p + ylab(NULL)
  }

  p
}

#----- Plot function for images -----------------------------------------------
plot_image <- function(result_df, value_name = "value", facet_rows = NULL,
                       facet_cols = NULL, calc_fill = TRUE, xticks = TRUE,
                       yticks = TRUE, legend_labels = NULL, boxplot = FALSE) {

  if (boxplot) {
    facet_rows <- NULL
    value_name <- paste0(value_name, "\n (median)")
  }

  # normalize result for all data points
  if (calc_fill) {
    if (boxplot) {
      result_df$fill <- ave(result_df$value,
                            result_df$boxplot_data,
                            as.character(result_df$output_node),
                            as.character(result_df$feature),
                            as.character(result_df$feature_2),
                            FUN = median)
      max_median <- max(abs(result_df$fill[result_df$boxplot_data]))
      result_df$fill <- if (max_median == 0) 0 else result_df$fill / max_median
    } else {
        group_max <- ave(result_df$value,
                         as.character(result_df$data),
                         as.character(result_df$output_node),
                         FUN = function(x) max(abs(x)))
        result_df$fill <- ifelse(group_max == 0, 0, result_df$value / group_max)
    }
  }

  # Define facets
  facet_rows <- if (is.null(facet_rows)) NULL else vars(.data[[facet_rows]])
  facet_cols <- if (is.null(facet_cols)) NULL else vars(.data[[facet_cols]])
  facet <- facet_grid(cols = facet_cols, rows = facet_rows, scales = "free")

  # Make axis continuous
  result_df$feature <- as.numeric(factor(result_df$feature,
                                         levels = unique(result_df$feature)))
  result_df$feature_2 <- as.numeric(
    factor(result_df$feature_2, levels = unique(result_df$feature_2)))

  # Get legend limits
  if (is.null(legend_labels)) {
    legend_labels <- c("<0", "0", ">0")
  }

  # Create plot/boxplot
  max_value <- max(result_df$fill)
  min_value <- min(result_df$fill)

  if (min_value >= 0) {
    breaks <- c(0, 1)
    legend_labels <- legend_labels[-1]
    limits <- c(0, 1)
  } else if (max_value <= 0) {
    breaks <- c(-1, 0)
    legend_labels <- legend_labels[-3]
    limits <- c(-1, 0)
  } else {
    breaks <- c(-1, 0, 1)
    limits <- c(-1, 1)
  }
  p <- ggplot(result_df, aes(x = .data$feature_2, y = .data$feature)) +
    geom_raster(aes(fill = .data$fill)) +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                         midpoint = 0,
                         breaks = breaks,
                         limits = limits,
                         labels = legend_labels) +
    facet +
    xlab("Image Width") +
    labs(fill = value_name) +
    ylab("Image Height") +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0))

  # Remove ticks and labels
  if (!xticks) {
    p <- p + xlab(NULL) +
      theme(axis.ticks.x = element_blank(),
            axis.text.x = element_blank())
  }
  if (!yticks) {
    p <- p + ylab(NULL) +
      theme(axis.ticks.y = element_blank(),
            axis.text.y = element_blank())
  }

  p
}

#----- Plot function for multimodal data --------------------------------------
plot_extended <- function(result_df, value_name, include_data, boxplot,
                          data_idx = NULL) {
  # Load required packages
  for (pkg in c("grid", "gtable", "gridExtra")) {
    if (!requireNamespace(pkg, quietly = FALSE)) {
      stop(
        "Please install the '", pkg, "' package if you want to create an ",
        "plot for multiple input layers."
      )
    }
  }

  # Combine output node with output layer
  result_df$output_node <- paste(as.character(result_df$model_output),
                                 as.character(result_df$output_node),
                                 sep = ": ")

  # Get names of all output nodes, data points and input layers
  level_outnodes <- as.character(unique(result_df$output_node))
  level_inputs <- unique(result_df$model_input)
  if (boxplot) {
    level_data <- "summarized"
    result_df$data <- "summarized"
  } else {
    level_data <- as.character(levels(result_df$data))
  }

  # We create for each combination of output nodes, data point and input
  # layer the corresponding plot and store these in 'grobs'
  grobs <- array(list(),
                 dim = c(length(level_data), length(level_inputs),
                         length(level_outnodes)))

  for (i in seq_along(level_outnodes)) {
    for (j in seq_along(level_data)) {
      # Create temporary dataset and create 'fill' value
      temp_df <- result_df[result_df$data == level_data[j] &
                             result_df$output_node == level_outnodes[i], ]
      if (boxplot) {
        temp_df$fill <- ave(temp_df$value,
                            temp_df$boxplot_data,
                            as.character(temp_df$output_node),
                            as.character(temp_df$feature),
                            as.character(temp_df$feature_2),
                            FUN = median)
        max_value <- max(temp_df$fill[temp_df$boxplot_data])
        min_value <- min(temp_df$fill[temp_df$boxplot_data])
      } else {
        max_value <- max(temp_df$value)
        min_value <- min(temp_df$value)
        temp_df$fill <- temp_df$value / max(abs(max_value), abs(min_value))
      }

      for (k in seq_along(level_inputs)) {
        # Get the data
        data <- temp_df[temp_df$model_input == level_inputs[k], ]
        # Get facet vars
        facets <- get_facets(i, j, k, length(level_outnodes),
                             length(level_data),
                             length(level_inputs), include_data)
        # Get labels
        labels <- get_labels(i, j, k, length(level_outnodes),
                             length(level_data),
                             length(level_inputs))

        # Create the plot
        if (unique(data$input_dimension) == 3) {
          p <- plot_image(data, value_name,
                          facet_rows = facets$facet_rows,
                          facet_cols = facets$facet_cols,
                          calc_fill = FALSE,
                          xticks = labels$xticks,
                          yticks = labels$yticks,
                          legend_labels =
                            signif(c(min_value, 0, max_value), 2),
                          boxplot = boxplot)
        } else {
          p <- plot_bar(data, value_name,
                        facet_rows = facets$facet_rows,
                        facet_cols = facets$facet_cols,
                        calc_fill = FALSE,
                        xticks = labels$xticks,
                        yticks = labels$yticks,
                        boxplot = boxplot,
                        data_idx = data_idx)
        }

        grobs[j, k, i] <- list(p)
      }
    }
  }

  # Convert grobs to matrix
  dim(grobs) <- c(dim(grobs)[1], prod(dim(grobs)[-1]))


  # Render strips for output
  output_strips <- list(
    labels = data.frame(output_node = level_outnodes),
    theme = theme_gray()
  )

  new("innsight_ggplot2",
      grobs = grobs,
      multiplot = TRUE,
      output_strips = output_strips,
      col_dims = lapply(level_outnodes, function(x) length(level_inputs)),
      boxplot = boxplot)
}

###############################################################################
#                             Utility functions
###############################################################################
get_format <- function(x) {
  x_labels <- as.character(x)
  x_labels[is.na(x)] <- ""

  x_abs <- abs(x)
  x_labels[!is.na(x) & x_abs <= 1e-3] <-
    format(x[!is.na(x) & x_abs <= 1e-3],
           scientific = TRUE, digits = 2, width = 8)
  x_labels[!is.na(x) & x_abs == 0] <-
    format(0, scientific = FALSE, digits = 1, width = 8)
  x_labels[!is.na(x) & x_abs >= 1e3] <-
    format(x[!is.na(x) & x_abs >= 1e3],
           scientific = TRUE, digits = 2, width = 8)
  x_labels[!is.na(x) & x_abs > 1e-3 & x_abs < 10] <-
    format(round(x[!is.na(x) & x_abs > 1e-3 & x_abs < 10], digits = 3),
           scientific = FALSE, nsmall = 3, width = 8)
  x_labels[!is.na(x) & x_abs >= 10 & x_abs < 100] <-
    format(round(x[!is.na(x) & x_abs >= 10 & x_abs < 100], digits = 2),
           scientific = FALSE, nsmall = 2, width = 8)
  x_labels[!is.na(x) & x_abs >= 100 & x_abs < 1e3] <-
    format(round(x[!is.na(x) & x_abs >= 100 & x_abs < 1e3], digits = 1),
           scientific = FALSE, nsmall = 1, width = 8)

  x_labels
}

# i: Output node
# j: data point
# k: input layer
get_facets <- function(i, j, k, i_total, j_total, k_total, include_data) {
  facet_cols <- NULL
  facet_rows <- NULL

  # first datapoint, last output node and last input layer
  # grob top right
  if (j == 1 && i == i_total && k == k_total) {
    facet_cols <- "model_input"
    facet_rows <- "data"
  } else if (j == 1) {
    # first datapoint and all input and all output nodes
    # other grobs in the top row
    facet_cols <- "model_input"
  } else if (i == i_total && k == k_total) {
    # last output node, all datapoints and last input
    # other grobs in the last column
   facet_rows <- "data"
  }

  if (!include_data) {
    facet_rows <- NULL
  }

  list(facet_cols = facet_cols, facet_rows = facet_rows)
}

get_labels <- function(i, j, k, i_total, j_total, k_total) {
  xticks <- TRUE
  yticks <- TRUE

  # not last datapoint, remove x ticks and labels
  if (j != j_total) {
    xticks <- FALSE
  }
  # not first input and not first output layer, remove y ticks and labels
  if (i != 1 || k != 1) {
    yticks <- FALSE
  }

  list(xticks = xticks, yticks = yticks)
}
