#' @import ggplot2
#' @importFrom stats ave
#' @importFrom stats aggregate
#' @importFrom stats median
#' @importFrom grDevices rgb
#' @importFrom grDevices colorRamp
#' @importFrom grDevices boxplot.stats
#'

##############################################################################
#                           Individual Plots
##############################################################################

plot_1d_input <- function(result, value_name, data_names, input_names,
                          output_names, channels_first, no_data) {

  # For nice fills, rescale result for each datapoint individually
  result_scaled <-
    result / torch_amax(torch_abs(result), dim = 2:3, keepdim = TRUE)
  # Store the number of datapoints
  num_data <- dim(result)[1]
  # Transform the result in a data.frame and add new column with the
  # rescaled values
  result <- get_dataframe(
    result, data_names, input_names, output_names,
    channels_first
  )
  result$value_scaled <- as.vector(as_array(result_scaled))

  result$x <- as.numeric(result$feature)
  result[[value_name]] <- round2(result$value)

  # Define the hovertext for plotly
  # If 'no_data = TRUE', there will be no datapoint in the hovertext
  if (no_data) {
    text <- paste(
      "<b></br>", value_name, ":", result[[value_name]], "</b>\n",
      "</br> Class:       ", result$class,
      "</br> Feature:    ", result$feature
    )
  } else {
    text <- paste(
      "<b></br>", value_name, ":", result[[value_name]], "</b>\n",
      "</br> Datapoint: ", result$data,
      "</br> Class:       ", result$class,
      "</br> Feature:    ", result$feature
    )
  }
  # Create facets
  if (num_data == 1) {
    facet <- facet_grid(cols = vars(class), scales = "free_y")
  } else {
    facet <- facet_grid(data ~ class, scales = "free_y")
  }
  # For a custom hovertext in the plotly-plot, we have to define the
  # aesthetic "text" what results in a warning "Unknown aesthetics". But as
  # you can see in the documentation of plotly::ggplotly, this is the way
  # to go.
  p <- ggplot(data = result) +
    suppressWarnings(
      geom_rect(aes(
        xmin = .data$x - 0.3,
        xmax = .data$x + 0.3,
        ymin = 0,
        ymax = .data$value,
        fill = .data$value_scaled,
        text = text
      ),
      show.legend = FALSE
      )
    ) +
    scale_fill_gradient2(low = "blue", mid = "gray", high = "red") +
    facet +
    scale_x_discrete(
      limits = levels(result$feature),
      guide = guide_axis(check.overlap = TRUE)
    ) +
    geom_hline(yintercept = 0) +
    xlab("Feature") +
    ylab(value_name)

  p
}

# --------------------------- 2D Plots ----------------------------------
plot_2d_input <- function(result, value_name, data_names, input_names,
                          output_names, channels_first, no_data) {

  # For nice fills, rescale result for each datapoint individually
  result_scaled <-
    result / torch_amax(torch_abs(result), dim = 2:4, keepdim = TRUE)
  # Store the number of datapoints
  num_data <- dim(result)[1]
  # Transform the result in a data.frame and add new column with the
  # rescaled values
  result <- get_dataframe(
    result, data_names, input_names, output_names,
    channels_first
  )
  result$value_scaled <- as.vector(as_array(result_scaled))

  result$x <- as.numeric(result$feature_l)
  result[[value_name]] <- round2(result$value)

  # Define the hovertext for plotly
  # If 'no_data = TRUE', there will be no datapoint in the hovertext
  if (no_data) {
    text <- paste(
      "<b></br>", value_name, ":", result[[value_name]], "</b>\n",
      "</br> Class:       ", result$class,
      "</br> Length:     ", result$feature_l
    )
  } else {
    text <- paste(
      "<b></br>", value_name, ":", result[[value_name]], "</b>\n",
      "</br> Datapoint: ", result$data,
      "</br> Class:       ", result$class,
      "</br> Length:     ", result$feature_l
    )
  }

  # Create facets
  if (num_data == 1) {
    facet <- facet_grid(cols = vars(class), scales = "free_y")
  } else {
    facet <- facet_grid(data ~ class, scales = "free_y")
  }

  # Make signal length continuous
  result$feature_l <- as.numeric(result$feature_l)

  # For a custom hovertext in the plotly-plot, we have to define the
  # aesthetic "text" what results in a warning "Unknown aesthetics". But as
  # you can see in the documentation of plotly::ggplotly, this is the way
  # to go.
  suppressWarnings(
    p <- ggplot(data = result) +
      geom_rect(aes(
        xmin = .data$x - 0.5,
        xmax = .data$x + 0.5,
        ymin = 0,
        ymax = .data$value,
        fill = .data$value_scaled,
        text = text
      ),
      show.legend = FALSE
      ) +
      scale_fill_gradient2(low = "blue", mid = "grey", high = "red") +
      facet +
      geom_hline(yintercept = 0) +
      xlab("Signal Length") +
      ylab(value_name)
  )
  p
}

# ------------------------- 3D Plots -------------------------------------
plot_3d_input <- function(result, value_name, data_names, input_names,
                          output_names, channels_first, no_data) {

  # For nice fills, rescale result for each datapoint individually
  result_scaled <-
    result / torch_amax(torch_abs(result), dim = 2:5, keepdim = TRUE)
  num_data <- dim(result)[1]
  # Transform the result in a data.frame and add new column with the
  # rescaled values
  result <- get_dataframe(
    result, data_names, input_names, output_names,
    channels_first
  )
  result$value_scaled <- as.vector(as_array(result_scaled))
  result[[value_name]] <- round2(result$value)

  # Define the hovertext for plotly
  # If 'no_data = TRUE', there will be no datapoint in the hovertext
  if (no_data) {
    text <- paste(
      "<b></br>", value_name, ":", result[[value_name]], "</b>\n",
      "</br> Class:       ", result$class,
      "</br> Height:      ", result$feature_h,
      "</br> Width:       ", result$feature_w
    )
  } else {
    text <- paste(
      "<b></br>", value_name, ":", result[[value_name]], "</b>\n",
      "</br> Datapoint: ", result$data,
      "</br> Class:       ", result$class,
      "</br> Height:      ", result$feature_h,
      "</br> Width:       ", result$feature_w
    )
  }

  # Create facets
  if (num_data == 1) {
    facet <- facet_grid(cols = vars(class), scales = "free_y")
  } else {
    facet <- facet_grid(data ~ class, scales = "free_y")
  }

  # Define color
  if (min(result$value_scaled) >= 0) {
    col <- colorRamp(c("white", "red"))
    colors <- seq(0, 1, length.out = 50)^0.8
    value_max <- max(result$value_scaled)
    value_min <- 0
  } else if (max(result$value_scaled) <= 0) {
    col <- colorRamp(c("blue", "white"))
    colors <- seq(0, 1, length.out = 50)^0.8
    value_max <- 0
    value_min <- min(result$value_scaled)
  } else {
    col <- colorRamp(c("blue", "white", "red"))
    colors <- seq(-1, 1, length.out = 50)
    colors <- abs(colors)^0.8 * sign(colors) * 0.5 + 0.5
    value_max <- max(abs(result$value_scaled))
    value_min <- -value_max
  }


  # Make axis continuous
  result$feature_h <- as.numeric(result$feature_h)
  result$feature_w <- as.numeric(result$feature_w)
  suppressWarnings(
    p <-
      ggplot(data = result) +
      geom_raster(aes(
        x = .data$feature_w, y = .data$feature_h, fill = .data$value_scaled,
        text = text
      )) +
      scale_fill_gradientn(
        colours = rgb(col(colors) / 255),
        limits = c(value_min, value_max),
        labels = c("<0","0", ">0"),
        n.breaks = 3
      ) +
      coord_cartesian(
        xlim = c(2, max(result$feature_w) - 1),
        ylim = c(2, max(result$feature_h) - 1)
      ) +
      facet +
      xlab("Image Width") +
      labs(fill = value_name) +
      ylab("Image Height") +
      theme(legend.title = element_text(size = 11))
  )
  p
}

##############################################################################
#                             Boxplot Plots
##############################################################################

#
# ggplot2
#
boxplot_ggplot <- function(result, aggr_channels, ref_datapoint, value_name,
                           boxplot_data, classes, input_names, output_names,
                           preprocess_FUN, channels_first) {
  # Get the number of input dimension of the result
  num_dims <- length(dim(result)) - 2

  # Plot the result stratified by the input dimension

  #
  # 1D Input
  #
  if (num_dims == 1) {
    # Filter the result for datapoints 'boxplot_data' and class 'classes'
    # In addition, apply the preprocess function 'preprocess_FUN'
    res <- preprocess_FUN(result[boxplot_data, , classes, drop = FALSE])
    # If a reference datapoint should be plotted, filter for this specific
    # datapoint and create a data.frame from it
    if (length(ref_datapoint) > 0) {
      res_ref <- preprocess_FUN(result[ref_datapoint, , classes, drop = FALSE])
      res_ref <- get_dataframe(
        res_ref, paste0("data_", ref_datapoint),
        input_names, output_names, channels_first
      )
    } else {
      res_ref <- NULL
    }

    # Get a data.frame with the boxplot information and another one
    # for the outliers
    res <- get_boxplot_df(res, input_names[[1]], output_names)

    # Plot the result
    p <- boxplot_1d_2d_ggplot(
      res[[1]], res[[2]], res_ref, value_name,
      "Feature", 0.8
    )
  }
  #
  # 2D Input
  #
  else if (num_dims == 2) {
    # Filter the result for datapoints 'boxplot_data' and class 'classes'
    # In addition, apply the preprocess function 'preprocess_FUN'
    res <-
      as_array(preprocess_FUN(result[boxplot_data, , , classes, drop = FALSE]))

    # Aggregate the channels with function 'aggr_channels'
    if (channels_first) {
      dims <- c(1, 3, 4)
    } else {
      dims <- c(1, 2, 4)
    }
    res <- torch_tensor(apply(res, dims, aggr_channels))

    # If a reference datapoint should be plotted, filter for this specific
    # datapoint, aggregate channels and then create a data.frame from it
    if (length(ref_datapoint) > 0) {
      res_ref <- as_array(
        preprocess_FUN(result[ref_datapoint, , , classes, drop = FALSE])
      )
      res_ref <- apply(res_ref, dims, aggr_channels)
      res_ref <- get_dataframe(
        res_ref, paste0("data_", ref_datapoint),
        input_names[2], output_names, channels_first
      )
    } else {
      res_ref <- NULL
    }

    # Get a data.frame with the boxplot information and another one
    # for the outliers
    res <- get_boxplot_df(res, input_names[[2]], output_names)

    # Plot the result
    p <- boxplot_1d_2d_ggplot(
      res[[1]], res[[2]], res_ref, value_name,
      "Signal Length", 1
    )
  }
  #
  # 3D Input
  #
  else if (num_dims == 3) {
    # Filter the result for datapoints 'boxplot_data' and class 'classes'
    # In addition, apply the preprocess function 'preprocess_FUN'
    res <- preprocess_FUN(
      as_array(result[boxplot_data, , , , classes, drop = FALSE])
    )
    # Aggregate the channels with function 'aggr_channels'
    if (channels_first) {
      dims <- c(1, 3, 4, 5)
      d <- 2
    } else {
      dims <- c(1, 2, 3, 5)
      d <- 4
    }
    res <- torch_tensor(apply(res, dims, aggr_channels))$unsqueeze(d)

    # Calculate the median value
    res <- torch_median(res, dim = 1, keepdim = TRUE)[[1]]
    input_names[[1]] <- c("aggr")

    # Get a data.frame for the median value
    res <- get_dataframe(
      res, list("data_1"), input_names, output_names,
      channels_first
    )

    # Plot the result
    p <- boxplot_3d_ggplot(res, value_name)
  }
  p
}


boxplot_1d_2d_ggplot <- function(res, outliers, res_ref, value_name, xlabel,
                                 width) {
  # Create boxplots without outliers
  breaks <- levels(res$feature)
  if (length(breaks) > 10) {
    breaks <- breaks[seq(1, length(breaks), by = length(breaks) %/% 10)]
  }
  mapping <- aes(
    x = .data$feature, ymin = .data$min,
    lower = .data$lower, middle = .data$median,
    upper = .data$upper, ymax = .data$max
  )
  p <- ggplot() +
    geom_errorbar(data = res, aes(
      x = .data$feature, ymin = .data$min,
      ymax = .data$max
    ), width = 0.8) +
    geom_boxplot(
      data = res, mapping = mapping, color = "black",
      fill = "grey40", width = width, stat = "identity"
    ) +
    facet_grid(cols = vars(class), scales = "free_y") +
    scale_x_discrete(
      breaks = breaks,
      guide = guide_axis(check.overlap = TRUE)
    ) +
    geom_hline(yintercept = 0) +
    xlab(xlabel) +
    labs(fill = value_name) +
    ylab(value_name) +
    theme(
      strip.text.y = element_text(size = 8),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12)
    )

  # Add outliers
  if (!is.null(outliers)) {
    p <- p +
      geom_point(data = outliers, aes(x = .data$feature, y = .data$value))
  }

  # Add reference datapoint
  if (!is.null(res_ref)) {
    one_data <- res_ref[rep(seq_len(nrow(res_ref)), each = 2), ]
    one_data$x <- as.numeric(one_data$feature) + c(-width / 2, width / 2)

    p <- p +
      geom_line(
        data = one_data,
        aes(x = .data$x, y = .data$value, group = .data$feature),
        color = "red",
        size = 1
      )
  }

  p
}


boxplot_3d_ggplot <- function(result, value_name) {

  # Set the colorbar for the plot
  if (min(result$value) >= 0) {
    col <- colorRamp(c("black", "red"))
    value_max <- max(result$value)
    value_min <- 0
  } else if (max(result$value) <= 0) {
    col <- colorRamp(c("blue", "black"))
    value_max <- 0
    value_min <- min(result$value)
  } else {
    col <- colorRamp(c("blue", "black", "red"))
    value_max <- max(abs(result$value))
    value_min <- -value_max
  }

  breaks_w <- levels(result$feature_w)
  if (length(breaks_w) > 10) {
    breaks_w <-
      breaks_w[seq(1, length(breaks_w), by = length(breaks_w) %/% 10)]
  }
  breaks_h <- levels(result$feature_h)
  if (length(breaks_h) > 10) {
    breaks_h <-
      breaks_h[seq(1, length(breaks_h), by = length(breaks_h) %/% 10)]
  }

  p <-
    ggplot(data = result) +
    geom_raster(
      aes(x = .data$feature_w, y = .data$feature_h, fill = .data$value)
    ) +
    scale_fill_gradientn(
      colors = rgb(col(seq(0, 1, length.out = 50)) / 255),
      limits = c(value_min, value_max)
    ) +
    facet_grid(cols = vars(class), scales = "free_y") +
    scale_x_discrete(
      breaks = breaks_w,
      guide = guide_axis(check.overlap = TRUE)
    ) +
    scale_y_discrete(
      breaks = breaks_h,
      guide = guide_axis(check.overlap = TRUE)
    ) +
    xlab("Image Width") +
    labs(fill = value_name) +
    ylab("Image Height") +
    theme(
      strip.text.y = element_text(size = 8),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      legend.title = element_text(size = 11)
    )

  p
}


#
# plotly
#

boxplot_plotly <- function(result, aggr_channels, ref_datapoint, value_name) {
  if (!requireNamespace("plotly", quietly = FALSE)) {
    stop("Please install the 'plotly' package if you want to create an
         interactive plot.")
  }

  # Set the reference datapoint to "None" if ref_datapoint = NULL
  if (is.null(ref_datapoint)) {
    ref_datapoint <- 0
  }

  # Get the input dimension of the result
  input_dim <- ncol(result) - 5

  # Plot the result stratified by the input dimension

  #
  # 1D Input
  #
  if (input_dim == 1) {
    # 1D Input has no channel, hence we add one for the plot function
    result$channel <- "C1"
    p <- boxplot_1d_2d_plotly(result, aggr_channels, ref_datapoint, value_name,
      true_channel = "C1",
      channels_list = c("C1"),
      channel_text = FALSE,
      feature_name = "Feature:  ",
      xtitle = "Feature",
      width = 0.8
    )
  }
  #
  # 2D Input
  #
  else if (input_dim == 2) {
    channel_list <- levels(result$channel)
    true_channel <- "aggr"
    channels_list <- c(as.character(channel_list), "aggr")
    names(result)[which(names(result) == "feature_l")] <- "feature"
    p <- boxplot_1d_2d_plotly(result, aggr_channels, ref_datapoint, value_name,
      true_channel = true_channel,
      channels_list = channels_list,
      channel_text = TRUE,
      feature_name = "Length:   ",
      xtitle = "Signal Length",
      width = 1
    )
  }
  #
  # 3D Input
  #
  else {
    channel_list <- as.character(unique(result$channel))
    ref_channel <- "aggr"

    channel_list <- c(channel_list, "aggr")
    p <- boxplot_3d_plotly(
      result, aggr_channels, ref_channel, value_name, channel_list
    )
  }
}

boxplot_1d_2d_plotly <- function(result, aggr_channels, ref_datapoint,
                                 value_name, true_channel, channels_list,
                                 channel_text, feature_name, xtitle, width) {
  if (ref_datapoint == 0) {
    ref_data_name <- "None"
  } else {
    ref_data_name <- paste("data_", ref_datapoint, sep = "")
  }
  subplot_list <- NULL

  # We plot the result for each class in its own figure, separately
  for (class in unique(result$class)) {
    # Filter for the class
    df_class <- result[result$class == class, ]
    fig <- plotly::plot_ly(showlegend = FALSE)

    # We plot the result for each channel plus the aggregated channels in one
    # figure. But with use of the argument "visible", only the plot for the
    # chosen channel "true_channel" is visible
    for (channel in channels_list) {

      # Set the argument "visible" for this channel
      if (channel == true_channel) {
        visible <- TRUE
      } else {
        visible <- FALSE
      }

      # Create the dateframe for the boxplot plot and for the individual
      # results
      if (channel == "aggr") {
        df <- aggregate(list(value = df_class$value),
          by = list(
            data = df_class$data,
            feature = df_class$feature,
            class = df_class$class,
            summary_data = df_class$summary_data,
            individual_data = df_class$individual_data
          ), FUN = aggr_channels
        )
        df <- df[order(df$feature, df$data), ]
        df$channel <- "aggr"
        individual_df <- df[df$individual_data, -c(4, 5)]
        df <- df[df$summary_data, -c(4, 5)]
      } else {
        df <- df_class[df_class$channel == channel &
          df_class$summary_data, -c(6, 7)]
        individual_df <- df_class[df_class$channel == channel &
          df_class$individual_data, -c(6, 7)]
      }

      # Set the data list for the individual plots
      data_list <- as.character(unique(individual_df$data))

      # Calculate the boxplot information
      stats <- aggregate(list(value = df$value),
        by = list(
          feature = df$feature,
          class = df$class
        ),
        FUN = function(x) boxplot.stats(x)$stats
      )

      stats_df <-
        stats[rep(seq_len(nrow(stats)), each = length(unique(df$data))), ]
      outliers <-
        df[df$value < stats_df$value[, 1] | df$value > stats_df$value[, 5], ]
      outliers$x <- as.numeric(outliers$feature)

      df_boxplot <-
        data.frame(
          feature = stats$feature, class = stats$class,
          ymin = stats$value[, 1], lower = stats$value[, 2],
          middle = stats$value[, 3], upper = stats$value[, 4],
          ymax = stats$value[, 5]
        )
      df_boxplot$x <- as.numeric(df_boxplot$feature)

      # Plot the outliers, if there are some
      if (nrow(outliers) > 0) {
        fig <- plotly::add_markers(fig,
          data = outliers, visible = visible, x = ~x, y = ~value,
          color = I("black"), hoverinfo = "text", name = channel,
          text = paste(
            "<b> Outlier </b>",
            make_hovertext(
              outliers$data, outliers$feature, feature_name,
              outliers$class, outliers$value, value_name,
              outliers$channel, channel_text
            )
          )
        )
      }

      # Add the boxplot without the outliers and the default hovertext of
      # the box to the figure
      fig <- plotly::add_trace(fig,
        type = "box", visible = visible, name = channel, color = I("black"),
        data = plotly::group_by(df_boxplot, df_boxplot$feature),
        q1 = ~lower, q3 = ~upper, median = ~middle, lowerfence = ~ymin,
        upperfence = ~ymax, x0 = 1, dx = 1, width = width, hoverinfo = "y"
      )

      # Plot some individual results in red.
      for (i in unique(individual_df$data)) {
        one_data_orig <- individual_df[individual_df$data == i, ]
        one_data <-
          one_data_orig[rep(seq_len(nrow(one_data_orig)), each = 2), ]
        one_data$x <- as.numeric(one_data$feature) + c(-width / 2, width / 2)

        # Only the individual result for the reference data_id and for the
        # true channel should be visible
        if (i == ref_data_name && channel == true_channel) {
          visible <- TRUE
        } else {
          visible <- FALSE
        }

        # Add the individual results to the figure
        fig <- plotly::add_lines(fig,
          x = ~x, y = ~value, y0 = ~value, color = I("red"),
          data = plotly::group_by(one_data, one_data$feature),
          visible = visible, hoverinfo = "text", line = list(width = 3),
          name = paste(channel, i), x0 = 1, dx = 1,
          text = make_hovertext(
            i, one_data$feature, feature_name, class, one_data$value,
            value_name, one_data$channel, channel_text
          )
        )
      }
    }

    # Add the ggplot2-like facet frame to the figure with the class label
    fig <- make_facet_frame(fig,
      xtitle = xtitle,
      annot_text = class,
      ytitle = value_name,
      xtype = "linear",
      tickvals = seq_len(length(unique(df_class$feature))),
      ticktext = unique(df_class$feature)
    )

    # Add the figure for this class to the subplot list
    subplot_list[[class]] <- fig
  }

  # Combine all plots from the subplot list in one row to one big figure with
  # subplots for each class
  fig <- plotly::subplot(subplot_list, nrows = 1, shareY = TRUE, shareX = TRUE)

  # The names of the traces have either the form "channel" or
  # "channel_datapoint". This allows an easier definition of the
  # slider and buttons.
  fig_names <- sapply(fig$x$data, FUN = function(x) x$name)
  sliders <- list()
  buttons <- list()

  for (datapoint in c("None", data_list)) {
    step <- lapply(
      channels_list,
      FUN = function(n) {
        list(
          method = "restyle", label = n,
          args = list(
            "visible",
            fig_names == paste(n, datapoint) | fig_names == n
          )
        )
      }
    )
    num_true_channel <- which(channels_list == true_channel)
    sliders[[datapoint]] <-
      list(
        active = num_true_channel - 1,
        pad = list(t = 40),
        currentvalue = list(
          prefix = "Channel: ",
          font = list(size = 16)
        ),
        steps = step
      )

    buttons[[datapoint]] <-
      list(
        method = "update",
        label = datapoint,
        args = list(
          list(
            visible =
              sliders[[datapoint]]$steps[[num_true_channel]]$args[[2]]
          ),
          list(sliders = list(sliders[[datapoint]]))
        )
      )
  }
  names(buttons) <- NULL

  # Add the sliders and buttons to the plot and return the resulting figure
  fig <- plotly::layout(fig,
    sliders = list(sliders[[ref_data_name]]),
    updatemenus = list(list(
      y = 1.055,
      x = 0,
      pad = list(r = 15),
      bgcolor = "#F0F0F0FF",
      active = which(data_list == ref_data_name),
      buttons = buttons
    ))
  )
  fig
}


boxplot_3d_plotly <- function(result, aggr_channels, ref_channel, value_name,
                              channel_list) {
  # We don't plot individual results, hence we collect only the summary data
  # from the given data "result"
  result <- result[result$summary_data, -c(7, 8)]

  # Set some constants
  feat_h <- unique(result$feature_h)
  n_row <- length(feat_h)
  feat_w <- unique(result$feature_w)
  n_col <- length(feat_w)
  name <- c("Min", "Lower", "Median", "Upper", "Max")

  # Set the colorbar for the plot
  if (min(result$value) >= 0) {
    col <- colorRamp(c("black", "red"))
    value_max <- max(result$value)
    value_min <- 0
  } else if (max(result$value) <= 0) {
    col <- colorRamp(c("blue", "black"))
    value_max <- 0
    value_min <- min(result$value)
  } else {
    col <- colorRamp(c("blue", "black", "red"))
    value_max <- max(abs(result$value))
    value_min <- -value_max
  }

  # We create a plot for each class separately.
  subplot_list <- NULL
  for (class in unique(result$class)) {

    # We collect all traces to be plotted in the variable "traces"
    traces <- NULL
    df_class <- result[result$class == class, ]
    p <- plotly::plot_ly(showlegend = FALSE)

    # We plot the result for each channel plus the aggregated channels in one
    # figure. But with use of the argument "visible", only the plot for the
    # chosen channel "true_channel" is visible
    for (chan in channel_list) {

      # Create the dateframe for the boxplot plot and for the individual
      # results
      if (chan == "aggr") {
        df <- aggregate(list(value = df_class$value),
          by = list(
            data = df_class$data,
            feature_h = df_class$feature_h,
            feature_w = df_class$feature_w,
            class = df_class$class
          ), FUN = aggr_channels
        )
      } else {
        df <- df_class[df_class$channel == chan, ]
      }
      stats <- aggregate(list(value = df$value),
        by = list(
          feature_h = df$feature_h,
          feature_w = df$feature_w,
          class = df$class
        ),
        FUN = function(x) boxplot.stats(x)$stats
      )
      rounded_stats <- round2(stats$value)

      # We have 5 statistical values: Min, Lower Whisker, Median, Upper Whisker
      # and Max. For each value, we add the required information in the
      # variable traces to plot this trace
      for (i in 1:5) {
        if (i == 3 & chan == ref_channel) {
          visible <- TRUE
        } else {
          visible <- FALSE
        }
        traces[[paste(chan, name[i])]] <-
          list(
            visible = visible, name = paste(chan, name[i]), x = feat_w,
            y = feat_h, z = matrix(stats$value[, i], nrow = n_row),
            text = matrix(
              paste(
                "<b>", name[i], "</b>",
                "</br><b>", rounded_stats[, i], "</b><br>",
                "</br> Max:      ", rounded_stats[, 5],
                "</br> Q3:        ", rounded_stats[, 4],
                "</br> Median: ", rounded_stats[, 3],
                "</br> Q1:        ", rounded_stats[, 2],
                "</br> Min:       ", rounded_stats[, 1],
                "<br>",
                "</br> Height:    ", stats$feature_h,
                "</br> Width:     ", stats$feature_w,
                "</br> Channel: ", chan
              ),
              n_row, n_col
            )
          )
      }
    }

    # Now we can plot the traces with plotly::add_trace as a heatmap
    for (trace in traces) {
      p <- plotly::add_trace(p,
        x = trace$x, y = trace$y, z = trace$z, visible = trace$visible,
        name = trace$name, type = "heatmap", hoverinfo = "text",
        text = trace$text, showlegend = FALSE, zmin = value_min,
        zmax = value_max, colors = col,
        colorbar = list(
          len = 1, x = 1, y = 1, title = list(
            text = value_name,
            font = list(size = 15, face = "bold")
          )
        )
      )
    }

    # Add a ggplot2-like facet frame to the plot
    p <- make_facet_frame(p,
      xtitle = "Width", annot_text = class, ytitle = "Height",
      xtype = "category"
    )
    subplot_list[[class]] <- p
  }

  # Create a bigplot with subplots for each class
  fig <- plotly::subplot(subplot_list, nrows = 1, shareY = TRUE)

  # Create slider for the boxplot values and buttons for the channels
  fig_names <- sapply(fig$x$data, FUN = function(x) x$name)
  sliders <- list()
  buttons <- list()

  for (chan in channel_list) {
    step <- lapply(name,
      FUN = function(n) {
        list(
          method = "restyle",
          label = n,
          args = list("visible", fig_names == paste(chan, n))
        )
      }
    )
    sliders[[chan]] <-
      list(
        active = 2,
        pad = list(t = 40),
        currentvalue = list(
          prefix = "Current: ",
          font = list(size = 16)
        ),
        steps = step
      )
    buttons[[chan]] <-
      list(
        method = "update",
        label = chan,
        args = list(
          list(visible = sliders[[chan]]$steps[[3]]$args[[2]]),
          list(sliders = list(sliders[[chan]]))
        )
      )
  }
  names(buttons) <- NULL

  # Add slider and buttons to the plot
  plotly::layout(fig,
    sliders = list(sliders[[ref_channel]]),
    updatemenus = list(list(
      y = 1.055,
      x = 0,
      pad = list(r = 15),
      bgcolor = "#F0F0F0FF",
      active = which(channel_list == ref_channel) - 1,
      buttons = buttons
    ))
  )
}


##############################################################################
#                                Utils
##############################################################################

make_hovertext <- function(datapoint, feature, feature_name, class, value,
                           value_name, channel, show_channel) {
  hovertext <- paste(
    "<b></br>", datapoint,
    "</br>", paste(value_name, ":", sep = ""), round2(value),
    "</b>\n",
    "</br> Class:     ", class,
    "</br>", feature_name, feature
  )
  if (show_channel) {
    hovertext <- paste(hovertext, "</br> Channel: ", channel)
  }

  hovertext
}

make_facet_frame <- function(fig, annot_text = "", xtitle = "x", ytitle = "y",
                             xtype = "linear", tickvals = NULL,
                             ticktext = NULL) {
  if (xtype == "linear") {
    xaxis <- list(
      tickvals = tickvals,
      ticktext = ticktext,
      title = list(
        text = xtitle,
        standoff = 2
      ),
      titlefont = list(size = 18, face = "bold"),
      gridcolor = "#ffff",
      type = "linear"
    )
  } else {
    xaxis <- list(
      title = list(
        text = xtitle,
        standoff = 2
      ),
      titlefont = list(size = 18, face = "bold")
    )
  }
  fig <- plotly::add_annotations(fig,
    text = annot_text,
    x = 0.5,
    y = 1,
    xanchor = "center",
    yref = "paper",
    xref = "paper",
    showarrow = FALSE,
    yanchor = "bottom",
    font = list(size = 15, face = "bold")
  )
  plotly::layout(fig,
    plot_bgcolor = "rgba(240,240,240,1)",
    yaxis = list(
      title = ytitle,
      titlefont = list(size = 18, face = "bold"),
      hoverformat = ".4f",
      zeroline = TRUE,
      gridcolor = "#ffff"
    ),
    xaxis = xaxis,
    shapes = list(
      type = "rect",
      x0 = 0,
      x1 = 1,
      xref = "paper",
      y0 = 0,
      y1 = 22,
      yanchor = 1,
      yref = "paper",
      ysizemode = "pixel",
      fillcolor = "rgba(204,204,204,1)",
      line = list(color = "transparent")
    )
  )
}

get_dataframe <- function(result, data_names, input_names, output_names,
                          channels_first) {
  result <- as.array(result)

  if (length(input_names) == 1) {
    df <- expand.grid(
      data = data_names,
      feature = input_names[[1]],
      class = output_names
    )
  }
  # input (channels, signal_length)
  else if (length(input_names) == 2) {
    if (channels_first) {
      df <- expand.grid(
        data = data_names,
        channel = input_names[[1]],
        feature_l = input_names[[2]],
        class = output_names
      )
    } else {
      df <- expand.grid(
        data = data_names,
        feature_l = input_names[[2]],
        channel = input_names[[1]],
        class = output_names
      )
    }
  } else if (length(input_names) == 3) {
    if (channels_first) {
      df <- expand.grid(
        data = data_names,
        channel = input_names[[1]],
        feature_h = input_names[[2]],
        feature_w = input_names[[3]],
        class = output_names
      )
    } else {
      df <- expand.grid(
        data = data_names,
        feature_h = input_names[[2]],
        feature_w = input_names[[3]],
        channel = input_names[[1]],
        class = output_names
      )
    }
  }
  df$value <- as.vector(result)
  df
}


round2 <- function(value) {
  #rounded_value <-
  #  round(value, 0) * (abs(value) >= 100) +
  #  round(value, 2) * ((abs(value) < 100) & (abs(value) >= 1)) +
  #  round(value, 4) * ((abs(value) < 1) & (abs(value) >= 0.0001)) +
  rounded_value <- as.numeric(formatC(value))
  if (is.array(value)) {
    rounded_value <- array(rounded_value, dim = dim(value))
  }
  rounded_value

}

get_boxplot_df <- function(res, input_names, output_names) {
  summary_df <- apply(
    as_array(res), c(2, 3),
    function(x) boxplot.stats(x, do.conf = FALSE)$stats
  )
  df <- expand.grid(feature = input_names, class = output_names)
  df$min <- as.vector(summary_df[1, , ])
  df$lower <- as.vector(summary_df[2, , ])
  df$median <- as.vector(summary_df[3, , ])
  df$upper <- as.vector(summary_df[4, , ])
  df$max <- as.vector(summary_df[5, , ])

  out_names <- rep(output_names, each = length(input_names))
  in_names <- rep(input_names, times = length(output_names))
  outliers <- apply(
    as_array(res), 2:3,
    function(x) boxplot.stats(x, do.conf = FALSE)$out
  )
  if (length(outliers) != 0) {
    k <- sapply(outliers, length)
    num <- rep(1:(length(in_names)), k)

    outliers <- data.frame(
      value = unlist(outliers[k >= 1]),
      feature = in_names[num],
      class = out_names[num]
    )
  } else {
    outliers <- NULL
  }

  list(df, outliers)
}
