#' @import ggplot2


plot_func <- function(result_df, value_name = "Relevance", as_plotly = FALSE) {

  num_inputs <- length(unique(result_df$model_input))
  num_outputs <- length(unique(result_df$model_output))
  num_classes <- length(unique(result_df$output_node))

  # regular plots for neural networks with one input and one output layer
  if (num_inputs == 1 & num_outputs == 1) {
    if (all(result_df$input_dimension == 3)) {
      p <- plot_image(result_df, value_name)
    } else if (all(result_df$input_dimension %in% c(1,2))) {
      p <- plot_bar(result_df, value_name)
    } else {
      stop("Error")
    }
  }
  # This is for models with multiple input and/or output layers
  # This is more of a hacked solution
  else {
    warning("Plotting multiple inputs is currently still a big work in progress. ",
            "For this reason there are still some bugs to be fixed and the ",
            "output might change a lot in the future.")
    p <- plot_extended(result_df, value_name)
  }

  p <- p +
    theme(
      strip.text.x = element_text(size = 10),
      strip.text.y = element_text(size = 10),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12)
    )

  if (as_plotly) {
    if (!requireNamespace("plotly", quietly = FALSE)) {
      stop("Please install the 'plotly' package if you want to create",
           "an interactive plot.")
    }
    p <-
      plotly::ggplotly(p, tooltip = "text", dynamicTicks = FALSE)
  }

  p
}

plot_bar <- function(result_df, value_name) {
  # normalize result for all data points
  result_df$fill <- result_df$value /
    ave(result_df$value, result_df$data, result_df$output_node,
        FUN = function(x) max(abs(x)))

  if (all(result_df$input_dimension == 2)) {
    x_label <- "Signal Length"
    hovertext <- get_hovertext(result_df, value_name, 2)
    result_df$feature <- as.numeric(result_df$feature)
    x_scale <- scale_x_continuous(expand = c(0,0))
  } else {
    x_label <- "Feature"
    hovertext <-  get_hovertext(result_df, value_name, 1)
    x_scale <- scale_x_discrete(
      guide = guide_axis(check.overlap = TRUE)
    )
  }

  facet <- facet_grid(data ~ output_node, scales = "free_y")

  # For a custom hovertext in the plotly-plot, we have to define the
  # aesthetic "text" what results in a warning "Unknown aesthetics". But as
  # you can see in the documentation of plotly::ggplotly, this is the way
  # to go.
  suppressWarnings(
    p <- ggplot(data = result_df) +
      geom_bar(aes(x = .data$feature, y = .data$value, fill = .data$fill,
                   text = hovertext), stat = "identity", show.legend = FALSE) +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                           breaks = c(min(result_df$fill), 0, max(result_df$fill)),
                           labels = c(signif(min(result_df$value), 2), 0,
                                      signif(max(result_df$value), 2))) +
      facet +
      geom_hline(yintercept = 0) +
      xlab(x_label) +
      ylab(value_name) +
      x_scale
  )
  p
}

plot_image <- function(result_df, value_name) {
  # normalize result for all data points
  result_df$fill <- result_df$value /
    ave(result_df$value, result_df$data, result_df$output_node,
        FUN = function(x) max(abs(x)))

  # Define hovertext for plotly
  hovertext <-  get_hovertext(result_df, value_name, 3)
  facet <- facet_grid(data ~ output_node, scales = "free")

  # Make axis continuous
  result_df$feature <- as.integer(result_df$feature)
  result_df$feature_2 <- as.integer(result_df$feature_2)

  suppressWarnings(
    p <-
      ggplot(data = result_df) +
      geom_raster(aes(
        x = .data$feature_2, y = .data$feature, fill = .data$fill,
        text = hovertext
      )) +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                           breaks = c(min(result_df$fill), 0, max(result_df$fill)),
                           labels=c(signif(min(result_df$value), 2), 0,
                                    signif(max(result_df$value), 2))) +
      facet +
      xlab("Image Width") +
      labs(fill = value_name) +
      ylab("Image Height") +
      theme(legend.title = element_text(size = 11)) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0))
  )
  p
}

plot_extended <- function(result_df, value_name) {
  # Get all hovertexts
  hover_1 <- get_hovertext(result_df[result_df$input_dimension == 1,],
                           value_name, 1)
  hover_2 <- get_hovertext(result_df[result_df$input_dimension == 2,],
                           value_name, 2)
  hover_3 <- get_hovertext(result_df[result_df$input_dimension == 3,],
                           value_name, 3)

  result_df$feature <- as.character(result_df$feature)

  # Make feature_2 numeric
  result_df$feature_2 <-
    as.numeric(ave(result_df$feature_2, result_df$model_input, FUN = function(x) {
      if (all(x == "NaN")) {
        res <- rep(0, length(x))
      } else {
        res <- (as.numeric(substring(x,2)) - 1 )
        res <- res / max(res)
      }
      res
    }))

  # Calculate min and max value of all panels with continuous scale
  res_temp <- result_df[result_df$input_dimension != 3,]
  min_and_max <- aggregate(res_temp$value,
                           by = list(res_temp$output_node, res_temp$model_output),
                           FUN = min)
  min_and_max$max <- aggregate(res_temp$value,
                               by = list(res_temp$output_node, res_temp$model_output),
                               FUN = max)$x
  for (i in seq_len(nrow(min_and_max))) {
    row <- min_and_max[i,]
    result_df$feature_2[result_df$output_node == row[[1]] &
                          result_df$model_output == row[[2]]] <-
      result_df$feature_2[result_df$output_node == row[[1]] &
                            result_df$model_output == row[[2]]] *
      (row[[4]][[1]] - row[[3]][[1]])*1.5 + row[[3]][[1]] * 1.5
  }

  if (length(unique(result_df$data)) == 1) {
    # normalize result for all data points, and outputs
    result_df$fill <-  result_df$value /
      ave(result_df$value, result_df$output_node, result_df$model_output,
          FUN = function(x) max(abs(x)))

    facet <- facet_grid(model_output + output_node ~ model_input,
                        scales = "free")

  } else {
    result_df$fill <-  result_df$value /
      ave(result_df$value, result_df$data, FUN = function(x) max(abs(x)))

    facet <- facet_grid(model_output + output_node ~ data + model_input,
                        scales = "free")
  }

  suppressWarnings(
    p <- ggplot() +
      geom_raster(data = result_df[result_df$input_dimension == 3,],
                  mapping = aes(x = .data$feature,
                                y = .data$feature_2,
                                fill = .data$fill,
                                text = hover_3)) +
      geom_bar(data = result_df[result_df$input_dimension == 2,],
               mapping = aes(x = .data$feature,
                             y = .data$value,
                             fill = .data$fill,
                             text = hover_2),
               stat = "identity") +
      geom_bar(data = result_df[result_df$input_dimension == 1,],
               mapping = aes(x = .data$feature,
                             y = .data$value,
                             fill = .data$fill,
                             text = hover_1),
               stat = "identity") +
      facet +
      geom_hline(data = result_df[result_df$input_dimension != 3,],
                 aes(yintercept = 0), show.legend = FALSE) +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                           breaks = c(min(result_df$fill), 0, max(result_df$fill)),
                           labels=c(signif(min(result_df$value), 2), 0,
                                    signif(max(result_df$value), 2))) +
      scale_y_continuous(expand = c(0,0)) +
      #scale_x_discrete(breaks = unique(result_df$feature)) +
      xlab("") + ylab(value_name) + labs(fill = value_name)
  )

  p
}



#
#       Utils
#
get_hovertext <- function(result_df, value_name, dim) {
  if (dim == 1) {
    hovertext <- paste(
      "<b></br>", value_name, ":", signif(result_df$value), "</b>\n",
      "</br> Datapoint: ", result_df$data,
      "</br> Output:      ", result_df$output_node,
      "</br> Feature:    ", result_df$feature
    )
  } else if (dim == 2) {
    hovertext <- paste(
      "<b></br>", value_name, ":", signif(result_df$value), "</b>\n",
      "</br> Datapoint: ", result_df$data,
      "</br> Output:      ", result_df$output_node,
      "</br> Length:      ", result_df$feature
    )
  } else {
    text <- paste(
      "<b></br>", value_name, ":", signif(result_df$value), "</b>\n",
      "</br> Datapoint: ", result_df$data,
      "</br> Output:      ", result_df$output_node,
      "</br> Height:      ", result_df$feature_2,
      "</br> Width:       ", result_df$feature
    )
  }
}
