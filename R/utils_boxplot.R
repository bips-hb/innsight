
boxplot_func <- function(boxplot_df, individual_df, value_name = "Relevance", as_plotly = FALSE) {

  num_inputs <- length(unique(boxplot_df$model_input))
  num_outputs <- length(unique(boxplot_df$model_output))
  num_classes <- length(unique(boxplot_df$output_node))

  # regular plots for neural networks with one input and one output layer
  if (num_inputs == 1 & num_outputs == 1) {
    if (all(boxplot_df$input_dimension == 3)) {
      p <- boxplot_image(boxplot_df, value_name)
    } else if (all(boxplot_df$input_dimension %in% c(1,2))) {
      p <- boxplot_bar(boxplot_df, individual_df, value_name)
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

    stop("ToDo: Boxplot-function for multiple inputs!")
    p <- boxplot_extended(boxplot_df, individual_df, value_name)
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
    #p <-
    #  plotly::ggplotly(p, tooltip = "text", dynamicTicks = FALSE)
    warning("The plotly-plots for 'boxplot' are not implemented yet! Coming soon!")
  }

  p
}

boxplot_bar <- function(boxplot_df, individual_df, value_name) {

  p <- ggplot() +
    geom_boxplot(data = boxplot_df,
                 aes(x = .data$feature, y = .data$value), show.legend = FALSE, width = 0.8) +
    stat_boxplot(data = boxplot_df, aes(x = .data$feature, y = .data$value),
                  geom='errorbar', linetype=1, width = 0.8) +
    geom_errorbar(data = individual_df,
                 aes(x = .data$feature, ymin = .data$value, ymax = .data$value),
                 color = "red", size = 1, width = 0.8) +
    facet_grid(cols = vars(output_node), scales = "fixed") +
    geom_hline(yintercept = 0) +
    xlab(ifelse(all(boxplot_df$input_dimension == 2), "Signal Length", "Feature")) +
    ylab(value_name) +
    scale_x_discrete(
      guide = guide_axis(check.overlap = TRUE)
    )

  p

}

boxplot_image <- function(boxplot_df, value_name) {
  # Calculate median values
  res_df <- aggregate(
    boxplot_df$value,
    by = list(
      model_input = boxplot_df$model_input, model_output = boxplot_df$model_output,
      output_node = boxplot_df$output_node, feature = boxplot_df$feature,
      feature_2 = boxplot_df$feature_2), FUN = median)

  res_df$feature <- as.numeric(res_df$feature)
  res_df$feature_2 <- as.numeric(res_df$feature_2)

  ggplot(data = res_df) +
    geom_raster(aes(x = .data$feature, y = .data$feature_2, fill = .data$x)) +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
    facet_grid(cols = vars(output_node)) +
    xlab("Image Width") +
    labs(fill = paste0(value_name, "\n (median)")) +
    ylab("Image Height") +
    theme(legend.title = element_text(size = 11)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0))
}


