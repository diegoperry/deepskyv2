#!/usr/bin/env Rscript

# Headless adapter for the MIT-licensed AstroSharp PSF models.
# Upstream: https://github.com/deepskydetail/AstroSharp
#
# This adapter intentionally runs the single-PSF luminance model only. DeepSky
# supplies its own continuous nebula/star/background protection masks after
# inference, avoiding AstroSharp's GUI-only hard star-mask path.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(
    paste(
      "Usage: headless_astrosharp.R",
      "<input.tif> <output.tif> <astrosharp_app_dir>",
      "<psf_fwhm_div_2.35> <strength_0_to_1> <chunk_size>"
    )
  )
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
output_path <- args[[2]]
app_dir <- normalizePath(args[[3]], mustWork = TRUE)
psf_value <- as.numeric(args[[4]])
strength <- as.numeric(args[[5]])
chunk_size <- as.integer(args[[6]])

if (!is.finite(psf_value) || psf_value < 1 || psf_value > 8) {
  stop("AstroSharp PSF value must be between 1 and 8.")
}
psf_value <- round(psf_value * 4) / 4
if (!is.finite(strength) || strength < 0 || strength > 1) {
  stop("AstroSharp strength must be between 0 and 1.")
}
if (!is.finite(chunk_size) || chunk_size < 32 || chunk_size > 750) {
  stop("AstroSharp chunk size must be between 32 and 750.")
}

portable_library <- file.path(app_dir, "R-Portable-Win", "library")
if (dir.exists(portable_library)) {
  .libPaths(c(portable_library, .libPaths()))
}

suppressPackageStartupMessages(library(tiff))
suppressPackageStartupMessages(library(neuralnet))
source(file.path(app_dir, "GetMatrixFun9x9.R"))

format_psf <- function(value) {
  if (abs(value - round(value)) < 1e-9) {
    as.character(as.integer(round(value)))
  } else {
    sub("0+$", "", sub("\\.$", "", format(value, scientific = FALSE)))
  }
}

model_path <- file.path(
  app_dir,
  "PSF",
  paste0("81_1_FWHM_", format_psf(psf_value), ".RDS")
)
if (!file.exists(model_path)) {
  stop(paste("AstroSharp PSF model not found:", model_path))
}

image <- readTIFF(input_path, native = FALSE, convert = FALSE)
is_color <- length(dim(image)) == 3 && dim(image)[[3]] >= 3
if (is_color) {
  image <- image[, , 1:3, drop = FALSE]
  rgb_frame <- data.frame(
    R = as.vector(image[, , 1]),
    G = as.vector(image[, , 2]),
    B = as.vector(image[, , 3])
  )
  luv_frame <- grDevices::convertColor(rgb_frame, from = "sRGB", to = "Luv")
  luminance <- matrix(luv_frame[, 1], nrow = dim(image)[[1]]) / 100
} else {
  luminance <- as.matrix(image)
}

height <- nrow(luminance)
width <- ncol(luminance)
if (height < 16 || width < 16) {
  stop("AstroSharp requires both image dimensions to be at least 16 pixels.")
}

model <- readRDS(model_path)
result_luminance <- luminance
row_starts <- seq(5, height - 4, by = chunk_size)
col_starts <- seq(5, width - 4, by = chunk_size)
tile_count <- length(row_starts) * length(col_starts)
tile_index <- 0

started <- proc.time()[["elapsed"]]
for (col_start in col_starts) {
  col_end <- min(width - 4, col_start + chunk_size - 1)
  for (row_start in row_starts) {
    row_end <- min(height - 4, row_start + chunk_size - 1)
    tile_index <- tile_index + 1

    tile <- luminance[
      (row_start - 4):(row_end + 4),
      (col_start - 4):(col_end + 4),
      drop = FALSE
    ]
    features <- getmatrix9(tile)
    predicted <- as.numeric(
      neuralnet::compute(model, as.matrix(features[, 1:81]))$net.result
    )
    original <- features[, 9]
    blended <- pmin(1, pmax(0, predicted * strength + original * (1 - strength)))
    result_luminance[row_start:row_end, col_start:col_end] <- matrix(
      blended,
      nrow = row_end - row_start + 1
    )

    cat(
      sprintf(
        "AstroSharp tile %d/%d (%.1f%%)\n",
        tile_index,
        tile_count,
        tile_index * 100 / tile_count
      )
    )
    flush.console()
  }
}

if (is_color) {
  luv_frame[, 1] <- as.vector(result_luminance) * 100
  output_frame <- grDevices::convertColor(luv_frame, from = "Luv", to = "sRGB")
  output_frame[output_frame < 0] <- 0
  output_frame[output_frame > 1] <- 1
  output <- array(
    c(
      matrix(output_frame[, 1], nrow = height),
      matrix(output_frame[, 2], nrow = height),
      matrix(output_frame[, 3], nrow = height)
    ),
    dim = c(height, width, 3)
  )
} else {
  output <- result_luminance
}

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
writeTIFF(output, output_path, bits.per.sample = 16, compression = "LZW")
elapsed <- proc.time()[["elapsed"]] - started
cat(
  sprintf(
    "AstroSharp complete: psf=%s strength=%.3f chunks=%d elapsed=%.2fs output=%s\n",
    format_psf(psf_value),
    strength,
    tile_count,
    elapsed,
    normalizePath(output_path, mustWork = FALSE)
  )
)
