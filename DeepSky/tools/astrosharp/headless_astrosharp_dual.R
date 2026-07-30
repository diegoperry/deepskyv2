#!/usr/bin/env Rscript

# Headless dual-model adapter for the MIT-licensed AstroSharp DualPSF branch.
# Upstream: https://github.com/deepskydetail/AstroSharp
#
# AstroSharp's UI runs one PSF model for extended DSO structure and a second
# PSF model for stars. This adapter exports both luminance donors. DeepSky then
# performs the continuous stellar blend and signal/background protection.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop(
    paste(
      "Usage: headless_astrosharp_dual.R",
      "<input.tif> <dso_output.tif> <star_output.tif> <astrosharp_app_dir>",
      "<dso_psf> <star_psf> <aggressiveness_0_to_1> <chunk_size>"
    )
  )
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
dso_output_path <- args[[2]]
star_output_path <- args[[3]]
app_dir <- normalizePath(args[[4]], mustWork = TRUE)
dso_psf <- round(as.numeric(args[[5]]) * 4) / 4
star_psf <- round(as.numeric(args[[6]]) * 4) / 4
aggressiveness <- as.numeric(args[[7]])
chunk_size <- as.integer(args[[8]])

for (value in c(dso_psf, star_psf)) {
  if (!is.finite(value) || value < 1 || value > 8) {
    stop("AstroSharp PSF values must be between 1 and 8.")
  }
}
if (!is.finite(aggressiveness) || aggressiveness < 0 || aggressiveness > 1) {
  stop("AstroSharp aggressiveness must be between 0 and 1.")
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

load_psf_model <- function(value) {
  path <- file.path(
    app_dir,
    "PSF",
    paste0("81_1_FWHM_", format_psf(value), ".RDS")
  )
  if (!file.exists(path)) {
    stop(paste("AstroSharp PSF model not found:", path))
  }
  readRDS(path)
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

dso_model <- load_psf_model(dso_psf)
star_model <- load_psf_model(star_psf)
dso_luminance <- luminance
star_luminance <- luminance
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
    matrix81 <- as.matrix(features[, 1:81])
    original <- features[, 9]
    dso_predicted <- as.numeric(
      neuralnet::compute(dso_model, matrix81)$net.result
    )
    star_predicted <- as.numeric(
      neuralnet::compute(star_model, matrix81)$net.result
    )
    dso_values <- pmin(
      1,
      pmax(
        0,
        dso_predicted * aggressiveness + original * (1 - aggressiveness)
      )
    )
    star_values <- pmin(
      1,
      pmax(
        0,
        star_predicted * aggressiveness + original * (1 - aggressiveness)
      )
    )
    output_rows <- row_end - row_start + 1
    dso_luminance[row_start:row_end, col_start:col_end] <- matrix(
      dso_values,
      nrow = output_rows
    )
    star_luminance[row_start:row_end, col_start:col_end] <- matrix(
      star_values,
      nrow = output_rows
    )
    cat(
      sprintf(
        "AstroSharp DualPSF tile %d/%d (%.1f%%)\n",
        tile_index,
        tile_count,
        tile_index * 100 / tile_count
      )
    )
    flush.console()
  }
}

restore_color <- function(output_luminance) {
  if (!is_color) {
    return(output_luminance)
  }
  output_luv <- luv_frame
  output_luv[, 1] <- as.vector(output_luminance) * 100
  output_frame <- grDevices::convertColor(
    output_luv,
    from = "Luv",
    to = "sRGB"
  )
  output_frame[output_frame < 0] <- 0
  output_frame[output_frame > 1] <- 1
  array(
    c(
      matrix(output_frame[, 1], nrow = height),
      matrix(output_frame[, 2], nrow = height),
      matrix(output_frame[, 3], nrow = height)
    ),
    dim = c(height, width, 3)
  )
}

dir.create(dirname(dso_output_path), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(star_output_path), recursive = TRUE, showWarnings = FALSE)
writeTIFF(
  restore_color(dso_luminance),
  dso_output_path,
  bits.per.sample = 16,
  compression = "LZW"
)
writeTIFF(
  restore_color(star_luminance),
  star_output_path,
  bits.per.sample = 16,
  compression = "LZW"
)
elapsed <- proc.time()[["elapsed"]] - started
cat(
  sprintf(
    paste0(
      "AstroSharp DualPSF complete: dso_psf=%s star_psf=%s ",
      "aggressiveness=%.3f chunks=%d elapsed=%.2fs ",
      "dso_output=%s star_output=%s\n"
    ),
    format_psf(dso_psf),
    format_psf(star_psf),
    aggressiveness,
    tile_count,
    elapsed,
    normalizePath(dso_output_path, mustWork = FALSE),
    normalizePath(star_output_path, mustWork = FALSE)
  )
)
