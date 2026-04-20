packages <- c(
  "arrow",
  "broom",
  "cluster",
  "factoextra",
  "ggplot2",
  "lme4",
  "openxlsx",
  "tidyverse"
)

missing_packages <- packages[!packages %in% installed.packages()[, "Package"]]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
} else {
  message("All required R packages are already installed.")
}
