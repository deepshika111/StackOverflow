suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(factoextra)
  library(ggplot2)
  library(readr)
  library(stringr)
  library(tibble)
  library(tidyr)
})

get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))))
  }
  normalizePath(".")
}

project_root <- normalizePath(file.path(get_script_dir(), ".."))
data_path <- file.path(project_root, "data", "processed", "survey_clean.parquet")
outputs_dir <- file.path(project_root, "outputs")

df <- arrow::read_parquet(data_path) %>% as_tibble()
skill_cols <- names(df)[grepl("^(lang_|db_|plat_)", names(df))]

role_df <- df %>%
  filter(str_detect(coalesce(DevType, ""), regex("data scientist|data engineer|data analyst|data or business analyst", ignore_case = TRUE))) %>%
  mutate(
    role_clean = case_when(
      str_detect(DevType, regex("data scientist", ignore_case = TRUE)) ~ "Data Scientist",
      str_detect(DevType, regex("data engineer", ignore_case = TRUE)) ~ "Data Engineer",
      TRUE ~ "Data Analyst"
    )
  )

skill_matrix <- role_df %>%
  select(all_of(skill_cols))

skill_scaled <- scale(skill_matrix)

set.seed(42)
elbow_plot <- fviz_nbclust(skill_scaled, kmeans, method = "wss") +
  labs(title = "Optimal Number of Skill Clusters")

ggsave(
  filename = file.path(outputs_dir, "cluster_elbow_plot.png"),
  plot = elbow_plot,
  width = 9,
  height = 6,
  dpi = 300
)

km <- kmeans(skill_scaled, centers = 3, nstart = 25, iter.max = 100)

role_df$cluster <- km$cluster

cluster_labels <- role_df %>%
  select(role_clean, cluster, ConvertedCompYearly)

cluster_profiles <- as_tibble(km$centers, rownames = "cluster") %>%
  pivot_longer(-cluster, names_to = "skill", values_to = "center") %>%
  group_by(cluster) %>%
  slice_max(order_by = center, n = 8, with_ties = FALSE) %>%
  ungroup()

cluster_alignment <- role_df %>%
  count(role_clean, cluster) %>%
  group_by(role_clean) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup()

write_csv(cluster_labels, file.path(outputs_dir, "cluster_labels.csv"))
write_csv(cluster_profiles, file.path(outputs_dir, "cluster_profiles.csv"))
write_csv(cluster_alignment, file.path(outputs_dir, "cluster_role_alignment.csv"))

message(sprintf("Wrote %s", file.path(outputs_dir, "cluster_labels.csv")))
message(sprintf("Wrote %s", file.path(outputs_dir, "cluster_profiles.csv")))
message(sprintf("Wrote %s", file.path(outputs_dir, "cluster_role_alignment.csv")))
