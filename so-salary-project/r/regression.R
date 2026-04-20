suppressPackageStartupMessages({
  library(arrow)
  library(broom)
  library(dplyr)
  library(lme4)
  library(readr)
  library(stringr)
  library(tibble)
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
results_path <- file.path(project_root, "outputs", "regression_results.csv")
summary_path <- file.path(project_root, "outputs", "regression_model_summary.csv")

df <- arrow::read_parquet(data_path) %>% as_tibble()

skill_cols <- names(df)[grepl("^(lang_|db_|plat_)", names(df))]

model_df <- df %>%
  filter(
    is_data_role == 1,
    !is.na(log_salary),
    !is.na(years_exp),
    !is.na(edu_num),
    !is.na(country_group),
    !is.na(exp_bucket)
  ) %>%
  mutate(
    country_group = factor(country_group),
    exp_bucket = factor(exp_bucket)
  )

formula_str <- paste(
  "log_salary ~ years_exp + edu_num + country_group + exp_bucket +",
  paste(skill_cols, collapse = " + ")
)

model <- lm(stats::as.formula(formula_str), data = model_df)

results <- broom::tidy(model, conf.int = TRUE) %>%
  mutate(
    pct_effect = if_else(term == "(Intercept)", NA_real_, (exp(estimate) - 1) * 100),
    significant = p.value < 0.05
  ) %>%
  arrange(desc(abs(estimate)))

model_summary <- broom::glance(model) %>%
  transmute(
    n_obs = nobs(model),
    r_squared = r.squared,
    adj_r_squared = adj.r.squared,
    sigma = sigma,
    aic = AIC(model),
    bic = BIC(model)
  )

readr::write_csv(results, results_path)
readr::write_csv(model_summary, summary_path)

sql_row <- results %>% filter(term == "lang_sql")
if (nrow(sql_row) > 0) {
  message(sprintf(
    "SQL premium: %.1f%% (p=%.4f)",
    sql_row$pct_effect[[1]],
    sql_row$p.value[[1]]
  ))
}

mixed_terms <- intersect(
  c("lang_python", "lang_sql", "lang_r", "db_postgresql", "plat_aws"),
  names(model_df)
)

if ("Country" %in% names(model_df) && length(mixed_terms) >= 2) {
  mixed_formula <- paste(
    "log_salary ~ years_exp + edu_num +",
    paste(mixed_terms, collapse = " + "),
    "+ (1 | Country)"
  )
  mixed_model <- lmer(stats::as.formula(mixed_formula), data = model_df)
  capture.output(
    summary(mixed_model),
    file = file.path(project_root, "outputs", "mixed_model_summary.txt")
  )
}

message(sprintf("Wrote %s", results_path))
message(sprintf("Wrote %s", summary_path))
