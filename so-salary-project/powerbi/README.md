# Power BI Dashboard Blueprint

## Import these files

- `outputs/feature_matrix.csv`
- `outputs/skill_premiums.csv`
- `outputs/regression_results.csv`
- `outputs/cluster_labels.csv`
- `outputs/cluster_profiles.csv`
- `outputs/benchmarking_table.csv`

## Page 1: Salary Benchmarking

- Slicers: `Country`, `role` or `DevType`, `exp_bucket`, `country_group`
- Cards: median salary, 75th percentile salary, sample size
- Bar chart: median salary by country
- Box plot: salary distribution within the current filter

Use `feature_matrix.csv` for individual-row visuals and `benchmarking_table.csv` for summarized tables.

## Page 2: Skill Premiums

- Horizontal bar chart: `skill_label` by `premium_pct`
- Scatter plot: `prevalence_pct` vs `premium_pct`
- Conditional coloring: green for positive premium, red for negative premium
- Optional slicer: `skill_family`

Use `skill_premiums.csv`.

## Page 3: Learning Paths

- Matrix heatmap: `cluster` by `skill` with `center` as the value
- Cluster comparison visual: average salary by cluster
- What-if salary estimator backed by the regression coefficients

Use `cluster_profiles.csv`, `cluster_labels.csv`, and `regression_results.csv`.

## Suggested relationships

- Keep the import tables disconnected unless you add a custom dimension table for roles or skills.
- If you want cross-page filtering, create shared dimension tables for `role`, `country_group`, and `exp_bucket`.

## Build order

1. Import the CSV outputs
2. Add the DAX measures from `powerbi/measures.dax`
3. Build slicers and cards first
4. Add charts and conditional formatting
5. Add What-If parameters for experience and skill flags
