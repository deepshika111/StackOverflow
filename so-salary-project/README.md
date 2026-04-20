# Stack Overflow Salary Project

End-to-end scaffold for analyzing the 2024 Stack Overflow Developer Survey with Python, SQLite, R, Streamlit, Power BI, and Excel.

## Project layout

```text
so-salary-project/
├── .streamlit/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
├── powerbi/
├── python/
├── r/
├── sql/
└── streamlit_app.py
```

## 1. Get the data

Download the 2024 full CSV from [survey.stackoverflow.co](https://survey.stackoverflow.co/) and place it here:

`data/raw/survey_results_public.csv`

## 2. Python environment

Create a virtual environment and install the Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. R packages

Install the R dependencies:

```bash
Rscript r/install_packages.R
```

`arrow` is included because the R scripts read the cleaned parquet output.

## 4. Run the pipeline

From the project root:

```bash
python3 python/etl.py
python3 python/load_sqlite.py
python3 python/export_outputs.py
Rscript r/regression.R
Rscript r/clustering.R
```

## 5. Run the Streamlit app

Once the parquet and output tables exist, launch the dashboard from the project root:

```bash
streamlit run streamlit_app.py
```

The app reads the cleaned parquet file plus any optional analysis outputs that exist, so it will get richer as you run the later R scripts.

## 6. Outputs you will get

- `data/processed/survey_clean.parquet`
- `data/processed/survey.db`
- `outputs/etl_summary.json`
- `outputs/feature_matrix.csv`
- `outputs/skill_premiums.csv`
- `outputs/benchmarking_table.csv`
- `outputs/benchmarking_table.xlsx`
- `outputs/regression_results.csv`
- `outputs/regression_model_summary.csv`
- `outputs/cluster_labels.csv`
- `outputs/cluster_profiles.csv`
- `outputs/cluster_role_alignment.csv`
- `outputs/cluster_elbow_plot.png`

## 7. Power BI

Use the notes in [powerbi/README.md](/Users/deepshikathakur/Documents/Google/StackOverflow/so-salary-project/powerbi/README.md) and [powerbi/measures.dax](/Users/deepshikathakur/Documents/Google/StackOverflow/so-salary-project/powerbi/measures.dax) for the dashboard setup.

## Notes

- The ETL script logs row-count changes and the selected top multi-select values.
- `streamlit_app.py` gives you a code-first dashboard alternative if you want something shareable without rebuilding visuals in Power BI.
- The scaffold assumes you want a fresh project directory inside this workspace rather than using the workspace root as the project root.
- Some survey text labels vary slightly across Stack Overflow releases, so the role and education parsing is intentionally tolerant.
