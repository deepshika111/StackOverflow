from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "survey_clean.parquet"
FEATURE_MATRIX_PATH = PROJECT_ROOT / "outputs" / "feature_matrix.csv"
SKILL_PREMIUMS_PATH = PROJECT_ROOT / "outputs" / "skill_premiums.csv"
REGRESSION_RESULTS_PATH = PROJECT_ROOT / "outputs" / "regression_results.csv"
REGRESSION_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "regression_model_summary.csv"
CLUSTER_LABELS_PATH = PROJECT_ROOT / "outputs" / "cluster_labels.csv"
CLUSTER_PROFILES_PATH = PROJECT_ROOT / "outputs" / "cluster_profiles.csv"
CLUSTER_ALIGNMENT_PATH = PROJECT_ROOT / "outputs" / "cluster_role_alignment.csv"
CLUSTER_ELBOW_PLOT_PATH = PROJECT_ROOT / "outputs" / "cluster_elbow_plot.png"

SALARY_COLUMN = "ConvertedCompYearly"
EXP_BUCKETS = ["0-2yr", "3-5yr", "6-10yr", "11-20yr", "20yr+"]
COUNTRY_GROUPS = ["tier1", "india", "other"]
SKILL_PREFIXES = ("lang_", "db_", "plat_")
SKILL_PREMIUM_COLUMNS = [
    "skill",
    "skill_label",
    "skill_family",
    "prevalence_pct",
    "with_skill",
    "without_skill",
    "premium_pct",
    "n_with",
    "n_without",
]


st.set_page_config(
    page_title="SO Salary Studio",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255,107,53,0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(15,118,110,0.16), transparent 26%),
                linear-gradient(180deg, #fff8f0 0%, #fffdf9 48%, #f7efe3 100%);
            color: #1f2937;
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        .hero-panel {
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(10,37,64,0.96), rgba(15,118,110,0.88));
            color: #f8fafc;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }
        .hero-copy {
            font-size: 1rem;
            max-width: 62rem;
            line-height: 1.5;
            color: rgba(248,250,252,0.9);
        }
        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            color: #64748b;
        }
        .section-note {
            padding: 0.9rem 1rem;
            border-left: 4px solid #ff6b35;
            background: rgba(255, 255, 255, 0.82);
            border-radius: 12px;
            margin: 0.75rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1f}%"


def tokenize_semicolon(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return [token.strip() for token in str(text).split(";") if token.strip()]


def optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def normalize_loaded_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    return df


def load_feature_matrix() -> tuple[pd.DataFrame | None, str | None]:
    source_path: Path | None = None
    parquet_error: Exception | None = None
    if PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(PARQUET_PATH)
            source_path = PARQUET_PATH
        except Exception as exc:
            parquet_error = exc
            df = None
    else:
        df = None

    if df is None and FEATURE_MATRIX_PATH.exists():
        df = pd.read_csv(FEATURE_MATRIX_PATH)
        source_path = FEATURE_MATRIX_PATH

    if df is None:
        if parquet_error is not None:
            return (
                None,
                "The parquet file exists but could not be read, and the CSV fallback is missing. "
                f"Parquet error: `{parquet_error}`. Run `python3 python/export_outputs.py` "
                "or regenerate the pipeline outputs.",
            )
        return None, "No processed dataset found yet. Run `python3 python/etl.py` first."

    df = normalize_loaded_frame(df)
    if SALARY_COLUMN not in df.columns:
        available = ", ".join(df.columns[:12].tolist())
        return (
            None,
            "The processed dataset was loaded, but the required salary column "
            f"`{SALARY_COLUMN}` is missing from `{source_path}`. "
            "This usually means Streamlit is serving stale state or the ETL output is out of date. "
            f"Visible columns start with: {available}",
        )

    df["Country"] = df.get("Country", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df["country_group"] = df.get("country_group", pd.Series(index=df.index, dtype="object")).fillna("other")
    df["exp_bucket"] = df.get("exp_bucket", pd.Series(index=df.index, dtype="object")).astype("string")
    df["DevType"] = df.get("DevType", pd.Series(index=df.index, dtype="object")).fillna("")
    df["role_tokens"] = df["DevType"].apply(tokenize_semicolon)
    df["primary_role"] = df["role_tokens"].apply(lambda tokens: tokens[0] if tokens else "Unspecified")
    return df, None


def load_optional_outputs() -> dict[str, pd.DataFrame | None]:
    return {
        "skill_premiums": optional_csv(SKILL_PREMIUMS_PATH),
        "regression_results": optional_csv(REGRESSION_RESULTS_PATH),
        "regression_summary": optional_csv(REGRESSION_SUMMARY_PATH),
        "cluster_labels": optional_csv(CLUSTER_LABELS_PATH),
        "cluster_profiles": optional_csv(CLUSTER_PROFILES_PATH),
        "cluster_alignment": optional_csv(CLUSTER_ALIGNMENT_PATH),
    }


def build_role_options(df: pd.DataFrame) -> list[str]:
    exploded = df["role_tokens"].explode().dropna()
    return exploded.value_counts().head(30).index.tolist()


def filter_rows(
    df: pd.DataFrame,
    countries: list[str],
    country_groups: list[str],
    exp_buckets: list[str],
    selected_roles: list[str],
    salary_range: tuple[int, int],
    data_roles_only: bool,
) -> pd.DataFrame:
    filtered = df.copy()

    if countries:
        filtered = filtered[filtered["Country"].isin(countries)]
    if country_groups:
        filtered = filtered[filtered["country_group"].isin(country_groups)]
    if exp_buckets:
        filtered = filtered[filtered["exp_bucket"].isin(exp_buckets)]
    if data_roles_only and "is_data_role" in filtered.columns:
        filtered = filtered[filtered["is_data_role"] == 1]
    if selected_roles:
        filtered = filtered[
            filtered["role_tokens"].apply(lambda tokens: any(role in tokens for role in selected_roles))
        ]

    low, high = salary_range
    filtered = filtered[filtered[SALARY_COLUMN].between(low, high)]
    return filtered


def compute_skill_premiums(df: pd.DataFrame) -> pd.DataFrame:
    skill_columns = [column for column in df.columns if column.startswith(SKILL_PREFIXES)]
    if not skill_columns or df.empty:
        return pd.DataFrame(columns=SKILL_PREMIUM_COLUMNS)

    rows: list[dict[str, float | int | str]] = []
    for skill in skill_columns:
        with_skill = df.loc[df[skill] == 1, SALARY_COLUMN]
        without_skill = df.loc[df[skill] == 0, SALARY_COLUMN]
        if with_skill.empty or without_skill.empty:
            continue

        with_mean = float(with_skill.mean())
        without_mean = float(without_skill.mean())
        rows.append(
            {
                "skill": skill,
                "skill_label": skill.split("_", 1)[1].replace("_", " ").title(),
                "skill_family": skill.split("_", 1)[0],
                "prevalence_pct": float(df[skill].mean() * 100),
                "with_skill": with_mean,
                "without_skill": without_mean,
                "premium_pct": float((with_mean / without_mean - 1) * 100),
                "n_with": int(with_skill.shape[0]),
                "n_without": int(without_skill.shape[0]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=SKILL_PREMIUM_COLUMNS)

    return pd.DataFrame(rows).sort_values("premium_pct", ascending=False)


def country_salary_table(df: pd.DataFrame, min_sample: int = 25) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Country", "n", "median_salary", "p75_salary"])

    grouped = (
        df.groupby("Country")[SALARY_COLUMN]
        .agg(
            n="count",
            median_salary=lambda salary: salary.quantile(0.5),
            p75_salary=lambda salary: salary.quantile(0.75),
        )
        .reset_index()
    )
    return grouped[grouped["n"] >= min_sample].sort_values("median_salary", ascending=False)


def benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["role", "country_group", "exp_bucket", "median_salary", "p75_salary", "n"])

    exploded = (
        df[["role_tokens", "country_group", "exp_bucket", SALARY_COLUMN]]
        .explode("role_tokens")
        .dropna(subset=["role_tokens", "country_group", "exp_bucket"])
        .rename(columns={"role_tokens": "role"})
    )
    table = (
        exploded.groupby(["role", "country_group", "exp_bucket"])[SALARY_COLUMN]
        .agg(
            median_salary=lambda salary: salary.quantile(0.5),
            p75_salary=lambda salary: salary.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    return table[table["n"] >= 10].sort_values(["role", "country_group", "exp_bucket"])


def render_hero(df: pd.DataFrame) -> None:
    median_salary = df[SALARY_COLUMN].median() if not df.empty else None
    p75_salary = df[SALARY_COLUMN].quantile(0.75) if not df.empty else None
    avg_breadth = df["skill_breadth"].mean() if "skill_breadth" in df.columns and not df.empty else None

    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-title">Stack Overflow Salary Studio</div>
            <div class="hero-copy">
                Explore salary benchmarks, skill premiums, regression outputs, and role-cluster overlap from the
                2024 Stack Overflow survey. The dashboard reads your local pipeline outputs, so it stays aligned with
                your resume bullet numbers and interview talking points.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Rows in view", f"{len(df):,}")
    metric_2.metric("Median salary", format_currency(median_salary))
    metric_3.metric("75th percentile", format_currency(p75_salary))
    metric_4.metric("Avg skill breadth", f"{avg_breadth:.1f}" if avg_breadth is not None else "n/a")


def render_overview(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-note">Use this page to sanity-check the filtered sample before you quote any headline number in a resume bullet or dashboard screenshot.</div>', unsafe_allow_html=True)

    countries = country_salary_table(df)
    col_left, col_right = st.columns((1.15, 1))

    with col_left:
        if countries.empty:
            st.info("No country-level salary table is available for the current filters.")
        else:
            fig = px.bar(
                countries.head(15),
                x="median_salary",
                y="Country",
                orientation="h",
                color="n",
                color_continuous_scale="Tealgrn",
                title="Top Countries by Median Salary",
                labels={"median_salary": "Median salary", "n": "Sample size"},
            )
            fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if df.empty:
            st.info("No salary distribution available for the current filters.")
        else:
            exp_order = [bucket for bucket in EXP_BUCKETS if bucket in df["exp_bucket"].dropna().unique().tolist()]
            fig = px.box(
                df,
                x="exp_bucket",
                y=SALARY_COLUMN,
                color="country_group",
                category_orders={"exp_bucket": exp_order},
                points=False,
                title="Salary Distribution by Experience Bucket",
                labels={SALARY_COLUMN: "Salary", "exp_bucket": "Experience bucket"},
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

    sample_mix = (
        df.groupby(["country_group", "primary_role"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(20)
    )
    if not sample_mix.empty:
        fig = px.treemap(
            sample_mix,
            path=["country_group", "primary_role"],
            values="n",
            color="n",
            color_continuous_scale="Sunsetdark",
            title="Where the Filtered Sample Comes From",
        )
        fig.update_layout(height=420, margin=dict(t=50, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_benchmarking(df: pd.DataFrame) -> None:
    benchmark = benchmark_table(df)
    col_left, col_right = st.columns((1.2, 1))

    with col_left:
        if benchmark.empty:
            st.info("Benchmark table is empty for the current filters.")
        else:
            preview = benchmark.copy()
            preview["median_salary"] = preview["median_salary"].round(0)
            preview["p75_salary"] = preview["p75_salary"].round(0)
            st.dataframe(preview.head(50), use_container_width=True, hide_index=True)

    with col_right:
        if df.empty:
            st.info("No benchmarking visuals available for the current filters.")
        else:
            role_rank = (
                df.explode("role_tokens")
                .dropna(subset=["role_tokens"])
                .groupby("role_tokens")[SALARY_COLUMN]
                .median()
                .reset_index()
                .sort_values(SALARY_COLUMN, ascending=False)
                .head(12)
            )
            fig = px.bar(
                role_rank,
                x=SALARY_COLUMN,
                y="role_tokens",
                orientation="h",
                color=SALARY_COLUMN,
                color_continuous_scale="Viridis",
                title="Median Salary by Role Token",
                labels={"role_tokens": "Role"},
            )
            fig.update_layout(height=540, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)


def render_skills(df: pd.DataFrame, optional_outputs: dict[str, pd.DataFrame | None]) -> None:
    st.caption("Skill premiums are recalculated from the filtered rows below, so slicers immediately change the ranking.")

    premiums = compute_skill_premiums(df)
    if premiums.empty:
        st.info(
            "No skill premiums are available for the current filters. "
            "This usually means the slice is too small and there are no skills with both "
            "a comparison group that has the skill and a group that does not."
        )
        return

    family_options = premiums["skill_family"].dropna().unique().tolist()
    selected_families = st.multiselect(
        "Skill families",
        options=family_options,
        default=family_options,
        key="skill_family_filter",
    )

    filtered_premiums = premiums[premiums["skill_family"].isin(selected_families)].copy()

    col_left, col_right = st.columns((1.05, 1))
    with col_left:
        fig = px.bar(
            filtered_premiums.head(20),
            x="premium_pct",
            y="skill_label",
            orientation="h",
            color="premium_pct",
            color_continuous_scale=["#b91c1c", "#f59e0b", "#0f766e"],
            title="Top Skill Premiums in the Current Slice",
            labels={"premium_pct": "Premium %", "skill_label": "Skill"},
        )
        fig.update_layout(height=620, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.scatter(
            filtered_premiums,
            x="prevalence_pct",
            y="premium_pct",
            size="n_with",
            color="skill_family",
            hover_name="skill_label",
            title="Prevalence vs. Salary Premium",
            labels={"prevalence_pct": "Skill prevalence %", "premium_pct": "Premium %"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    regression = optional_outputs.get("regression_results")
    if regression is not None and not regression.empty:
        skill_coeffs = regression[
            regression["term"].str.startswith(SKILL_PREFIXES) & regression["significant"].fillna(False)
        ].copy()
        if not skill_coeffs.empty:
            skill_coeffs["label"] = skill_coeffs["term"].str.split("_", n=1).str[-1].str.replace("_", " ").str.title()
            skill_coeffs["pct_effect"] = skill_coeffs["pct_effect"].round(1)
            fig = px.bar(
                skill_coeffs.sort_values("pct_effect", ascending=False).head(15),
                x="pct_effect",
                y="label",
                orientation="h",
                color="pct_effect",
                color_continuous_scale="Emrld",
                title="Top Significant Regression Effects",
                labels={"pct_effect": "Estimated salary effect %", "label": "Skill"},
            )
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)


def estimator_controls(
    regression: pd.DataFrame,
    regression_summary: pd.DataFrame | None,
) -> None:
    coeffs = dict(zip(regression["term"], regression["estimate"]))

    intercept = coeffs.get("(Intercept)", 0.0)
    years_exp = st.slider("Years of professional coding", min_value=0, max_value=25, value=5)
    edu_num = st.select_slider("Education level score", options=list(range(1, 9)), value=5)

    present_country_terms = {term.replace("country_group", "") for term in coeffs if term.startswith("country_group")}
    baseline_country = next((group for group in COUNTRY_GROUPS if group not in present_country_terms), COUNTRY_GROUPS[0])
    country_choice = st.selectbox("Country group", options=COUNTRY_GROUPS, index=COUNTRY_GROUPS.index(baseline_country))

    present_exp_terms = {term.replace("exp_bucket", "") for term in coeffs if term.startswith("exp_bucket")}
    baseline_exp = next((bucket for bucket in EXP_BUCKETS if bucket not in present_exp_terms), EXP_BUCKETS[0])
    exp_choice = st.selectbox("Experience bucket", options=EXP_BUCKETS, index=EXP_BUCKETS.index(baseline_exp))

    positive_skills = regression[
        regression["term"].str.startswith(SKILL_PREFIXES) & regression["estimate"].gt(0)
    ].sort_values("estimate", ascending=False)
    selected_skills = st.multiselect(
        "Modeled skills",
        options=positive_skills["term"].head(12).tolist(),
        format_func=lambda term: term.split("_", 1)[1].replace("_", " ").title(),
    )

    predicted_log_salary = intercept
    predicted_log_salary += coeffs.get("years_exp", 0.0) * years_exp
    predicted_log_salary += coeffs.get("edu_num", 0.0) * edu_num
    predicted_log_salary += coeffs.get(f"country_group{country_choice}", 0.0)
    predicted_log_salary += coeffs.get(f"exp_bucket{exp_choice}", 0.0)
    predicted_log_salary += sum(coeffs.get(skill, 0.0) for skill in selected_skills)

    predicted_salary = float(np.exp(predicted_log_salary))
    st.metric("Estimated salary from the OLS model", format_currency(predicted_salary))

    if regression_summary is not None and not regression_summary.empty:
        model_row = regression_summary.iloc[0]
        st.caption(
            f"Adjusted R²: {model_row['adj_r_squared']:.3f} | "
            f"Observations: {int(model_row['n_obs']):,}"
        )
    st.caption(f"Reference group for country is `{baseline_country}` and for experience bucket is `{baseline_exp}`.")


def render_clusters_and_model(
    df: pd.DataFrame,
    optional_outputs: dict[str, pd.DataFrame | None],
) -> None:
    cluster_profiles = optional_outputs.get("cluster_profiles")
    cluster_alignment = optional_outputs.get("cluster_alignment")
    cluster_labels = optional_outputs.get("cluster_labels")
    regression = optional_outputs.get("regression_results")
    regression_summary = optional_outputs.get("regression_summary")

    left_col, right_col = st.columns((1.05, 0.95))

    with left_col:
        if cluster_profiles is None or cluster_profiles.empty:
            st.info("Run `Rscript r/clustering.R` to populate the cluster views.")
        else:
            matrix = cluster_profiles.pivot(index="skill", columns="cluster", values="center").fillna(0)
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix.values,
                    x=[f"Cluster {column}" for column in matrix.columns],
                    y=[skill.replace("_", " ").title() for skill in matrix.index],
                    colorscale="YlGnBu",
                )
            )
            fig.update_layout(height=520, title="Cluster Signature Heatmap")
            st.plotly_chart(fig, use_container_width=True)

            if CLUSTER_ELBOW_PLOT_PATH.exists():
                st.image(str(CLUSTER_ELBOW_PLOT_PATH), caption="Elbow plot used to justify k = 3")

    with right_col:
        if cluster_alignment is None or cluster_alignment.empty:
            st.info("Cluster-role alignment appears here after the clustering script runs.")
        else:
            fig = px.bar(
                cluster_alignment,
                x="role_clean",
                y="pct",
                color="cluster",
                barmode="stack",
                title="Role Membership Across Clusters",
                labels={"pct": "Percent of role", "role_clean": "Role"},
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

        if cluster_labels is not None and not cluster_labels.empty:
            fig = px.box(
                cluster_labels,
                x="cluster",
                y="ConvertedCompYearly",
                color="role_clean",
                title="Salary Spread by Cluster",
                labels={"ConvertedCompYearly": "Salary"},
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("What-If Salary Estimator")
    if regression is None or regression.empty:
        st.info("Run `Rscript r/regression.R` to enable the salary estimator.")
    else:
        estimator_controls(regression, regression_summary)


inject_styles()
df, load_error = load_feature_matrix()
optional_outputs = load_optional_outputs()

if load_error is not None or df is None:
    st.error(load_error or "The dashboard could not load the processed survey dataset.")
    st.stop()

salary_min = int(df[SALARY_COLUMN].min())
salary_max = int(df[SALARY_COLUMN].max())
role_options = build_role_options(df)

st.sidebar.header("Filter the sample")
data_roles_only = st.sidebar.toggle("Data roles only", value=True)
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=sorted(df["Country"].dropna().unique().tolist()),
)
selected_country_groups = st.sidebar.multiselect(
    "Country groups",
    options=COUNTRY_GROUPS,
    default=COUNTRY_GROUPS,
)
selected_exp_buckets = st.sidebar.multiselect(
    "Experience buckets",
    options=EXP_BUCKETS,
    default=EXP_BUCKETS,
)
selected_roles = st.sidebar.multiselect(
    "Role tokens",
    options=role_options,
)
selected_salary = st.sidebar.slider(
    "Salary range",
    min_value=salary_min,
    max_value=salary_max,
    value=(salary_min, salary_max),
    step=5_000,
)

filtered_df = filter_rows(
    df=df,
    countries=selected_countries,
    country_groups=selected_country_groups,
    exp_buckets=selected_exp_buckets,
    selected_roles=selected_roles,
    salary_range=selected_salary,
    data_roles_only=data_roles_only,
)

render_hero(filtered_df)

overview_tab, benchmark_tab, skills_tab, cluster_tab = st.tabs(
    ["Overview", "Benchmarking", "Skills", "Clusters + Model"]
)

with overview_tab:
    render_overview(filtered_df)

with benchmark_tab:
    render_benchmarking(filtered_df)

with skills_tab:
    render_skills(filtered_df, optional_outputs)

with cluster_tab:
    render_clusters_and_model(filtered_df, optional_outputs)
