import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def persist_widget_keys(*keys):
    """Keep keyed widget values alive when switching between Streamlit pages."""
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


st.title("📈 Visualize Data")

persist_widget_keys(
    "visualization_type",
    "bar_x",
    "bar_y",
    "bar_agg",
    "bar_color",
    "pie_label",
    "pie_value",
    "pie_agg",
    "hist_value",
    "hist_bins",
    "line_x",
    "line_y",
    "line_agg",
    "scatter_x",
    "scatter_y",
    "scatter_color",
    "area_x",
    "area_y",
    "area_agg",
    "box_y",
    "box_x",
    "bubble_x",
    "bubble_y",
    "bubble_size",
    "bubble_color",
)


def get_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [col for col in df.columns if col not in numeric_columns]
    return numeric_columns, categorical_columns


def build_aggregated_frame(
    df: pd.DataFrame, group_column: str, value_column: str, aggregation: str
) -> pd.DataFrame:
    aggregation_map = {
        "Sum": "sum",
        "Average": "mean",
        "Median": "median",
        "Minimum": "min",
        "Maximum": "max",
    }

    grouped = (
        df.groupby(group_column, dropna=False)[value_column]
        .agg(aggregation_map[aggregation])
        .reset_index()
    )
    grouped.columns = [group_column, value_column]
    return grouped


def finalize_figure(fig: plt.Figure) -> None:
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_chart_safely(render_fn, df: pd.DataFrame, numeric_columns: list[str]) -> None:
    try:
        render_fn(df, numeric_columns)
    except Exception:
        st.error("Something went wrong.")


def render_bar_chart(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if not numeric_columns:
        st.warning("Bar chart needs at least one numeric column for the Y-axis.")
        return

    x_column = st.selectbox("Choose X-axis column", df.columns, key="bar_x")
    y_column = st.selectbox("Choose Y-axis column", numeric_columns, key="bar_y")
    aggregation = st.selectbox(
        "Choose Y-axis aggregation",
        ["Sum", "Average", "Median", "Minimum", "Maximum"],
        key="bar_agg",
    )
    color_column = st.selectbox(
        "Optional color grouping",
        ["None"] + df.columns.tolist(),
        key="bar_color",
    )

    chart_df = build_aggregated_frame(df, x_column, y_column, aggregation).sort_values(
        by=y_column, ascending=False
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = None
    if color_column != "None" and color_column == x_column:
        colors = plt.cm.tab20.colors[: len(chart_df)]
    ax.bar(chart_df[x_column].astype(str), chart_df[y_column], color=colors)
    ax.set_xlabel(x_column)
    ax.set_ylabel(f"{aggregation} of {y_column}")
    ax.set_title(f"Bar Chart: {x_column} vs {y_column}")
    ax.tick_params(axis="x", rotation=45)
    finalize_figure(fig)


def render_pie_chart(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if not numeric_columns:
        st.warning("Pie chart needs one numeric value column.")
        return

    label_column = st.selectbox("Choose category column", df.columns, key="pie_label")
    value_column = st.selectbox("Choose value column", numeric_columns, key="pie_value")
    aggregation = st.selectbox(
        "Choose value aggregation",
        ["Sum", "Average", "Median", "Minimum", "Maximum"],
        key="pie_agg",
    )

    chart_df = build_aggregated_frame(df, label_column, value_column, aggregation)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        chart_df[value_column],
        labels=chart_df[label_column].astype(str),
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(f"Pie Chart: {aggregation} of {value_column} by {label_column}")
    ax.axis("equal")
    finalize_figure(fig)
    st.dataframe(chart_df, use_container_width=True)


def render_histogram(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if not numeric_columns:
        st.warning("Histogram needs at least one numeric column.")
        return

    value_column = st.selectbox(
        "Choose numeric column for histogram",
        numeric_columns,
        key="hist_value",
    )
    max_bins = st.slider(
        "Choose number of bins", min_value=5, max_value=60, value=20, key="hist_bins"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[value_column].dropna(), bins=max_bins, edgecolor="black")
    ax.set_xlabel(value_column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {value_column}")
    finalize_figure(fig)


def render_line_chart(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if not numeric_columns:
        st.warning("Line chart needs at least one numeric column for the Y-axis.")
        return

    x_column = st.selectbox("Choose X-axis column", df.columns, key="line_x")
    y_column = st.selectbox("Choose Y-axis column", numeric_columns, key="line_y")
    aggregation = st.selectbox(
        "Choose Y-axis aggregation",
        ["Sum", "Average", "Median", "Minimum", "Maximum"],
        key="line_agg",
    )

    chart_df = build_aggregated_frame(df, x_column, y_column, aggregation)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(chart_df[x_column].astype(str), chart_df[y_column], marker="o")
    ax.set_xlabel(x_column)
    ax.set_ylabel(f"{aggregation} of {y_column}")
    ax.set_title(f"Line Chart: {x_column} vs {y_column}")
    ax.tick_params(axis="x", rotation=45)
    finalize_figure(fig)


def render_scatter_plot(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if len(numeric_columns) < 2:
        st.warning("Scatter plot needs at least two numeric columns.")
        return

    x_column = st.selectbox("Choose X-axis column", numeric_columns, key="scatter_x")
    y_choices = [col for col in numeric_columns if col != x_column]
    y_column = st.selectbox("Choose Y-axis column", y_choices, key="scatter_y")
    color_column = st.selectbox(
        "Optional color grouping",
        ["None"] + df.columns.tolist(),
        key="scatter_color",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    if color_column != "None":
        color_codes = pd.Categorical(df[color_column].astype(str)).codes
        scatter = ax.scatter(df[x_column], df[y_column], c=color_codes, alpha=0.7)
        fig.colorbar(scatter, ax=ax, label=color_column)
    else:
        ax.scatter(df[x_column], df[y_column], alpha=0.7)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
    finalize_figure(fig)


def render_area_chart(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if not numeric_columns:
        st.warning("Area chart needs at least one numeric column for the Y-axis.")
        return

    x_column = st.selectbox("Choose X-axis column", df.columns, key="area_x")
    y_column = st.selectbox("Choose Y-axis column", numeric_columns, key="area_y")
    aggregation = st.selectbox(
        "Choose Y-axis aggregation",
        ["Sum", "Average", "Median", "Minimum", "Maximum"],
        key="area_agg",
    )

    chart_df = build_aggregated_frame(df, x_column, y_column, aggregation)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(chart_df[x_column].astype(str), chart_df[y_column], alpha=0.6)
    ax.plot(chart_df[x_column].astype(str), chart_df[y_column], linewidth=2)
    ax.set_xlabel(x_column)
    ax.set_ylabel(f"{aggregation} of {y_column}")
    ax.set_title(f"Area Chart: {x_column} vs {y_column}")
    ax.tick_params(axis="x", rotation=45)
    finalize_figure(fig)


def render_box_plot(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if not numeric_columns:
        st.warning("Box plot needs at least one numeric column.")
        return

    y_column = st.selectbox("Choose numeric column", numeric_columns, key="box_y")
    x_column = st.selectbox(
        "Optional grouping column",
        ["None"] + df.columns.tolist(),
        key="box_x",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    if x_column != "None":
        grouped = [
            df.loc[df[x_column] == category, y_column].dropna()
            for category in df[x_column].dropna().unique()
        ]
        labels = [str(category) for category in df[x_column].dropna().unique()]
        ax.boxplot(grouped, tick_labels=labels)
        ax.set_xlabel(x_column)
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.boxplot(df[y_column].dropna())
        ax.set_xticks([1])
        ax.set_xticklabels([y_column])
    ax.set_ylabel(y_column)
    ax.set_title(f"Box Plot of {y_column}")
    finalize_figure(fig)


def render_bubble_chart(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if len(numeric_columns) < 3:
        st.warning("Bubble chart needs at least three numeric columns for X, Y, and bubble size.")
        return

    x_column = st.selectbox("Choose X-axis column", numeric_columns, key="bubble_x")
    y_choices = [col for col in numeric_columns if col != x_column]
    y_column = st.selectbox("Choose Y-axis column", y_choices, key="bubble_y")
    size_choices = [col for col in numeric_columns if col not in {x_column, y_column}]
    size_column = st.selectbox("Choose bubble size column", size_choices, key="bubble_size")
    color_column = st.selectbox(
        "Optional color grouping",
        ["None"] + df.columns.tolist(),
        key="bubble_color",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bubble_sizes = df[size_column].fillna(0).abs()
    max_size = bubble_sizes.max()
    scaled_sizes = (bubble_sizes / max_size * 1200).clip(lower=50) if max_size else 100

    if color_column != "None":
        color_codes = pd.Categorical(df[color_column].astype(str)).codes
        scatter = ax.scatter(
            df[x_column], df[y_column], s=scaled_sizes, c=color_codes, alpha=0.6
        )
        fig.colorbar(scatter, ax=ax, label=color_column)
    else:
        ax.scatter(df[x_column], df[y_column], s=scaled_sizes, alpha=0.6)

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Bubble Chart: {x_column} vs {y_column} sized by {size_column}")
    finalize_figure(fig)


processed_df = st.session_state.get("processed_df")

if processed_df is None:
    st.warning("Upload and process a CSV file first to visualize data.")
else:
    numeric_columns, categorical_columns = get_column_groups(processed_df)

    st.success(
        f"Data is ready for visualization: {len(processed_df)} rows and {len(processed_df.columns)} columns."
    )
    st.dataframe(processed_df, use_container_width=True)

    st.subheader("Choose Visualization")
    visualization_type = st.selectbox(
        "Select visualization type",
        [
            "Bar Chart",
            "Pie Chart",
            "Histogram",
            "Line Chart",
            "Scatter Plot",
            "Area Chart",
            "Box Plot",
            "Bubble Chart",
        ],
        key="visualization_type",
    )

    st.caption(
        "Choose the columns for this visualization. Axis-based charts let you decide what should appear on the X-axis and Y-axis from the cleaned dataset."
    )

    if visualization_type == "Bar Chart":
        render_chart_safely(render_bar_chart, processed_df, numeric_columns)
    elif visualization_type == "Pie Chart":
        render_chart_safely(render_pie_chart, processed_df, numeric_columns)
    elif visualization_type == "Histogram":
        render_chart_safely(render_histogram, processed_df, numeric_columns)
    elif visualization_type == "Line Chart":
        render_chart_safely(render_line_chart, processed_df, numeric_columns)
    elif visualization_type == "Scatter Plot":
        render_chart_safely(render_scatter_plot, processed_df, numeric_columns)
    elif visualization_type == "Area Chart":
        render_chart_safely(render_area_chart, processed_df, numeric_columns)
    elif visualization_type == "Box Plot":
        render_chart_safely(render_box_plot, processed_df, numeric_columns)
    elif visualization_type == "Bubble Chart":
        render_chart_safely(render_bubble_chart, processed_df, numeric_columns)

    if not categorical_columns:
        st.info("This dataset currently has only numeric columns, so category-based groupings may be limited.")

st.divider()

components.html(
    """
    <button
        onclick="window.parent.history.back()"
        style="
            width: 100%;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            background: white;
            color: rgb(49, 51, 63);
            cursor: pointer;
            font-family: sans-serif;
            font-size: 1rem;
            padding: 0.45rem 0.75rem;
        "
    >
        ⬅ Back to Previous Page
    </button>
    """,
    height=46,
)
