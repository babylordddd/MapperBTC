import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    plot_static_mapper_graph,
)
from gtda.mapper.cluster import FirstSimpleGap
from gtda.plotting import plot_point_cloud
from gtda.time_series import SingleTakensEmbedding
from sklearn.decomposition import PCA

def load_and_preprocess_data(file_path="BTC_data.csv"):
    """Loads the yfinance dataset, skips metadata headers, and computes log returns."""
    df = pd.read_csv(file_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    return df

def compute_time_delay_embedding(returns, d=7, tau=1):
    """Embeds the 1D time series log returns into a d-dimensional point cloud."""
    print(f"Embedding time series with d={d} and tau={tau}...")
    STE = SingleTakensEmbedding(
        parameters_type="fixed", time_delay=tau, dimension=d
    )
    signal_embedded = STE.fit_transform(returns)
    return signal_embedded

def build_mapper_pipeline():
    """Defines filter functions, intervals, and clusters, compiling the Giotto-TDA pipeline."""
    filter_func = PCA(n_components=2, random_state=42)
    cover = CubicalCover(n_intervals=15, overlap_frac=0.3)
    clusterer = FirstSimpleGap()

    print("Constructing Mapper pipeline...")
    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=False,
        n_jobs=-1,  # Use all available CPU cores
    )
    return pipe, filter_func

def classify_topological_nodes(graph, max_outlier_size=7):
    """Extracts the iGraph topology and partitions nodes into Core, Flare, and Outlier lists."""
    components = graph.connected_components()
    membership = components.membership
    component_sizes = components.sizes()
    coreness = np.array(graph.coreness())

    outlier_nodes = []
    center_nodes = []
    flare_nodes = []

    for node_idx in range(graph.vcount()):
        comp_id = membership[node_idx]
        comp_size = component_sizes[comp_id]

        if comp_size <= max_outlier_size:
            outlier_nodes.append(node_idx)
        else:
            if coreness[node_idx] >= 3:
                center_nodes.append(node_idx)
            else:
                flare_nodes.append(node_idx)

    return (
        np.array(center_nodes),
        np.array(flare_nodes),
        np.array(outlier_nodes),
        membership,
        component_sizes,
        coreness,
    )

def map_topological_states_to_days(
    df, graph, valid_indices, center_nodes, flare_nodes, outlier_nodes
):
    """Maps the structural node assignments back onto the chronological pandas dataframe."""
    df["Topological_State"] = "Unclassified"

    def _apply_state(node_list, state_label):
        for node_idx in node_list:
            embedded_indices = graph.vs[node_idx]["node_elements"]
            df_indices = [valid_indices[i] for i in embedded_indices]
            df.iloc[
                df_indices, df.columns.get_loc("Topological_State")
            ] = state_label

    # Priority routing execution order (Outliers overwrite Centers)
    _apply_state(center_nodes, "Center (Core/Consolidation)")
    _apply_state(flare_nodes, "Flare (Branch/Trend)")
    _apply_state(outlier_nodes, "Outlier (Anomaly)")

    # Generate numeric representation for plotting metrics
    state_mapping = {
        "Unclassified": 0,
        "Center (Core/Consolidation)": 1,
        "Flare (Branch/Trend)": 2,
        "Outlier (Anomaly)": 3,
    }
    df["State_Num"] = df["Topological_State"].map(state_mapping)
    return df

def generate_plots(df, y_periodic_embedded_pca, valid_indices):
    """Generates and showcases all spatial point clouds and dual-axis comparison subplots."""
    # 1. Spatial Embeddings Check
    fig2 = plot_point_cloud(y_periodic_embedded_pca)
    fig2.show()

    # 2. Synchronize PCA continuous signals with baseline array indices
    pc1_values = y_periodic_embedded_pca[:, 0]
    df["PCA_PC1"] = np.nan
    for i, valid_idx in enumerate(valid_indices):
        df.iloc[valid_idx, df.columns.get_loc("PCA_PC1")] = pc1_values[i]

    print("Generating comparison subplots...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 15), sharex=True)

    def _draw_base_line(ax):
        ax.plot(df.index, df["Close"], color="lightgray", alpha=0.5, zorder=1)
        ax.set_ylabel("BTC Close (USD)")
        ax.grid(True, linestyle="--", alpha=0.5)

    # PANEL 1: TDA Mapper Features Discrete Scatter
    _draw_base_line(axes[0])
    mapper_colors = {
        "Center (Core/Consolidation)": "#3498db",
        "Flare (Branch/Trend)": "#f39c12",
        "Outlier (Anomaly)": "#e74c3c",
    }
    for state, color in mapper_colors.items():
        subset = df[df["Topological_State"] == state]
        if not subset.empty:
            axes[0].scatter(
                subset.index,
                subset["Close"],
                color=color,
                label=state,
                s=20,
                zorder=5,
            )
    axes[0].set_title(
        "1. TDA Mapper: Topological Features (Core, Flares, Outliers)",
        fontweight="bold",
    )
    axes[0].legend(loc="upper left")

    # PANEL 2: PCA Continuous Variance Scale
    _draw_base_line(axes[1])
    subset_pca = df.dropna(subset=["PCA_PC1"])
    scatter2 = axes[1].scatter(
        subset_pca.index,
        subset_pca["Close"],
        c=subset_pca["PCA_PC1"],
        cmap="coolwarm",
        s=20,
        zorder=5,
    )
    axes[1].set_title(
        "2. PCA Continuous: Colored by Principal Component 1 (Max Variance)",
        fontweight="bold",
    )
    fig.colorbar(
        scatter2, ax=axes[1], pad=0.01, label="PC1 Score (Momentum/Volatility)"
    )

    plt.xlabel("Date", fontweight="bold")
    plt.tight_layout()
    plt.show()

def export_node_analysis_csv(
    df,
    graph,
    valid_indices,
    membership,
    component_sizes,
    coreness,
    max_outlier_size=7,
    output_path="BTC_Mapper_Nodes_Analysis.csv",
):
    """Compiles individual node topological properties and extracts chronological dates into a clean CSV."""
    print("Extracting Node Data to CSV...")
    node_data = []

    for node_idx in range(graph.vcount()):
        comp_id = membership[node_idx]
        comp_size = component_sizes[comp_id]

        if comp_size <= max_outlier_size:
            state = "Outlier (Anomaly)"
        elif coreness[node_idx] >= 3:
            state = "Center (Core/Consolidation)"
        else:
            state = "Flare (Branch/Trend)"

        embedded_indices = graph.vs[node_idx]["node_elements"]
        df_indices = [valid_indices[i] for i in embedded_indices]

        node_dates = df.index[df_indices]
        date_strs = "|".join(sorted([d.strftime("%Y-%m-%d") for d in node_dates]))

        mean_price = df["Close"].iloc[df_indices].mean()
        mean_return = df["Log_Return"].iloc[df_indices].mean()

        node_data.append(
            {
                "Node_ID": node_idx,
                "Topological_State": state,
                "Component_ID": comp_id,
                "Component_Size": comp_size,
                "Coreness": coreness[node_idx],
                "Node_Size": len(embedded_indices),
                "Mean_Close_USD": round(mean_price, 2),
                "Mean_Log_Return": round(mean_return, 6),
                "Dates_Included": date_strs,
            }
        )

    out_df = pd.DataFrame(node_data)
    out_df.to_csv(output_path, index=False)
    print(f"Extraction Complete! File saved as {output_path}")

def main():
    # --- Configuration Hyperparameters ---
    CONFIG = {"d": 7, "tau": 1, "max_outlier_size": 7}

    # 1. Preprocessing Data
    df = load_and_preprocess_data("BTC_data.csv")
    print(df.head())
    returns = df["Log_Return"].values

    # 2. Delays and Dimensional Embeddings
    signal_embedded = compute_time_delay_embedding(
        returns, d=CONFIG["d"], tau=CONFIG["tau"]
    )
    valid_indices = range((CONFIG["d"] - 1) * CONFIG["tau"], len(df))

    # 3. Compile Pipeline & Project Graph
    pipe, filter_func = build_mapper_pipeline()

    print("Generating mapper graph...")
    color_data = df["Log_Return"].iloc[valid_indices].values
    #takes the start of the day log return
    color_df = pd.DataFrame(color_data, columns=["Average Log return"])

    fig = plot_static_mapper_graph(
        pipe,
        signal_embedded,
        layout_dim=2,
        color_data=color_df,
        node_color_statistic=np.mean,
    )
    fig.show()

    y_periodic_embedded_pca = filter_func.fit_transform(signal_embedded)

    # 4. Topology Deconstruction
    print("Extracting raw graph object...")
    graph = pipe.fit_transform(signal_embedded)

    (
        center_nodes,
        flare_nodes,
        outlier_nodes,
        membership,
        component_sizes,
        coreness,
    ) = classify_topological_nodes(
        graph, max_outlier_size=CONFIG["max_outlier_size"]
    )

    print(
        f"Found {len(center_nodes)} Core nodes, {len(flare_nodes)} Flare nodes, and {len(outlier_nodes)} Outlier nodes."
    )

    # 5. Data Mapping & Formatting
    df = map_topological_states_to_days(
        df, graph, valid_indices, center_nodes, flare_nodes, outlier_nodes
    )

    # 6. Static Output Visualizations
    outlier_dates = df[df["Topological_State"] == "Outlier (Anomaly)"].index
    print(f"\n--- Found {len(outlier_dates)} Outlier Days ---")
    for date in outlier_dates:
        print(date.strftime("%Y-%m-%d"))

    print("Generating classified mapper graph...")
    numeric_states = df.iloc[valid_indices]["State_Num"].values
    color_df_states = pd.DataFrame(
        numeric_states, columns=["State (1=Core, 2=Flare, 3=Outlier)"]
    )

    fig_classified = plot_static_mapper_graph(
        pipe,
        signal_embedded,
        layout_dim=2,
        color_data=color_df_states,
        node_color_statistic=np.max,
    )
    fig_classified.show()

    # 7. Comparison Matplotlib Plotting & Disk Outputs
    generate_plots(df, y_periodic_embedded_pca, valid_indices)
    export_node_analysis_csv(
        df,
        graph,
        valid_indices,
        membership,
        component_sizes,
        coreness,
        max_outlier_size=CONFIG["max_outlier_size"],
    )

if __name__ == "__main__":
    main()