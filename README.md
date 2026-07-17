# 📈📈 TDA Market Mapper: Bitcoin Topology Analysis 📈📈

This project applies **Topological Data Analysis (TDA)** to financial time series to identify structural market regimes. By leveraging Takens' Time Delay Embedding and the Mapper algorithm via `giotto-tda`, the script transforms 1D Bitcoin log returns into a high-dimensional point cloud to extract hidden geometric structures in price action.

Instead of traditional lagging indicators, this approach classifies market days into three distinct topological states based on their graph connectivity:
*   **Centers (Cores):** Consolidation and normal market noise.
*   **Flares (Branches):** Directional trends and momentum branches.
*   **Outliers (Anomalies):** Extreme volatility and market shocks.

## Key Features

*   **Time Delay Embedding:** Converts standard 1D log returns into a multi-dimensional point cloud using a configurable delay ($\tau=1$) and embedding dimension ($d=7$).
*   **Custom Mapper Pipeline:** Utilizes Principal Component Analysis (PCA) for projection, a 1D Cubical Cover, and First Simple Gap clustering.
*   **Graph-Theoretic State Classification:** Uses `iGraph` network metrics (connected component sizes and $k$-coreness) to automatically classify the Mapper graph nodes.
*   **Chronological Mapping:** Maps the topological graph states back onto the original chronological price chart for visual comparison.
*   **Rich Visualization:** Generates interactive 2D graph projections and a dual-panel Matplotlib dashboard comparing discrete TDA states against continuous PCA variance.
*   **Automated Data Extraction:** Compiles node-level topological metrics and their corresponding dates into a clean CSV for downstream quantitative research.

## Prerequisites

Ensure you have Python 3.8+ installed. The project relies heavily on `giotto-tda` for topological computations.

```bash
pip install pandas numpy matplotlib scikit-learn giotto-tda