import pandas as pd
import plotly.graph_objects as go

# Load your CSV
df = pd.read_csv("TSLA_financials_20251001.csv")

# Ensure numeric values
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Select key metrics
key_metrics = [
    "Automotive sales", "Automotive regulatory credits" , "Automotive leasing"
]

filtered_df = df[df["Metric"].isin(key_metrics)]

# Create interactive chart
fig = go.Figure()

for metric in key_metrics:
    subset = filtered_df[filtered_df["Metric"] == metric]
    fig.add_trace(go.Scatter(
        x=subset["Period"],
        y=subset["Value"],
        mode="lines+markers",
        name=metric
    ))

fig.update_layout(
    title="Tesla Financial Performance Over Time",
    xaxis_title="Reporting Period",
    yaxis_title="Value (millions USD)",
    hovermode="x unified",
    xaxis=dict(type="category", tickangle=-45),
    template="plotly_white"
)

fig.show()
