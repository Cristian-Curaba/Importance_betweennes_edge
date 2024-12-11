
# Edge Importance and Betweenness Calculation

## Overview

This project demonstrates how to compute edge betweenness centrality on a weighted graph, incorporating an additional node-based "importance" factor. The project:

- Reads input graph data from an Excel file (`paper_graph.xlsx`).
- Reads node importance values from another Excel file (`paper_node_importance.xlsx`).
- Computes both the standard weighted edge betweenness (using NetworkX) and a custom variant that integrates node importance values into the calculation.
- Compares the results and saves them into `comparison_results.xlsx`.
- Generates a visualization of the graph, shading edges by their custom betweenness values.

## Files

- **paper_graph.xlsx**: Contains the graph’s edges, sources, targets, and weights.
- **paper_node_importance.xlsx**: Contains each node’s importance value.
- **main_importance_betw_edge.py**: Main script that:

  - Reads the graph and importance data.
  - Computes standard and custom (importance-based) edge betweenness.
  - Saves the results to an Excel file.
  - Produces a PNG visualization of the graph with edges colored according to their custom betweenness.
- **comparison_results.xlsx**: Automatically generated file containing:

  - Each edge.
  - The built-in (standard) weighted betweenness.
  - The custom betweenness with R=1 (used for validation).
  - The difference between the built-in and custom (R=1) results.
  - The custom betweenness using actual importance values.
- **graph_custom_betweenness.png**: A visualization of the graph with edges shaded by their custom betweenness values.

## Requirements

- Python 3.7+
- `pip install pandas openpyxl networkx matplotlib`

## Running the Code

1. Activate your virtual environment if you have one:

   ```bash
   source .venv/bin/activate
   ```
2. Run the main script:

   ```bash
   python importance_betw_edge.py
   ```
3. After execution:

   - Check `comparison_results.xlsx` for the numerical results.
   - Check `graph_custom_betweenness.png` for the visual representation.

## Finding an Incongruence in the Paper

The paper from which this data originates seems to contain a discrepancy. Specifically, it highlights a shortest path contributing to betweenness via `(9,11)` and `(11,10)`, but does not properly acknowledge the betweenness that should also be assigned to the edge `(9,10)`. This oversight leads to an incorrect betweenness value reported in the paper.

In our calculations, the edge `(9,10)` is correctly accounted for, ensuring that all shortest paths and their corresponding betweenness contributions are represented accurately.
