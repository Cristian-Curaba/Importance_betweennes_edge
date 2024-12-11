import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def read_graph_from_excel(filename: str, sheet_name=0) -> nx.Graph:
    df = pd.read_excel(filename, sheet_name=sheet_name)
    
    required_cols = {'Edge', 'Source', 'Target', 'Weight'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"The input Excel must contain columns: {required_cols}")

    G = nx.Graph()
    for _, row in df.iterrows():
        s = row['Source']
        t = row['Target']
        w = row['Weight']
        G.add_edge(s, t, weight=w)
    return G

def read_node_importance(filename: str, sheet_name=0) -> dict:
    df = pd.read_excel(filename, sheet_name=sheet_name)
    required_cols = {'Numero nodi', 'Importance'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"The input Excel must contain columns: {required_cols}")

    R = {}
    for _, row in df.iterrows():
        node = row['Numero nodi']
        importance = row['Importance']
        R[node] = importance
    return R

def compute_custom_edge_betweenness(G: nx.Graph, R: dict) -> dict:
    edge_bet = {tuple(sorted(e)): 0.0 for e in G.edges()}
    nodes = sorted(G.nodes())

    for i, s in enumerate(nodes):
        dist, _ = nx.single_source_dijkstra(G, s, weight='weight')
        for j, t in enumerate(nodes):
            if j <= i:
                continue
            if t not in dist or dist[t] == float('inf'):
                continue

            all_paths = list(nx.all_shortest_paths(G, s, t, weight='weight'))
            count_paths = len(all_paths)
            importance_factor = R[s] * R[t]
            contrib_per_path = importance_factor / count_paths

            for path in all_paths:
                for k in range(len(path) - 1):
                    edge = tuple(sorted((path[k], path[k+1])))
                    edge_bet[edge] += contrib_per_path

    return edge_bet

def main():
    # Input files
    graph_file = "paper_graph.xlsx"
    importance_file = "paper_node_importance.xlsx"

    # Read graph and importance
    G = read_graph_from_excel(graph_file)
    R = read_node_importance(importance_file)

    # Compute built-in weighted edge betweenness
    built_in_betweenness = nx.edge_betweenness_centrality(G, normalized=False, weight='weight')

    # Compute custom betweenness with R=1 for testing
    R_test = {node: 1.0 for node in G.nodes()}
    custom_betweenness_R1 = compute_custom_edge_betweenness(G, R_test)

    # Compare differences
    differences_R1 = {}
    tol = 1e-9
    for e in G.edges():
        diff = built_in_betweenness[e] - custom_betweenness_R1[tuple(sorted(e))]
        differences_R1[e] = diff

    all_close = all(abs(d) < tol for d in differences_R1.values())
    if all_close:
        print("\nThe built-in and custom betweenness results match (within numerical tolerance) for R_s=1.")
    else:
        print("\nThe built-in and custom betweenness results differ for some edges.")

    # Compute custom betweenness with actual R
    custom_betweenness_R_actual = compute_custom_edge_betweenness(G, R)

    # Prepare data for Excel output
    edges = list(G.edges())
    data = []
    for e in edges:
        sorted_e = tuple(sorted(e))
        bi_val = built_in_betweenness[e]
        custom_val_R1 = custom_betweenness_R1[sorted_e]
        diff_R1 = differences_R1[e]
        custom_val_R = custom_betweenness_R_actual[sorted_e]

        data.append((str(sorted_e), bi_val, custom_val_R1, diff_R1, custom_val_R))

    df = pd.DataFrame(data, columns=[
        "Edge", 
        "Built-in Betweenness", 
        "Custom Betweenness (R=1)", 
        "Difference (Built-in - Custom R=1)", 
        "Custom Betweenness (R=actual)"
    ])

    # Round to 2 decimal places
    df = df.round(2)

    # Save results to Excel
    output_file = "comparison_results.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nComparison results saved to '{output_file}'")

    # Now create the graph visualization with custom betweenness (R=actual)
    bet_values = [custom_betweenness_R_actual[tuple(sorted(e))] for e in edges]
    vmin = min(bet_values)
    vmax = max(bet_values)
    cmap = plt.cm.Blues
    norm = Normalize(vmin=vmin, vmax=vmax)
    edge_colors = [cmap(norm(val)) for val in bet_values]

    # Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=500, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='black', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, ax=ax)

    # Create colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Custom Edge Betweenness (R=actual)", rotation=90)

    ax.set_title("Graph with Edges Shaded by Custom Betweenness (R=actual)", fontsize=14)
    ax.axis('off')

    plt.savefig("graph_custom_betweenness.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
