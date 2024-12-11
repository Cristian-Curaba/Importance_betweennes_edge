import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def read_graph_from_excel(filename: str, sheet_name: str = 0) -> nx.Graph:
    df = pd.read_excel(filename, sheet_name=sheet_name)
    
    required_cols = {'Source', 'Target', 'Weight'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"The input Excel must contain columns: {required_cols}")
    
    # Convert weight from comma-decimal to float
    df['Weight'] = df['Weight'].astype(str).str.replace(',', '.')
    df['Weight'] = df['Weight'].astype(float)
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
    
    return G

def compute_custom_edge_betweenness(G: nx.Graph, R: dict) -> dict:
    """
    Compute the custom edge betweenness given by:
    sum_{s < t} [ f(R_s, R_t) * (sum over all shortest paths: (1/σ(s,t)) * δ(e in path)) ]

    where f(R_s, R_t) = R_s * R_t, and σ(s,t) is the number of shortest paths from s to t.

    This version accounts for multiple shortest paths similarly to NetworkX's 
    edge_betweenness_centrality.

    Parameters
    ----------
    G : nx.Graph
        The graph on which to compute edge betweenness.
    R : dict
        A dictionary {node: R_node}, representing the importance value for each node.

    Returns
    -------
    dict
        A dictionary of edge betweenness values keyed by edges in the form (u,v) with u < v.
    """
    edge_bet = {tuple(sorted(e)): 0.0 for e in G.edges()}
    nodes = list(G.nodes())

    for i, s in enumerate(nodes):
        dist, _ = nx.single_source_dijkstra(G, s, weight='weight')
        for j, t in enumerate(nodes):
            if j <= i:
                continue  # Only consider each pair once (s < t)
            # If there's no path from s to t, skip
            if t not in dist or dist[t] == float('inf'):
                continue

            # Get all shortest paths from s to t
            all_paths = list(nx.all_shortest_paths(G, s, t, weight='weight'))
            count_paths = len(all_paths)

            # Compute f(R_s, R_t) = R_s * R_t
            importance_factor = R[s] * R[t]

            # Distribute importance among all shortest paths
            for path in all_paths:
                # Each shortest path contributes (R_s * R_t)/σ(s,t) for the edges it uses
                contrib = importance_factor / count_paths
                for k in range(len(path) - 1):
                    edge = tuple(sorted((path[k], path[k+1])))
                    edge_bet[edge] += contrib

    return edge_bet

def main():
    filename = "Raspano.xlsx"
    G = read_graph_from_excel(filename)
    
    # Set R_s = 1 for all nodes
    R = {node: 1 for node in G.nodes()}
    
    # Compute built-in weighted edge betweenness (non-normalized)
    built_in_betweenness = nx.edge_betweenness_centrality(G, normalized=False, weight='weight')
    
    # Compute custom betweenness
    custom_betweenness = compute_custom_edge_betweenness(G, R)
    
    # Prepare data for export
    edges = list(G.edges())
    data = []
    for e in edges:
        sorted_e = tuple(sorted(e))
        bi_val = built_in_betweenness[e]  # e is in same form as built_in_betweenness keys
        custom_val = custom_betweenness[sorted_e]
        diff = bi_val - custom_val
        data.append((str(sorted_e), bi_val, custom_val, diff))
    
    df = pd.DataFrame(data, columns=["Edge", "Built-in Betweenness", "Custom Betweenness", "Difference"])
    
    # Save results to Excel
    df.to_excel("comparison_results.xlsx", index=False)
    
    # Create a bar chart to show differences
    # We'll plot Built-in and Custom side by side
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = range(len(edges))
    width = 0.35
    
    ax.bar([x - width/2 for x in x_positions], df["Built-in Betweenness"], width=width, label="Built-in")
    ax.bar([x + width/2 for x in x_positions], df["Custom Betweenness"], width=width, label="Custom")
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df["Edge"], rotation=45, ha="right")
    ax.set_ylabel("Betweenness")
    ax.set_title("Comparison of Built-in vs Custom Edge Betweenness")
    ax.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("betweenness_comparison.png", dpi=300)
    plt.close(fig)
    
    # Print the differences for console reference
    print("\nComparison results saved to 'comparison_results.xlsx' and 'betweenness_comparison.png'")
    print("Differences:\n", df[["Edge", "Difference"]])

if __name__ == "__main__":
    main()
