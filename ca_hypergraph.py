#!/usr/bin/env python3
"""
Hypergraph analysis of the mitochondrial semantic CA rule table.

Builds a bipartite hypergraph representation where each rule is a hyperedge
connecting its input variables to its output variables. Computes metrics:

- Hyper-degree: number of rules each variable participates in (as input or output)
- Rule interference: Jaccard overlap between rules that share variables
- Centrality measures: which variables gate multiple pathways
- Modularity: clustering of rules into biological modules
"""

import json
import networkx as nx
import itertools
from collections import defaultdict, Counter
import sys
sys.path.insert(0, '.')

try:
    from ca_schema import CA_VAR_ORDER
    STATE_VARS = set(CA_VAR_ORDER)
except ImportError:
    # Fallback if ca_schema not available
    STATE_VARS = {
        "N_healthy", "N_deletion", "ATP", "ROS",
        "NAD", "Senescent_fraction", "Membrane_potential", "N_point"
    }

# Context variables that appear in rule context fields
CONTEXT_VARS = {
    "age_epoch", "rapamycin", "nad_supplement", "senolytic",
    "exercise", "yamanaka", "transplant", "near_transition",
    "cliff_proximity", "tissue_type"
}

def load_rules(path="final_tuned_rules.json"):
    """Load rule table from JSON."""
    with open(path, "r") as f:
        return json.load(f)

def build_bipartite_hypergraph(rules):
    """
    Build bipartite graph G where:
      - State variable nodes: "state:N_healthy", "state:ATP", etc.
      - Context variable nodes: "context:age_epoch", "context:rapamycin", etc.
      - Rule nodes: "rule:deletion_expansion_young"
      - Edges: var -> rule (input), rule -> var (output) with attribute 'role'
      - Context edges: context -> rule with attribute 'role': 'context'
    
    Returns NetworkX DiGraph.
    """
    G = nx.DiGraph()
    
    for rule in rules:
        rule_name = rule["name"]
        rule_node = f"rule:{rule_name}"
        G.add_node(rule_node, type="rule", tier=rule.get("tier", 0))
        
        # Input edges (state variable -> rule)
        for var_name in rule.get("inputs", {}):
            if var_name in STATE_VARS:
                var_node = f"state:{var_name}"
                G.add_node(var_node, type="state")
                G.add_edge(var_node, rule_node, role="input")
            else:
                # Could be a context variable used as input? Should not happen
                var_node = f"context:{var_name}"
                G.add_node(var_node, type="context")
                G.add_edge(var_node, rule_node, role="input")
        
        # Output edges (rule -> state variable)
        for var_name in rule.get("outputs", {}):
            if var_name in STATE_VARS:
                var_node = f"state:{var_name}"
                G.add_node(var_node, type="state")
                G.add_edge(rule_node, var_node, role="output")
            else:
                var_node = f"context:{var_name}"
                G.add_node(var_node, type="context")
                G.add_edge(rule_node, var_node, role="output")
        
        # Context edges (context variable -> rule)
        for ctx_key, ctx_value in rule.get("context", {}).items():
            if ctx_key in CONTEXT_VARS:
                ctx_node = f"context:{ctx_key}"
                G.add_node(ctx_node, type="context")
                G.add_edge(ctx_node, rule_node, role="context")
    
    return G

def compute_hyperdegree(G, node_type="state"):
    """Return dict of variable -> total rule participation count."""
    target_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == node_type]
    hyperdegree = {}
    for var in target_nodes:
        # Count rules where var appears as input, output, or context
        in_rules = set(G.successors(var))   # var -> rule edges
        out_rules = set(G.predecessors(var))  # rule -> var edges
        hyperdegree[var] = len(in_rules | out_rules)
    return hyperdegree

def compute_rule_overlap(G, consider_context=False):
    """
    Compute pairwise Jaccard similarity between rules based on shared variables.
    If consider_context=False, only state variables are considered.
    Returns list of (rule1, rule2, similarity) sorted descending.
    """
    rule_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "rule"]
    overlap = []
    
    for r1, r2 in itertools.combinations(rule_nodes, 2):
        # Variables connected to each rule
        if consider_context:
            vars1 = set(G.successors(r1)) | set(G.predecessors(r1))
            vars2 = set(G.successors(r2)) | set(G.predecessors(r2))
        else:
            # Only state variables
            vars1 = {v for v in (set(G.successors(r1)) | set(G.predecessors(r1)))
                     if G.nodes[v].get("type") == "state"}
            vars2 = {v for v in (set(G.successors(r2)) | set(G.predecessors(r2)))
                     if G.nodes[v].get("type") == "state"}
        
        if vars1 and vars2:
            jaccard = len(vars1 & vars2) / len(vars1 | vars2)
            overlap.append((r1, r2, jaccard))
        else:
            overlap.append((r1, r2, 0.0))
    
    overlap.sort(key=lambda x: x[2], reverse=True)
    return overlap

def compute_variable_centrality(G, node_type="state"):
    """
    Compute betweenness centrality for variables in the bipartite projection.
    
    Projects onto variable nodes of given type: two variables are connected if they appear
    in the same rule. Edge weight = number of rules they co-occur in.
    Returns centrality dict.
    """
    var_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == node_type]
    rule_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "rule"]
    
    # Build variable co-occurrence graph
    var_graph = nx.Graph()
    var_graph.add_nodes_from(var_nodes)
    
    for rule in rule_nodes:
        # Get all variables (of target type) connected to this rule
        vars_in_rule = [v for v in (list(G.successors(rule)) + list(G.predecessors(rule)))
                        if G.nodes[v].get("type") == node_type]
        for v1, v2 in itertools.combinations(vars_in_rule, 2):
            if var_graph.has_edge(v1, v2):
                var_graph[v1][v2]["weight"] += 1
            else:
                var_graph.add_edge(v1, v2, weight=1)
    
    if var_graph.number_of_edges() > 0:
        centrality = nx.betweenness_centrality(var_graph, weight="weight")
    else:
        centrality = {v: 0.0 for v in var_nodes}
    
    return centrality

def find_rule_clusters_louvain(G, consider_context=False):
    """
    Cluster rules using greedy modularity communities (approximate Louvain).
    Returns list of clusters (each cluster is list of rule nodes).
    """
    rule_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "rule"]
    if len(rule_nodes) < 2:
        return [rule_nodes]
    
    # Build rule similarity graph
    rule_graph = nx.Graph()
    rule_graph.add_nodes_from(rule_nodes)
    
    # Add edges weighted by Jaccard similarity
    overlap = compute_rule_overlap(G, consider_context=consider_context)
    for r1, r2, sim in overlap:
        if sim > 0.1:  # threshold
            rule_graph.add_edge(r1, r2, weight=sim)
    
    if rule_graph.number_of_edges() == 0:
        return [rule_nodes]
    
    # Use greedy modularity communities
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(rule_graph, weight="weight")
        clusters = [list(c) for c in communities]
    except ImportError:
        # Fallback: connected components
        clusters = list(nx.connected_components(rule_graph))
        clusters = [list(c) for c in clusters]
    
    return clusters

def identify_hub_variables(G, node_type="state", top_k=10):
    """Return top-k variables by hyperdegree."""
    hyperdegree = compute_hyperdegree(G, node_type=node_type)
    sorted_vars = sorted(hyperdegree.items(), key=lambda x: x[1], reverse=True)
    return sorted_vars[:top_k]

def identify_missing_connections(G, node_type="state"):
    """
    Suggest potentially missing cross-module connections.
    Returns pairs of variables that never co-occur in any rule.
    """
    var_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == node_type]
    rule_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "rule"]
    
    # Count co-occurrences
    var_pairs = defaultdict(int)
    for rule in rule_nodes:
        vars_in_rule = [v for v in (list(G.successors(rule)) + list(G.predecessors(rule)))
                        if G.nodes[v].get("type") == node_type]
        for v1, v2 in itertools.combinations(vars_in_rule, 2):
            var_pairs[(v1, v2)] += 1
    
    # Find variable pairs that never co-occur
    missing = []
    for v1, v2 in itertools.combinations(var_nodes, 2):
        if (v1, v2) not in var_pairs and (v2, v1) not in var_pairs:
            missing.append((v1, v2))
    
    return missing, var_pairs

def analyze_context_influence(G):
    """Analyze how context variables gate rule activation."""
    context_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "context"]
    influence = {}
    for ctx in context_nodes:
        rules = list(G.successors(ctx))  # context -> rule edges
        influence[ctx] = len(rules)
    return influence

def print_hypergraph_report(rules_path="final_tuned_rules.json"):
    """Main reporting function."""
    print("=== Mitochondrial CA Rule Hypergraph Analysis ===\n")
    
    rules = load_rules(rules_path)
    print(f"Loaded {len(rules)} rules from {rules_path}\n")
    
    G = build_bipartite_hypergraph(rules)
    print(f"Hypergraph contains:")
    print(f"  State variable nodes: {len([n for n in G.nodes() if G.nodes[n].get('type') == 'state'])}")
    print(f"  Context variable nodes: {len([n for n in G.nodes() if G.nodes[n].get('type') == 'context'])}")
    print(f"  Rule nodes: {len([n for n in G.nodes() if G.nodes[n]['type'] == 'rule'])}")
    print(f"  Edges: {G.number_of_edges()}\n")
    
    # State variable analysis
    print("=== STATE VARIABLES ===")
    hub_vars = identify_hub_variables(G, node_type="state", top_k=10)
    print("\nHub state variables (by hyperdegree):")
    for var, degree in hub_vars:
        var_name = var.replace("state:", "")
        print(f"  {var_name}: {degree} rules")
    
    centrality = compute_variable_centrality(G, node_type="state")
    if centrality:
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nCentral state variables (betweenness centrality):")
        for var, cent in top_central:
            var_name = var.replace("state:", "")
            print(f"  {var_name}: {cent:.3f}")
    
    # Context variable analysis
    print("\n=== CONTEXT VARIABLES ===")
    context_influence = analyze_context_influence(G)
    if context_influence:
        print("Context variable influence (number of rules gated):")
        for ctx, count in sorted(context_influence.items(), key=lambda x: x[1], reverse=True):
            ctx_name = ctx.replace("context:", "")
            print(f"  {ctx_name}: {count} rules")
    
    # Rule clusters (using only state variables)
    print("\n=== RULE CLUSTERS ===")
    clusters = find_rule_clusters_louvain(G, consider_context=False)
    print(f"Number of rule clusters (modularity): {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1}: {len(cluster)} rules")
        # Show rule names (shortened)
        rule_names = [r.replace("rule:", "") for r in cluster]
        # Group by tier
        tier_counts = Counter()
        for r in cluster:
            tier = G.nodes[r].get("tier", 0)
            tier_counts[tier] += 1
        print(f"  Tiers: {dict(tier_counts)}")
        if len(rule_names) <= 8:
            print(f"  Rules: {', '.join(rule_names[:8])}")
        else:
            print(f"  Sample rules: {', '.join(rule_names[:4])} ...")
    
    # Missing connections (state variables only)
    missing, var_pairs = identify_missing_connections(G, node_type="state")
    print(f"\n=== MISSING CONNECTIONS ===")
    print(f"State variable pairs that never co-occur in same rule: {len(missing)}")
    if missing:
        # Show only biologically interesting pairs (exclude self-pairs)
        interesting = []
        for v1, v2 in missing:
            v1_name = v1.replace("state:", "")
            v2_name = v2.replace("state:", "")
            # Skip pairs that are obviously unrelated?
            interesting.append((v1_name, v2_name))
        
        for v1, v2 in interesting[:10]:
            print(f"  {v1} ↔ {v2}")
        if len(interesting) > 10:
            print(f"  ... and {len(interesting)-10} more")
    
    # Rule interference
    print("\n=== RULE INTERFERENCE ===")
    overlap = compute_rule_overlap(G, consider_context=False)
    high_overlap = [(r1, r2, sim) for r1, r2, sim in overlap if sim > 0.5]
    print(f"Rule pairs with high similarity (>0.5): {len(high_overlap)}")
    if high_overlap:
        for r1, r2, sim in high_overlap[:5]:
            r1_name = r1.replace("rule:", "")
            r2_name = r2.replace("rule:", "")
            print(f"  {r1_name} ↔ {r2_name}: {sim:.3f}")
    
    # Tier distribution
    tier_counts = Counter()
    for rule in rules:
        tier_counts[rule.get("tier", 0)] += 1
    print("\n=== RULE TIER DISTRIBUTION ===")
    for tier in sorted(tier_counts.keys()):
        print(f"  Tier {tier}: {tier_counts[tier]} rules")
    
    # Rule complexity
    in_degree = []
    out_degree = []
    for rule in rules:
        in_degree.append(len(rule.get("inputs", {})))
        out_degree.append(len(rule.get("outputs", {})))
    
    print(f"\n=== RULE COMPLEXITY ===")
    print(f"  Avg inputs: {sum(in_degree)/len(in_degree):.2f}")
    print(f"  Avg outputs: {sum(out_degree)/len(out_degree):.2f}")
    print(f"  Max inputs: {max(in_degree)}")
    print(f"  Max outputs: {max(out_degree)}")
    
    # Summary insights
    print("\n=== KEY INSIGHTS ===")
    # 1. Most connected variable
    top_var, top_deg = hub_vars[0] if hub_vars else (None, 0)
    if top_var:
        var_name = top_var.replace("state:", "")
        print(f"1. {var_name} is the most connected variable ({top_deg} rules), suggesting it's a key regulator.")
    
    # 2. Cluster interpretation
    if len(clusters) > 1:
        print(f"2. Rules naturally separate into {len(clusters)} functional modules.")
        for i, cluster in enumerate(clusters):
            # Determine which variables dominate this cluster
            vars_in_cluster = set()
            for rule_node in cluster:
                vars_in_cluster.update([v for v in (set(G.successors(rule_node)) | set(G.predecessors(rule_node)))
                                        if G.nodes[v].get("type") == "state"])
            var_names = [v.replace("state:", "") for v in vars_in_cluster]
            print(f"   Cluster {i+1}: {', '.join(sorted(var_names)[:5])}")
    else:
        print("2. All rules are highly interconnected (single cluster).")
    
    # 3. Missing connections that might be biologically relevant
    biologically_plausible_missing = [
        ("ATP", "NAD"), ("ROS", "Membrane_potential"), 
        ("Senescent_fraction", "N_point"), ("N_healthy", "ROS")
    ]
    for v1, v2 in biologically_plausible_missing:
        v1_node = f"state:{v1}"
        v2_node = f"state:{v2}"
        if (v1_node, v2_node) in missing or (v2_node, v1_node) in missing:
            print(f"3. Missing connection: {v1} ↔ {v2} might be a biologically relevant interaction not captured in current rules.")
    
    return G

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hypergraph analysis of CA rules")
    parser.add_argument("--rules", default="final_tuned_rules.json", help="Path to rule JSON")
    parser.add_argument("--output", help="Output JSON path (optional)")
    args = parser.parse_args()
    
    G = print_hypergraph_report(args.rules)
    
    if args.output:
        import json
        from collections import defaultdict
        # Build JSON report
        report = {
            "rules_path": args.rules,
            "state_variables": list(STATE_VARS),
            "context_variables": list(CONTEXT_VARS),
            "hub_variables": identify_hub_variables(G, node_type="state", top_k=10),
            "centrality": compute_variable_centrality(G, node_type="state"),
            "missing_connections": identify_missing_connections(G, node_type="state")[0],
            "rule_clusters": find_rule_clusters_louvain(G, consider_context=False),
            "rule_overlap": compute_rule_overlap(G, consider_context=False),
            "context_influence": analyze_context_influence(G),
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to {args.output}")