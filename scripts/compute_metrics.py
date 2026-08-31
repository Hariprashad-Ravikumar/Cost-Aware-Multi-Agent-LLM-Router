import pandas as pd
import json
import os
import matplotlib.pyplot as plt

def compute_cost(model_name, input_tokens, output_tokens, pricing_data):
    # Default to 0 if model not found
    if model_name not in pricing_data:
        # Check if we can infer
        for k in pricing_data.keys():
            if k in model_name.lower():
                model_name = k
                break
        else:
            return 0.0
            
    rates = pricing_data[model_name]
    input_cost = (input_tokens / 1_000_000) * rates["input_cost_per_1m"]
    output_cost = (output_tokens / 1_000_000) * rates["output_cost_per_1m"]
    return input_cost + output_cost

def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    config_path = os.path.join(project_root, "config", "pricing.json")
    
    baseline_path = os.path.join(results_dir, "baseline.csv")
    routed_path = os.path.join(results_dir, "routed.csv")
    
    with open(config_path, "r") as f:
        pricing_data = json.load(f)
        
    # ROUTER overhead model cost (hardcoded from architecture for demo)
    router_model = "llama-3.1-8b-instant"
    
    # Process Baseline
    baseline_df = pd.read_csv(baseline_path)
    baseline_df['cost'] = baseline_df.apply(
        lambda row: compute_cost(row['model'], row['input_tokens'], row['output_tokens'], pricing_data), 
        axis=1
    )
    baseline_metrics = {
        "Total Cost ($)": round(baseline_df['cost'].sum(), 4),
        "Average Latency (ms)": round(baseline_df['latency_ms'].mean(), 2),
        "Accuracy (%)": round((baseline_df['correct'].sum() / len(baseline_df)) * 100, 2),
        "Capable Model Calls": len(baseline_df),
        "Cheap Model Calls": 0
    }
    
    # Process Routed
    # Only if it exists (for incremental testing)
    routed_metrics = None
    if os.path.exists(routed_path):
        routed_df = pd.read_csv(routed_path)
        
        # We also need to add the cost of the classifier per row
        def get_routed_cost(row):
            # Cost of the execution model
            exec_cost = compute_cost(row['model'], row['input_tokens'], row['output_tokens'], pricing_data)
            # Cost of the classifier model
            classifier_cost = compute_cost(
                router_model, 
                row.get('classifier_input_tokens', 100), # fallback if missing
                row.get('classifier_output_tokens', 5),
                pricing_data
            )
            return exec_cost + classifier_cost

        routed_df['cost'] = routed_df.apply(get_routed_cost, axis=1)
        routed_metrics = {
            "Total Cost ($)": round(routed_df['cost'].sum(), 4),
            "Average Latency (ms)": round(routed_df['latency_ms'].mean(), 2),
            "Accuracy (%)": round((routed_df['correct'].sum() / len(routed_df)) * 100, 2),
            "Capable Model Calls": len(routed_df[routed_df['model'].str.contains('gemini', case=False, na=False)]),
            "Cheap Model Calls": len(routed_df[routed_df['model'].str.contains('groq', case=False, na=False)])
        }
    
    # Generate Output Markdown
    table_path = os.path.join(results_dir, "comparison_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("### RouteDemo: Baseline vs Routed Metrics\n\n")
        f.write("| Metric | Baseline (Capable Only) | Routed (Adaptive) |\n")
        f.write("| --- | --- | --- |\n")
        
        metrics_keys = ["Total Cost ($)", "Accuracy (%)", "Average Latency (ms)", "Capable Model Calls", "Cheap Model Calls"]
        for key in metrics_keys:
            val_b = baseline_metrics.get(key, "N/A")
            val_r = routed_metrics.get(key, "N/A") if routed_metrics else "N/A"
            f.write(f"| {key} | {val_b} | {val_r} |\n")
            
    print(f"Metrics table written to {table_path}")
    
    # Generate Chart
    if routed_metrics:
        labels = ['Baseline', 'Routed']
        costs = [baseline_metrics["Total Cost ($)"], routed_metrics["Total Cost ($)"]]
        accuracies = [baseline_metrics["Accuracy (%)"], routed_metrics["Accuracy (%)"]]
        
        x = [0, 1]
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        ax1.set_ylabel('Total Cost ($)', color='tab:red')
        bars1 = ax1.bar([p - width/2 for p in x], costs, width, label='Cost', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='tab:blue')
        bars2 = ax2.bar([p + width/2 for p in x], accuracies, width, label='Accuracy', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        plt.title('Cost vs Accuracy: Baseline vs Routed Pipeline')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        chart_path = os.path.join(results_dir, "comparison_chart.png")
        plt.tight_layout()
        plt.savefig(chart_path)
        print(f"Metrics chart saved to {chart_path}")

if __name__ == "__main__":
    main()
