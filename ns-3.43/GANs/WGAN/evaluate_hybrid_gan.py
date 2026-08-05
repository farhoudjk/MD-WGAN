#!/usr/bin/env python3
# evaluate_hybrid_gan.py - Integration with hybrid GAN evaluation
import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import logging

# Import the evaluation metrics
try:
    from gan_evaluation_metrics import TabularGANEvaluator, evaluate_gan_performance
except ImportError:
    print("Error: Cannot import gan_evaluation_metrics.py")
    print("Please ensure gan_evaluation_metrics.py is in the same directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridGANEvaluationManager:
    """Manager for evaluating hybrid GAN performance across all nodes"""
    
    def __init__(self, base_dir="/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN"):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "results")
        self.datasets_dir = os.path.join(base_dir, "datasets")
        self.evaluation_dir = os.path.join(base_dir, "evaluation")
        
        # Create evaluation directory
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # Define vehicular IoT column types
        self.vehicular_columns = {
            'categorical': ['weather_condition', 'road_type', 'day_of_week', 'rush_hour', 'weekend'],
            'continuous': ['vehicle_speed', 'x_coord', 'y_coord', 'signal_strength', 'packet_loss_rate',
                          'traffic_density', 'time_of_day', 'message_frequency', 'message_size',
                          'distance_from_center']
        }
        
    def find_latest_generated_samples(self, node_id):
        """Find the latest generated samples for a node"""
        node_dir = os.path.join(self.results_dir, f"hybrid_node_{node_id}")
        if not os.path.exists(node_dir):
            return None
            
        # Look for generated samples files
        sample_files = []
        for file in os.listdir(node_dir):
            if file.startswith("generated_samples_iter_") and file.endswith(".csv"):
                iteration = int(file.split("_")[-1].split(".")[0])
                sample_files.append((iteration, os.path.join(node_dir, file)))
        
        if sample_files:
            # Return the latest iteration
            sample_files.sort(key=lambda x: x[0], reverse=True)
            return sample_files[0][1]  # Return file path
        
        return None
    
    def get_real_data_path(self, node_id):
        """Get the real data path for a node"""
        # Try hybrid node dataset first, then discriminator dataset
        paths_to_try = [
            os.path.join(self.datasets_dir, f"hybrid_node_{node_id}", "data.csv"),
            os.path.join(self.datasets_dir, f"discriminator_{node_id}", "data.csv")
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                return path
        
        return None
    
    def evaluate_single_node(self, node_id, save_individual_results=True):
        """Evaluate a single hybrid node"""
        print(f"Evaluating hybrid node {node_id}...")
        
        # Get data paths
        real_data_path = self.get_real_data_path(node_id)
        synth_data_path = self.find_latest_generated_samples(node_id)
        
        if not real_data_path:
            logging.warning(f"No real data found for node {node_id}")
            return None
            
        if not synth_data_path:
            logging.warning(f"No synthetic data found for node {node_id}")
            return None
        
        try:
            # Load data
            real_data = pd.read_csv(real_data_path)
            synth_data = pd.read_csv(synth_data_path)
            
            # Identify categorical columns (those that exist in our data)
            categorical_cols = [col for col in self.vehicular_columns['categorical'] 
                              if col in real_data.columns]
            
            # Initialize evaluator
            evaluator = TabularGANEvaluator(
                real_data=real_data,
                synthetic_data=synth_data,
                categorical_columns=categorical_cols,
                target_column=None  # No supervised evaluation for now
            )
            
            # Compute metrics
            results = evaluator.compute_all_metrics()
            
            # Add node information
            results['node_info'] = {
                'node_id': node_id,
                'real_data_shape': real_data.shape,
                'synthetic_data_shape': synth_data.shape,
                'real_data_path': real_data_path,
                'synthetic_data_path': synth_data_path,
                'categorical_columns': categorical_cols,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            if save_individual_results:
                # Save individual results
                node_eval_dir = os.path.join(self.evaluation_dir, f"node_{node_id}")
                os.makedirs(node_eval_dir, exist_ok=True)
                
                # Save detailed metrics
                with open(os.path.join(node_eval_dir, "detailed_metrics.json"), 'w') as f:
                    json.dump(self.convert_numpy_types(results), f, indent=2)
                
                # Generate and save report
                report = evaluator.generate_report()
                with open(os.path.join(node_eval_dir, "evaluation_report.txt"), 'w') as f:
                    f.write(report)
                
                # Generate and save plots
                try:
                    evaluator.plot_evaluation_results(
                        save_path=os.path.join(node_eval_dir, "evaluation_plots.png")
                    )
                except Exception as e:
                    logging.warning(f"Could not generate plots for node {node_id}: {e}")
            
            logging.info(f"Successfully evaluated node {node_id}")
            return results
            
        except Exception as e:
            logging.error(f"Error evaluating node {node_id}: {e}")
            return None
    
    def evaluate_all_nodes(self, max_nodes=20):
        """Evaluate all hybrid nodes"""
        print(f"Evaluating up to {max_nodes} hybrid nodes...")
        
        all_results = {}
        successful_evaluations = 0
        
        for node_id in range(1, max_nodes + 1):
            results = self.evaluate_single_node(node_id, save_individual_results=True)
            if results is not None:
                all_results[node_id] = results
                successful_evaluations += 1
        
        print(f"Successfully evaluated {successful_evaluations} out of {max_nodes} nodes")
        
        if successful_evaluations > 0:
            # Generate aggregate analysis
            self.generate_aggregate_analysis(all_results)
        
        return all_results
    
    def generate_aggregate_analysis(self, all_results):
        """Generate aggregate analysis across all nodes"""
        print("Generating aggregate analysis...")
        
        # Collect metrics from all nodes
        aggregate_metrics = {
            'overall_scores': [],
            'statistical_similarity': [],
            'distribution_comparison': [],
            'privacy_diversity': [],
            'node_performances': []
        }
        
        for node_id, results in all_results.items():
            if 'overall_scores' in results:
                scores = results['overall_scores'].copy()
                scores['node_id'] = node_id
                aggregate_metrics['overall_scores'].append(scores)
            
            # Collect key metrics for comparison
            node_perf = {'node_id': node_id}
            
            # Statistical metrics
            if 'statistical_similarity' in results:
                stat = results['statistical_similarity']
                node_perf['wasserstein_distance'] = stat.get('wasserstein_distance', {}).get('mean', np.inf)
                node_perf['ks_pvalue'] = stat.get('kolmogorov_smirnov', {}).get('mean_pvalue', 0)
                node_perf['js_divergence'] = stat.get('jensen_shannon_divergence', {}).get('mean', np.inf)
            
            # Distribution metrics
            if 'distribution_comparison' in results:
                dist = results['distribution_comparison']
                node_perf['mean_error'] = dist.get('mean_std_comparison', {}).get('mean_absolute_error', np.inf)
                node_perf['correlation_error'] = dist.get('correlation_comparison', {}).get('mean_absolute_error', np.inf)
            
            # Privacy metrics
            if 'privacy_diversity' in results:
                privacy = results['privacy_diversity']
                node_perf['privacy_score'] = privacy.get('privacy', {}).get('nearest_neighbor_ratio', 0)
                node_perf['coverage'] = privacy.get('coverage', {}).get('coverage_ratio', 0)
                node_perf['diversity'] = privacy.get('diversity', {}).get('diversity_ratio', 0)
            
            aggregate_metrics['node_performances'].append(node_perf)
        
        # Create aggregate DataFrame
        df_performance = pd.DataFrame(aggregate_metrics['node_performances'])
        
        if not df_performance.empty:
            # Generate aggregate statistics
            aggregate_stats = self.calculate_aggregate_statistics(df_performance)
            
            # Generate comparative analysis
            self.generate_comparative_analysis(df_performance, aggregate_stats)
            
            # Save aggregate results
            self.save_aggregate_results(df_performance, aggregate_stats, all_results)
    
    def calculate_aggregate_statistics(self, df_performance):
        """Calculate aggregate statistics across all nodes"""
        stats = {}
        
        numeric_columns = df_performance.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'node_id']
        
        for col in numeric_columns:
            col_data = df_performance[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(col_data) > 0:
                try:
                    stats[col] = {
                        'mean': float(col_data.mean()) if not np.isnan(col_data.mean()) else float('inf'),
                        'std': float(col_data.std()) if not np.isnan(col_data.std()) else 0.0,
                        'min': float(col_data.min()) if not np.isnan(col_data.min()) else float('inf'),
                        'max': float(col_data.max()) if not np.isnan(col_data.max()) else float('-inf'),
                        'median': float(col_data.median()) if not np.isnan(col_data.median()) else float('inf'),
                        'q25': float(col_data.quantile(0.25)) if not np.isnan(col_data.quantile(0.25)) else float('inf'),
                        'q75': float(col_data.quantile(0.75)) if not np.isnan(col_data.quantile(0.75)) else float('inf')
                    }
                except Exception as e:
                    # If calculation fails, use default values
                    stats[col] = {
                        'mean': float('inf'),
                        'std': 0.0,
                        'min': float('inf'),
                        'max': float('-inf'),
                        'median': float('inf'),
                        'q25': float('inf'),
                        'q75': float('inf')
                    }
            else:
                # No valid data for this column
                stats[col] = {
                    'mean': float('inf'),
                    'std': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'median': float('inf'),
                    'q25': float('inf'),
                    'q75': float('inf')
                }
        
        return stats
    
    def generate_comparative_analysis(self, df_performance, aggregate_stats):
        """Generate comparative analysis plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create comparative plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Hybrid GAN Performance Comparison Across Nodes', fontsize=16, fontweight='bold')
        
        # 1. Overall Quality Scores
        if 'wasserstein_distance' in df_performance.columns:
            ax = axes[0, 0]
            # Convert to quality score (lower distance = higher quality)
            quality_scores = 1 / (1 + df_performance['wasserstein_distance'].replace([np.inf], 10))
            ax.bar(df_performance['node_id'], quality_scores)
            ax.set_title('Statistical Quality by Node\n(Wasserstein Distance)')
            ax.set_xlabel('Node ID')
            ax.set_ylabel('Quality Score (0-1)')
            ax.grid(True, alpha=0.3)
        
        # 2. Privacy vs Utility Trade-off
        if 'privacy_score' in df_performance.columns and 'wasserstein_distance' in df_performance.columns:
            ax = axes[0, 1]
            utility = 1 / (1 + df_performance['wasserstein_distance'].replace([np.inf], 10))
            privacy = df_performance['privacy_score'].fillna(0)
            
            scatter = ax.scatter(privacy, utility, c=df_performance['node_id'], 
                               cmap='viridis', s=60, alpha=0.8)
            ax.set_xlabel('Privacy Score')
            ax.set_ylabel('Utility Score')
            ax.set_title('Privacy vs Utility Trade-off')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Node ID')
        
        # 3. Distribution Quality Heatmap
        if len(df_performance) > 1:
            ax = axes[1, 0]
            
            # Select key metrics for heatmap
            heatmap_cols = ['wasserstein_distance', 'ks_pvalue', 'mean_error', 'privacy_score']
            heatmap_cols = [col for col in heatmap_cols if col in df_performance.columns]
            
            if heatmap_cols:
                heatmap_data = df_performance[['node_id'] + heatmap_cols].set_index('node_id')
                
                # Normalize data for better visualization
                heatmap_normalized = heatmap_data.copy()
                for col in heatmap_data.columns:
                    col_data = heatmap_data[col].replace([np.inf, -np.inf], np.nan)
                    if col in ['wasserstein_distance', 'mean_error']:
                        # Lower is better - invert
                        heatmap_normalized[col] = 1 / (1 + col_data)
                    else:
                        # Higher is better - normalize to 0-1
                        heatmap_normalized[col] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                
                sns.heatmap(heatmap_normalized.T, annot=True, fmt='.3f', 
                           cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Quality Score'})
                ax.set_title('Quality Metrics Heatmap by Node')
                ax.set_ylabel('Metrics')
        
        # 4. Performance Distribution
        ax = axes[1, 1]
        if 'wasserstein_distance' in df_performance.columns:
            distances = df_performance['wasserstein_distance'].replace([np.inf], df_performance['wasserstein_distance'].replace([np.inf], np.nan).max())
            distances = distances.dropna()
            
            if len(distances) > 0:
                ax.hist(distances, bins=min(len(distances), 15), alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(distances.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {distances.mean():.4f}')
                ax.axvline(distances.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {distances.median():.4f}')
                ax.set_title('Wasserstein Distance Distribution')
                ax.set_xlabel('Wasserstein Distance')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 5. Node Ranking
        ax = axes[2, 0]
        if 'wasserstein_distance' in df_performance.columns:
            # Rank nodes by overall performance (lower Wasserstein distance = better)
            df_ranked = df_performance.sort_values('wasserstein_distance').reset_index(drop=True)
            df_ranked = df_ranked.head(10)  # Top 10 nodes
            
            colors = ['gold', 'silver', '#CD7F32'] + ['skyblue'] * 7  # Gold, silver, bronze, then blue
            
            bars = ax.bar(range(len(df_ranked)), df_ranked['wasserstein_distance'], 
                         color=colors[:len(df_ranked)])
            ax.set_title('Top 10 Performing Nodes\n(Lower is Better)')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Wasserstein Distance')
            ax.set_xticks(range(len(df_ranked)))
            ax.set_xticklabels([f'#{i+1}\n(Node {int(nid)})' for i, nid in enumerate(df_ranked['node_id'])])
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, df_ranked['wasserstein_distance']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 6. Correlation Analysis
        ax = axes[2, 1]
        numeric_cols = df_performance.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'node_id']
        
        if len(numeric_cols) > 1:
            corr_data = df_performance[numeric_cols].replace([np.inf, -np.inf], np.nan).corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_data, dtype=bool))
            
            sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdBu_r', center=0, ax=ax,
                       square=True, cbar_kws={'label': 'Correlation'})
            ax.set_title('Metric Correlations')
        
        plt.tight_layout()
        
        # Save comparative analysis
        comparison_path = os.path.join(self.evaluation_dir, "comparative_analysis.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparative analysis saved to {comparison_path}")
    
    def save_aggregate_results(self, df_performance, aggregate_stats, all_results):
        """Save aggregate results and generate summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save performance DataFrame
        df_performance.to_csv(os.path.join(self.evaluation_dir, f"node_performance_summary_{timestamp}.csv"), index=False)
        
        # Save aggregate statistics
        with open(os.path.join(self.evaluation_dir, f"aggregate_statistics_{timestamp}.json"), 'w') as f:
            json.dump(aggregate_stats, f, indent=2)
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(df_performance, aggregate_stats)
        
        with open(os.path.join(self.evaluation_dir, f"comprehensive_report_{timestamp}.md"), 'w') as f:
            f.write(report)
        
        print(f"Aggregate results saved with timestamp {timestamp}")
    
    def generate_comprehensive_report(self, df_performance, aggregate_stats):
        """Generate a comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Helper function to safely format numbers
        def safe_format(value, decimals=4):
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                return f"{value:.{decimals}f}"
            return "N/A"
        
        report = f"""# Hybrid GAN Comprehensive Evaluation Report
Generated: {timestamp}

## Executive Summary
This report provides a comprehensive evaluation of the Hybrid GAN performance across all nodes in the vehicular IoT network simulation.

### Key Findings
- **Total Nodes Evaluated**: {len(df_performance)}
- **Best Performing Node**: Node {int(df_performance.loc[df_performance['wasserstein_distance'].idxmin(), 'node_id']) if 'wasserstein_distance' in df_performance.columns and not df_performance['wasserstein_distance'].isna().all() else 'N/A'}
- **Average Wasserstein Distance**: {safe_format(aggregate_stats.get('wasserstein_distance', {}).get('mean', float('inf')), 6)}
- **Overall Quality Assessment**: {'Good' if isinstance(aggregate_stats.get('wasserstein_distance', {}).get('mean'), (int, float)) and aggregate_stats.get('wasserstein_distance', {}).get('mean') < 0.1 else 'Moderate' if isinstance(aggregate_stats.get('wasserstein_distance', {}).get('mean'), (int, float)) and aggregate_stats.get('wasserstein_distance', {}).get('mean') < 0.5 else 'Needs Improvement'}

## Statistical Quality Metrics

### Wasserstein Distance (Lower is Better)
"""
        
        if 'wasserstein_distance' in aggregate_stats:
            wd_stats = aggregate_stats['wasserstein_distance']
            report += f"""
- **Mean**: {wd_stats['mean']:.6f if isinstance(wd_stats['mean'], (int, float)) and not np.isnan(wd_stats['mean']) and not np.isinf(wd_stats['mean']) else 'N/A'}
- **Median**: {wd_stats['median']:.6f if isinstance(wd_stats['median'], (int, float)) and not np.isnan(wd_stats['median']) and not np.isinf(wd_stats['median']) else 'N/A'}
- **Standard Deviation**: {wd_stats['std']:.6f if isinstance(wd_stats['std'], (int, float)) and not np.isnan(wd_stats['std']) and not np.isinf(wd_stats['std']) else 'N/A'}
- **Range**: {wd_stats['min']:.6f if isinstance(wd_stats['min'], (int, float)) and not np.isnan(wd_stats['min']) and not np.isinf(wd_stats['min']) else 'N/A'} - {wd_stats['max']:.6f if isinstance(wd_stats['max'], (int, float)) and not np.isnan(wd_stats['max']) and not np.isinf(wd_stats['max']) else 'N/A'}
"""
        
        if 'ks_pvalue' in aggregate_stats:
            ks_stats = aggregate_stats['ks_pvalue']
            report += f"""
### Kolmogorov-Smirnov Test p-values (Higher is Better)
- **Mean**: {safe_format(ks_stats['mean'], 6)}
- **Median**: {safe_format(ks_stats['median'], 6)}
- **Percentage > 0.05**: {safe_format((df_performance['ks_pvalue'] > 0.05).mean() * 100 if 'ks_pvalue' in df_performance.columns else 0, 1)}%
"""
        
        # Performance rankings
        report += "\n## Node Performance Rankings\n"
        if 'wasserstein_distance' in df_performance.columns:
            top_nodes = df_performance.nsmallest(5, 'wasserstein_distance')
            report += "### Top 5 Performing Nodes (by Wasserstein Distance)\n"
            for i, (_, row) in enumerate(top_nodes.iterrows(), 1):
                report += f"{i}. Node {int(row['node_id'])}: {row['wasserstein_distance']:.6f}\n"
        
        # Privacy and diversity analysis
        if 'privacy_score' in aggregate_stats:
            privacy_stats = aggregate_stats['privacy_score']
            report += f"""
## Privacy and Diversity Analysis

### Privacy Scores
- **Mean Privacy Score**: {safe_format(privacy_stats['mean'], 4)}
- **Privacy Assessment**: {'Good' if isinstance(privacy_stats['mean'], (int, float)) and privacy_stats['mean'] > 1.5 else 'Moderate' if isinstance(privacy_stats['mean'], (int, float)) and privacy_stats['mean'] > 1.0 else 'Needs Improvement'}
"""
        
        if 'coverage' in aggregate_stats:
            coverage_stats = aggregate_stats['coverage']
            report += f"""
### Coverage Analysis
- **Mean Coverage**: {safe_format(coverage_stats['mean'], 4)}
- **Coverage Quality**: {'Excellent' if isinstance(coverage_stats['mean'], (int, float)) and coverage_stats['mean'] > 0.8 else 'Good' if isinstance(coverage_stats['mean'], (int, float)) and coverage_stats['mean'] > 0.6 else 'Moderate' if isinstance(coverage_stats['mean'], (int, float)) and coverage_stats['mean'] > 0.4 else 'Poor'}
"""
        
        # Recommendations
        report += "\n## Recommendations\n"
        
        if 'wasserstein_distance' in aggregate_stats:
            mean_wd = aggregate_stats['wasserstein_distance']['mean']
            if isinstance(mean_wd, (int, float)) and not np.isnan(mean_wd) and not np.isinf(mean_wd):
                if mean_wd > 0.5:
                    report += "- **High Priority**: Consider adjusting GAN architecture or training parameters to improve statistical similarity\n"
                elif mean_wd > 0.1:
                    report += "- **Medium Priority**: Fine-tune hyperparameters for better distribution matching\n"
                else:
                    report += "- **Excellent**: Current model performance is very good\n"
            else:
                report += "- **Data Quality Issue**: Some nodes have problematic data that prevented proper evaluation\n"
        
        if 'privacy_score' in aggregate_stats:
            mean_privacy = aggregate_stats['privacy_score']['mean']
            if isinstance(mean_privacy, (int, float)) and not np.isnan(mean_privacy) and not np.isinf(mean_privacy):
                if mean_privacy < 1.0:
                    report += "- **Privacy Concern**: Consider increasing noise or improving diversity to enhance privacy\n"
                elif mean_privacy > 3.0:
                    report += "- **Potential Utility Loss**: Very high privacy may indicate reduced utility\n"
            else:
                report += "- **Privacy Evaluation Limited**: Insufficient data for comprehensive privacy analysis\n"
        
        # Technical details
        report += f"""
## Technical Details

### Evaluation Methodology
- **Metrics Used**: Wasserstein Distance, Kolmogorov-Smirnov Test, Jensen-Shannon Divergence
- **Privacy Metrics**: Nearest Neighbor Distance Ratio, Coverage Analysis
- **Data Types**: Vehicular IoT tabular data with mixed continuous and categorical features

### Data Characteristics
- **Node Configuration**: Hybrid GAN (combined generator/discriminator)
- **Dataset Features**: Vehicle speed, location, network metrics, traffic conditions
- **Evaluation Scope**: Cross-node performance comparison

### Quality Thresholds
- **Wasserstein Distance**: < 0.1 (Excellent), 0.1-0.5 (Good), > 0.5 (Needs Improvement)
- **KS Test p-value**: > 0.05 (Statistically Similar)
- **Privacy Score**: > 1.5 (Good Privacy), 1.0-1.5 (Moderate), < 1.0 (Privacy Risk)

---
*Report generated by Hybrid GAN Evaluation System*
*Base Directory: {self.base_dir}*
"""
        
        return report
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def run_evaluation_workflow(self, max_nodes=20):
        """Run the complete evaluation workflow"""
        print("=" * 60)
        print("HYBRID GAN COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Check if required directories exist
        if not os.path.exists(self.results_dir):
            print(f"Error: Results directory not found: {self.results_dir}")
            print("Please run the simulation first.")
            return
        
        if not os.path.exists(self.datasets_dir):
            print(f"Error: Datasets directory not found: {self.datasets_dir}")
            print("Please generate datasets first.")
            return
        
        # Run evaluation
        all_results = self.evaluate_all_nodes(max_nodes)
        
        if not all_results:
            print("No nodes could be evaluated. Please check that:")
            print("1. The simulation has been run")
            print("2. Generated samples exist")
            print("3. Real datasets are available")
            return
        
        print("=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Results saved in: {self.evaluation_dir}")
        print("Key files:")
        print("- comprehensive_report_*.md: Main evaluation report")
        print("- comparative_analysis.png: Visual comparison plots")
        print("- node_performance_summary_*.csv: Numerical results")
        print("- node_*/: Individual node evaluation details")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid GAN Performance')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN',
                       help='Base directory for GAN simulation')
    parser.add_argument('--max_nodes', type=int, default=20,
                       help='Maximum number of nodes to evaluate')
    parser.add_argument('--single_node', type=int, default=None,
                       help='Evaluate only a specific node')
    
    args = parser.parse_args()
    
    # Initialize evaluation manager
    evaluator = HybridGANEvaluationManager(args.base_dir)
    
    if args.single_node is not None:
        # Evaluate single node
        print(f"Evaluating single node: {args.single_node}")
        results = evaluator.evaluate_single_node(args.single_node)
        if results:
            print("Evaluation completed successfully!")
        else:
            print("Evaluation failed!")
    else:
        # Run complete evaluation workflow
        evaluator.run_evaluation_workflow(args.max_nodes)

if __name__ == "__main__":
    main()