# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

class FixedGANEvaluator:
    def __init__(self, real_data_path, generated_data_path, normalization='standard'):
        """
        Initialize the fixed GAN evaluator with proper scale handling
        
        Args:
            real_data_path: Path to real dataset folder
            generated_data_path: Path to generated samples folder
            normalization: 'standard', 'minmax', or 'none'
        """
        self.real_data_path = real_data_path
        self.generated_data_path = generated_data_path
        self.normalization = normalization
        self.real_data = {}
        self.generated_data = {}
        
        # Column mapping
        self.column_mapping = {
            'vehicle_id': 0, 'frame_id': 1, 'total_frames': 2, 'time': 3,
            'position_x': 4, 'position_y': 5, 'global_x': 6, 'global_y': 7,
            'vehicle_length': 8, 'vehicle_width': 9, 'vehicle_class': 10,
            'speed': 11, 'acceleration': 12, 'lane_id': 13,
            'preceding_vehicle_id': 14, 'following_vehicle_id': 15, 'spacing': 16
        }
        
        # FIXED: Use only reasonable-scale features, exclude extreme ones
        self.evaluation_features = [
            'position_x', 'position_y',  # Local coordinates (reasonable scale)
            'speed', 'acceleration',  # Dynamics
            'spacing'  # Following distance
        ]
        
        # Optional: Include normalized versions of extreme-scale features
        self.extreme_features = ['time', 'global_x', 'global_y']  # Handle separately
        
        print("Evaluating features: {}".format(self.evaluation_features))
        print("Normalization method: {}".format(self.normalization))
        
    def load_real_data(self, sample_size=None):
        """Load real data - removed sampling limit for full evaluation"""
        print("Loading real data...")
        
        # Load ALL discriminators for complete evaluation
        for i in range(1, 25):  # Load all 24 discriminators
            folder_name = "discriminator_{}".format(i)
            file_path = os.path.join(self.real_data_path, folder_name, "data.csv")
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Optional sampling only if explicitly requested
                    if sample_size and len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42)
                        print("Loaded {}: sampled {} from {} rows".format(folder_name, sample_size, len(df)))
                    else:
                        print("Loaded {}: {} rows".format(folder_name, len(df)))
                        
                    self.real_data["discriminator_{}".format(i)] = df
                    
                except Exception as e:
                    print("Error loading {}: {}".format(file_path, e))
            else:
                print("File not found: {}".format(file_path))
                
    def load_generated_data(self, max_iterations=None):
        """Load generated data - removed iteration limit for full evaluation"""
        print("\nLoading ALL generated data...")
        
        for gen_i in range(1, 5):  # All generators  # generator
            generator_name = "hybrid_node_{}".format(gen_i)
            generator_path = os.path.join(self.generated_data_path, generator_name)
            
            if not os.path.exists(generator_path):
                print("Generator folder not found: {}".format(generator_path))
                continue
                
            self.generated_data[generator_name] = {}
            loaded_iterations = 0
            
            # Load ALL available iterations (not just every 10th)
            for iter_i in range(1, 501):  # Check up to 500 iterations
                file_name = "generated_samples_iter_{}.csv".format(iter_i)
                file_path = os.path.join(generator_path, file_name)
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Ensure column names are integers
                        if all(isinstance(col, str) for col in df.columns):
                            try:
                                df.columns = [int(col) for col in df.columns]
                            except ValueError:
                                df.columns = range(len(df.columns))
                        
                        self.generated_data[generator_name]["iter_{}".format(iter_i)] = df
                        loaded_iterations += 1
                        
                        # Stop if we've reached the max_iterations limit (if specified)
                        if max_iterations and loaded_iterations >= max_iterations:
                            break
                            
                    except Exception as e:
                        print("Error loading {}: {}".format(file_path, e))
                        
            print("Loaded {}: {} iterations".format(generator_name, loaded_iterations))
            
            # Show sample generated data
            if loaded_iterations > 0:
                first_iter = list(self.generated_data[generator_name].values())[0]
                print("  Sample shape: {}".format(first_iter.shape))
                
                # Show first few values for key features
                for feature in self.evaluation_features[:3]:
                    if feature in self.column_mapping:
                        col_idx = self.column_mapping[feature]
                        if col_idx in first_iter.columns:
                            values = first_iter[col_idx].iloc[:3].tolist()
                            print("  {} (col {}): {}".format(feature, col_idx, values))

    def prepare_data_for_comparison(self, real_df, gen_df, features=None):
        """Prepare and normalize data for fair comparison"""
        if features is None:
            features = self.evaluation_features
        
        # Get available features
        available_real_features = [f for f in features if f in real_df.columns]
        if not available_real_features:
            raise ValueError("No features found in real data")
        
        # Map generated data columns
        gen_column_mapping = {}
        for feature in available_real_features:
            if feature in self.column_mapping:
                gen_col_idx = self.column_mapping[feature]
                if gen_col_idx in gen_df.columns:
                    gen_column_mapping[feature] = gen_col_idx
        
        if not gen_column_mapping:
            raise ValueError("No corresponding columns found in generated data")
        
        # Extract and clean data
        real_subset = real_df[available_real_features].dropna()
        
        gen_renamed_cols = {gen_column_mapping[f]: f for f in gen_column_mapping.keys()}
        gen_subset = gen_df.rename(columns=gen_renamed_cols)[list(gen_column_mapping.keys())].dropna()
        
        common_features = list(set(gen_column_mapping.keys()) & 
                              set(real_subset.columns) & 
                              set(gen_subset.columns))
        
        if not common_features:
            raise ValueError("No common features after mapping")
        
        real_subset = real_subset[common_features]
        gen_subset = gen_subset[common_features]
        
        # Apply normalization
        if self.normalization == 'standard':
            scaler = StandardScaler()
            real_normalized = pd.DataFrame(
                scaler.fit_transform(real_subset),
                columns=real_subset.columns
            )
            gen_normalized = pd.DataFrame(
                scaler.transform(gen_subset),
                columns=gen_subset.columns
            )
        elif self.normalization == 'minmax':
            scaler = MinMaxScaler()
            real_normalized = pd.DataFrame(
                scaler.fit_transform(real_subset),
                columns=real_subset.columns
            )
            gen_normalized = pd.DataFrame(
                scaler.transform(gen_subset),
                columns=gen_subset.columns
            )
        else:  # no normalization
            real_normalized = real_subset
            gen_normalized = gen_subset
        
        return real_normalized, gen_normalized, common_features
    
    def calculate_metrics_normalized(self, real_df, gen_df, method='statistical'):
        """Calculate both RMSE and MAE on normalized data"""
        try:
            real_norm, gen_norm, features = self.prepare_data_for_comparison(real_df, gen_df)
            
            rmse_scores = {}
            mae_scores = {}
            
            if method == 'statistical':
                # Compare statistical properties
                for feature in features:
                    real_stats = np.array([
                        real_norm[feature].mean(),
                        real_norm[feature].std(),
                        real_norm[feature].median(),
                        np.percentile(real_norm[feature], 25),
                        np.percentile(real_norm[feature], 75)
                    ])
                    
                    gen_stats = np.array([
                        gen_norm[feature].mean(),
                        gen_norm[feature].std(),
                        gen_norm[feature].median(),
                        np.percentile(gen_norm[feature], 25),
                        np.percentile(gen_norm[feature], 75)
                    ])
                    
                    rmse_scores[feature] = np.sqrt(np.mean((real_stats - gen_stats) ** 2))
                    mae_scores[feature] = np.mean(np.abs(real_stats - gen_stats))
                    
            else:  # direct comparison
                # Sample-based comparison
                n_samples = min(len(real_norm), len(gen_norm), 1000)
                real_sample = real_norm.sample(n=n_samples, random_state=42)
                gen_sample = gen_norm.sample(n=n_samples, random_state=42)
                
                for feature in features:
                    rmse = np.sqrt(mean_squared_error(real_sample[feature], gen_sample[feature]))
                    mae = mean_absolute_error(real_sample[feature], gen_sample[feature])
                    rmse_scores[feature] = rmse
                    mae_scores[feature] = mae
            
            overall_rmse = np.mean(list(rmse_scores.values()))
            overall_mae = np.mean(list(mae_scores.values()))
            
            return rmse_scores, mae_scores, overall_rmse, overall_mae, features
            
        except Exception as e:
            print("Error in metrics calculation: {}".format(e))
            return None, None, None, None, None

    def evaluate_all_combinations(self):
        """Evaluate all loaded combinations with both RMSE and MAE - FULL EVALUATION"""
        print("\n" + "="*50)
        print("COMPLETE EVALUATION WITH FIXED SCALING (ALL ITERATIONS)")
        print("="*50)
        
        results = {'detailed': {}, 'summary': {}}
        all_rmse_values = []
        all_mae_values = []
        
        total_comparisons = 0
        successful_comparisons = 0
        
        start_time = time.time()
        
        for disc_name, real_df in self.real_data.items():
            results['detailed'][disc_name] = {}
            
            for gen_name, gen_iterations in self.generated_data.items():
                results['detailed'][disc_name][gen_name] = {}
                
                for iter_name, gen_df in gen_iterations.items():
                    total_comparisons += 1
                    
                    metrics_result = self.calculate_metrics_normalized(real_df, gen_df)
                    rmse_scores, mae_scores, overall_rmse, overall_mae, features = metrics_result
                    
                    if rmse_scores is not None:
                        results['detailed'][disc_name][gen_name][iter_name] = {
                            'feature_rmse': rmse_scores,
                            'feature_mae': mae_scores,
                            'overall_rmse': overall_rmse,
                            'overall_mae': overall_mae,
                            'features_used': features
                        }
                        all_rmse_values.append(overall_rmse)
                        all_mae_values.append(overall_mae)
                        successful_comparisons += 1
                    
                    # Progress update every 100 comparisons
                    if total_comparisons % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = total_comparisons / elapsed if elapsed > 0 else 0
                        remaining = (len(self.real_data) * sum(len(gen_iterations) for gen_iterations in self.generated_data.values()) - total_comparisons)
                        eta = remaining / rate if rate > 0 else 0
                        print("Progress: {}/{} comparisons ({:.1f}%), {:.1f} comp/sec, ETA: {:.1f} minutes".format(
                            total_comparisons, 
                            len(self.real_data) * sum(len(gen_iterations) for gen_iterations in self.generated_data.values()),
                            100 * total_comparisons / (len(self.real_data) * sum(len(gen_iterations) for gen_iterations in self.generated_data.values())),
                            rate,
                            eta / 60
                        ))
        
        # Calculate summary statistics
        if all_rmse_values and all_mae_values:
            generator_performance = {}
            
            for disc_name, disc_results in results['detailed'].items():
                for gen_name, gen_results in disc_results.items():
                    gen_rmse_values = []
                    gen_mae_values = []
                    for iter_results in gen_results.values():
                        if 'overall_rmse' in iter_results:
                            gen_rmse_values.append(iter_results['overall_rmse'])
                            gen_mae_values.append(iter_results['overall_mae'])
                    
                    if gen_rmse_values:
                        if gen_name not in generator_performance:
                            generator_performance[gen_name] = {'rmse': [], 'mae': []}
                        generator_performance[gen_name]['rmse'].extend(gen_rmse_values)
                        generator_performance[gen_name]['mae'].extend(gen_mae_values)
            
            gen_summary = {}
            for gen_name, metrics in generator_performance.items():
                gen_summary[gen_name] = {
                    'mean_rmse': np.mean(metrics['rmse']),
                    'std_rmse': np.std(metrics['rmse']),
                    'min_rmse': np.min(metrics['rmse']),
                    'max_rmse': np.max(metrics['rmse']),
                    'mean_mae': np.mean(metrics['mae']),
                    'std_mae': np.std(metrics['mae']),
                    'min_mae': np.min(metrics['mae']),
                    'max_mae': np.max(metrics['mae'])
                }
            
            results['summary'] = {
                'overall_statistics': {
                    'mean_rmse': np.mean(all_rmse_values),
                    'std_rmse': np.std(all_rmse_values),
                    'min_rmse': np.min(all_rmse_values),
                    'max_rmse': np.max(all_rmse_values),
                    'mean_mae': np.mean(all_mae_values),
                    'std_mae': np.std(all_mae_values),
                    'min_mae': np.min(all_mae_values),
                    'max_mae': np.max(all_mae_values)
                },
                'generator_performance': gen_summary
            }
        
        print("\nCompleted {}/{} comparisons".format(successful_comparisons, total_comparisons))
        return results

    def save_results(self, results, output_path="./rmse_mae_complete_results"):
        """Save results to CSV with both RMSE and MAE"""
        os.makedirs(output_path, exist_ok=True)
        
        summary_data = []
        for disc_name, disc_results in results['detailed'].items():
            for gen_name, gen_results in disc_results.items():
                for iter_name, iter_results in gen_results.items():
                    if 'overall_rmse' in iter_results and 'overall_mae' in iter_results:
                        summary_data.append({
                            'discriminator': disc_name,
                            'generator': gen_name,
                            'iteration': iter_name,
                            'overall_rmse': iter_results['overall_rmse'],
                            'overall_mae': iter_results['overall_mae'],
                            'normalization': self.normalization
                        })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(output_path, 'rmse_mae_complete_results.csv'), index=False)
            
            # Feature-wise results
            feature_data = []
            for disc_name, disc_results in results['detailed'].items():
                for gen_name, gen_results in disc_results.items():
                    for iter_name, iter_results in gen_results.items():
                        if 'feature_rmse' in iter_results and 'feature_mae' in iter_results:
                            for feature in iter_results['feature_rmse'].keys():
                                feature_data.append({
                                    'discriminator': disc_name,
                                    'generator': gen_name,
                                    'iteration': iter_name,
                                    'feature': feature,
                                    'rmse': iter_results['feature_rmse'][feature],
                                    'mae': iter_results['feature_mae'][feature]
                                })
            
            if feature_data:
                feature_df = pd.DataFrame(feature_data)
                feature_df.to_csv(os.path.join(output_path, 'feature_wise_rmse_mae_complete.csv'), index=False)
            
            print("Complete results saved to {}".format(output_path))
            print("  - Overall metrics: rmse_mae_complete_results.csv")
            print("  - Feature-wise: feature_wise_rmse_mae_complete.csv")
            print("  - Total records: {}".format(len(summary_data)))

def main():
    # Paths
    real_data_path = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets"
    generated_data_path = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"
    
    # Create fixed evaluator with standardization
    evaluator = FixedGANEvaluator(
        real_data_path, 
        generated_data_path, 
        normalization='standard'  # This fixes the scale issue!
    )
    
    # Load ALL data - no limits for complete evaluation
    print("Starting COMPLETE evaluation - this will take longer but provide full results...")
    evaluator.load_real_data()  # Load all discriminators, no sampling
    evaluator.load_generated_data()  # Load all available iterations
    
    # Skip sample evaluation and go straight to full evaluation
    print("Skipping sample evaluation, proceeding with complete evaluation...")
    
    # Full evaluation with all data
    full_results = evaluator.evaluate_all_combinations()
    
    # Print results
    if full_results['summary']:
        print("\n" + "="*50)
        print("FINAL COMPLETE RESULTS (NORMALIZED RMSE + MAE)")
        print("="*50)
        
        overall_stats = full_results['summary']['overall_statistics']
        print("OVERALL PERFORMANCE:")
        print("   RMSE: {:.6f} +/- {:.6f}".format(
            overall_stats['mean_rmse'], overall_stats['std_rmse']))
        print("   MAE:  {:.6f} +/- {:.6f}".format(
            overall_stats['mean_mae'], overall_stats['std_mae']))
        print("   Range RMSE: [{:.6f}, {:.6f}]".format(
            overall_stats['min_rmse'], overall_stats['max_rmse']))
        print("   Range MAE:  [{:.6f}, {:.6f}]".format(
            overall_stats['min_mae'], overall_stats['max_mae']))
        
        print("\nGENERATOR RANKINGS:")
        
        # Rank generators by RMSE
        gen_rmse_ranking = sorted(
            full_results['summary']['generator_performance'].items(),
            key=lambda x: x[1]['mean_rmse']
        )
        
        # Rank generators by MAE  
        gen_mae_ranking = sorted(
            full_results['summary']['generator_performance'].items(),
            key=lambda x: x[1]['mean_mae']
        )
        
        print("   By RMSE (lower is better):")
        for i, (gen_name, stats) in enumerate(gen_rmse_ranking[:10], 1):  # Show top 10
            print("   {}. {}: {:.6f} +/- {:.6f}".format(
                i, gen_name, stats['mean_rmse'], stats['std_rmse']))
        
        print("   By MAE (lower is better):")
        for i, (gen_name, stats) in enumerate(gen_mae_ranking[:10], 1):  # Show top 10
            print("   {}. {}: {:.6f} +/- {:.6f}".format(
                i, gen_name, stats['mean_mae'], stats['std_mae']))
        
        # Performance assessment
        avg_rmse = overall_stats['mean_rmse']
        avg_mae = overall_stats['mean_mae']
        
        print("\nPERFORMANCE ASSESSMENT:")
        print("   RMSE {:.3f}: ".format(avg_rmse), end="")
        if avg_rmse < 0.5:
            print("EXCELLENT")
        elif avg_rmse < 1.0:
            print("GOOD")
        elif avg_rmse < 2.0:
            print("MODERATE")
        else:
            print("POOR")
        
        print("   MAE  {:.3f}: ".format(avg_mae), end="")
        if avg_mae < 0.3:
            print("EXCELLENT")
        elif avg_mae < 0.7:
            print("GOOD") 
        elif avg_mae < 1.5:
            print("MODERATE")
        else:
            print("POOR")
    
    # Save complete results
    evaluator.save_results(full_results)
    
    return full_results

if __name__ == "__main__":
    results = main()