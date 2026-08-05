#!/usr/bin/env python3
# gan_evaluation_metrics.py - Comprehensive evaluation metrics for tabular GANs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, cdist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class TabularGANEvaluator:
    """Comprehensive evaluation metrics for tabular GAN performance"""
    
    def __init__(self, real_data, synthetic_data, categorical_columns=None, target_column=None):
        """
        Initialize evaluator with real and synthetic data
        
        Args:
            real_data: pandas DataFrame of real data
            synthetic_data: pandas DataFrame of synthetic data
            categorical_columns: list of categorical column names
            target_column: target column name (if supervised evaluation needed)
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.categorical_columns = categorical_columns or []
        self.target_column = target_column
        
        # Ensure same columns in both datasets
        common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
        self.real_data = self.real_data[common_columns]
        self.synthetic_data = self.synthetic_data[common_columns]
        
        self.numerical_columns = [col for col in common_columns 
                                if col not in self.categorical_columns]
        
        self.results = {}
        
    def _get_default_metrics(self, metric_type):
        """Return default metrics when evaluation fails"""
        defaults = {
            'statistical_similarity': {
                'wasserstein_distance': {'mean': np.inf, 'std': 0, 'per_column': {}},
                'kolmogorov_smirnov': {'mean_statistic': 1.0, 'mean_pvalue': 0.0, 'per_column_stats': {}, 'per_column_pvalues': {}},
                'jensen_shannon_divergence': {'mean': np.inf, 'std': 0, 'per_column': {}}
            },
            'distribution_comparison': {
                'mean_std_comparison': {'mean_absolute_error': np.inf, 'std_absolute_error': np.inf},
                'correlation_comparison': {'frobenius_norm': np.inf, 'mean_absolute_error': np.inf, 'max_absolute_error': np.inf},
                'marginal_distribution': {'mean_histogram_difference': np.inf}
            },
            'privacy_diversity': {
                'privacy': {'nearest_neighbor_ratio': 0, 'mean_distance_to_real': np.inf, 'mean_distance_within_real': np.inf},
                'coverage': {'coverage_ratio': 0, 'threshold_distance': np.inf},
                'diversity': {'diversity_ratio': 0, 'mean_pairwise_distance_synth': 0, 'mean_pairwise_distance_real': 0}
            }
        }
        return defaults.get(metric_type, {})
        
    def statistical_similarity_metrics(self):
        """Calculate statistical similarity metrics"""
        print("Computing statistical similarity metrics...")
        
        metrics = {}
        
        # Validate data first
        if self.real_data.empty or self.synthetic_data.empty:
            print("Warning: Empty datasets detected")
            return self._get_default_metrics('statistical_similarity')
        
        # 1. Wasserstein Distance (Earth Mover's Distance)
        wasserstein_distances = []
        valid_columns = []
        
        for col in self.numerical_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                real_col = self.real_data[col].dropna()
                synth_col = self.synthetic_data[col].dropna()
                
                # Check for valid data
                if (len(real_col) > 0 and len(synth_col) > 0 and 
                    np.isfinite(real_col).any() and np.isfinite(synth_col).any()):
                    try:
                        wd = stats.wasserstein_distance(real_col, synth_col)
                        if np.isfinite(wd):
                            wasserstein_distances.append(wd)
                            valid_columns.append(col)
                    except Exception as e:
                        print(f"Warning: Could not compute Wasserstein distance for {col}: {e}")
                        continue
        
        metrics['wasserstein_distance'] = {
            'mean': np.mean(wasserstein_distances) if wasserstein_distances else np.inf,
            'std': np.std(wasserstein_distances) if len(wasserstein_distances) > 1 else 0,
            'per_column': dict(zip(valid_columns, wasserstein_distances)) if wasserstein_distances else {}
        }
        
        # 2. Kolmogorov-Smirnov Test
        ks_statistics = []
        ks_pvalues = []
        for col in valid_columns:  # Use only valid columns
            real_col = self.real_data[col].dropna()
            synth_col = self.synthetic_data[col].dropna()
            if len(real_col) > 1 and len(synth_col) > 1:  # Need at least 2 points
                try:
                    ks_stat, p_value = stats.ks_2samp(real_col, synth_col)
                    if np.isfinite(ks_stat) and np.isfinite(p_value):
                        ks_statistics.append(ks_stat)
                        ks_pvalues.append(p_value)
                except Exception as e:
                    print(f"Warning: Could not compute KS test for {col}: {e}")
                    continue
        
        metrics['kolmogorov_smirnov'] = {
            'mean_statistic': np.mean(ks_statistics) if ks_statistics else 1.0,
            'mean_pvalue': np.mean(ks_pvalues) if ks_pvalues else 0.0,
            'per_column_stats': dict(zip(valid_columns[:len(ks_statistics)], ks_statistics)) if ks_statistics else {},
            'per_column_pvalues': dict(zip(valid_columns[:len(ks_pvalues)], ks_pvalues)) if ks_pvalues else {}
        }
        
        # 3. Jensen-Shannon Divergence
        def jensen_shannon_divergence(p, q, bins=50):
            """Calculate Jensen-Shannon divergence between two distributions"""
            try:
                # Ensure we have valid data
                if len(p) == 0 or len(q) == 0:
                    return np.inf
                
                # Create histograms
                min_val, max_val = min(p.min(), q.min()), max(p.max(), q.max())
                if min_val == max_val:
                    return 0.0  # Identical single values
                
                bin_edges = np.linspace(min_val, max_val, bins + 1)
                
                hist_p, _ = np.histogram(p, bins=bin_edges, density=True)
                hist_q, _ = np.histogram(q, bins=bin_edges, density=True)
                
                # Normalize to probability distributions
                hist_p = hist_p / (np.sum(hist_p) + 1e-10) + 1e-10
                hist_q = hist_q / (np.sum(hist_q) + 1e-10) + 1e-10
                
                # Calculate JS divergence
                m = 0.5 * (hist_p + hist_q)
                js_div = 0.5 * stats.entropy(hist_p, m) + 0.5 * stats.entropy(hist_q, m)
                return js_div if np.isfinite(js_div) else np.inf
            except Exception:
                return np.inf
        
        js_divergences = []
        for col in valid_columns:
            real_col = self.real_data[col].dropna()
            synth_col = self.synthetic_data[col].dropna()
            if len(real_col) > 1 and len(synth_col) > 1:
                js_div = jensen_shannon_divergence(real_col, synth_col)
                if np.isfinite(js_div):
                    js_divergences.append(js_div)
        
        metrics['jensen_shannon_divergence'] = {
            'mean': np.mean(js_divergences) if js_divergences else np.inf,
            'std': np.std(js_divergences) if len(js_divergences) > 1 else 0,
            'per_column': dict(zip(valid_columns[:len(js_divergences)], js_divergences)) if js_divergences else {}
        }
        
        self.results['statistical_similarity'] = metrics
        return metrics
    
    def distribution_comparison_metrics(self):
        """Compare statistical distributions"""
        print("Computing distribution comparison metrics...")
        
        metrics = {}
        
        # 1. Mean and Standard Deviation Comparison
        mean_errors = []
        std_errors = []
        
        for col in self.numerical_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                real_mean = self.real_data[col].mean()
                synth_mean = self.synthetic_data[col].mean()
                real_std = self.real_data[col].std()
                synth_std = self.synthetic_data[col].std()
                
                mean_error = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10)
                std_error = abs(real_std - synth_std) / (abs(real_std) + 1e-10)
                
                mean_errors.append(mean_error)
                std_errors.append(std_error)
        
        metrics['mean_std_comparison'] = {
            'mean_absolute_error': np.mean(mean_errors) if mean_errors else np.inf,
            'std_absolute_error': np.mean(std_errors) if std_errors else np.inf
        }
        
        # 2. Correlation Matrix Comparison
        if len(self.numerical_columns) > 1:
            real_corr = self.real_data[self.numerical_columns].corr()
            synth_corr = self.synthetic_data[self.numerical_columns].corr()
            
            # Handle NaN values
            real_corr = real_corr.fillna(0)
            synth_corr = synth_corr.fillna(0)
            
            corr_diff = np.abs(real_corr.values - synth_corr.values)
            
            metrics['correlation_comparison'] = {
                'frobenius_norm': np.linalg.norm(corr_diff, 'fro'),
                'mean_absolute_error': np.mean(corr_diff),
                'max_absolute_error': np.max(corr_diff)
            }
        
        # 3. Marginal Distribution Comparison (using histograms)
        marginal_differences = []
        for col in self.numerical_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                real_col = self.real_data[col].dropna()
                synth_col = self.synthetic_data[col].dropna()
                
                if len(real_col) > 0 and len(synth_col) > 0:
                    # Create common bins
                    min_val = min(real_col.min(), synth_col.min())
                    max_val = max(real_col.max(), synth_col.max())
                    bins = np.linspace(min_val, max_val, 50)
                    
                    real_hist, _ = np.histogram(real_col, bins=bins, density=True)
                    synth_hist, _ = np.histogram(synth_col, bins=bins, density=True)
                    
                    # Calculate difference
                    hist_diff = np.sum(np.abs(real_hist - synth_hist))
                    marginal_differences.append(hist_diff)
        
        metrics['marginal_distribution'] = {
            'mean_histogram_difference': np.mean(marginal_differences) if marginal_differences else np.inf
        }
        
        self.results['distribution_comparison'] = metrics
        return metrics
    
    def machine_learning_utility_metrics(self):
        """Evaluate ML utility using TSTR and TRTS"""
        print("Computing machine learning utility metrics...")
        
        if self.target_column is None or self.target_column not in self.real_data.columns:
            print("Warning: No target column specified for ML utility metrics")
            return {}
        
        metrics = {}
        
        # Prepare data
        real_X = self.real_data.drop(columns=[self.target_column])
        real_y = self.real_data[self.target_column]
        synth_X = self.synthetic_data.drop(columns=[self.target_column]) if self.target_column in self.synthetic_data.columns else self.synthetic_data
        synth_y = self.synthetic_data[self.target_column] if self.target_column in self.synthetic_data.columns else None
        
        # Handle categorical columns
        if self.categorical_columns:
            le = LabelEncoder()
            for col in self.categorical_columns:
                if col in real_X.columns:
                    real_X[col] = le.fit_transform(real_X[col].astype(str))
                if col in synth_X.columns:
                    synth_X[col] = le.transform(synth_X[col].astype(str))
        
        # Split real data
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            real_X, real_y, test_size=0.3, random_state=42
        )
        
        # Determine if classification or regression
        is_classification = len(np.unique(real_y)) < 20  # Heuristic
        
        if is_classification:
            model_real = RandomForestClassifier(n_estimators=100, random_state=42)
            model_synth = RandomForestClassifier(n_estimators=100, random_state=42)
            score_func = accuracy_score
        else:
            model_real = RandomForestRegressor(n_estimators=100, random_state=42)
            model_synth = RandomForestRegressor(n_estimators=100, random_state=42)
            score_func = r2_score
        
        # 1. Train on Real, Test on Real (baseline)
        model_real.fit(X_train_real, y_train_real)
        y_pred_real = model_real.predict(X_test_real)
        score_real_real = score_func(y_test_real, y_pred_real)
        
        # 2. Train on Synthetic, Test on Real (TSTR)
        if synth_y is not None and len(synth_X) > 0:
            # Ensure synthetic data has same features as real data
            common_features = list(set(X_train_real.columns) & set(synth_X.columns))
            synth_X_aligned = synth_X[common_features]
            X_train_real_aligned = X_train_real[common_features]
            X_test_real_aligned = X_test_real[common_features]
            
            model_synth.fit(synth_X_aligned, synth_y)
            y_pred_synth = model_synth.predict(X_test_real_aligned)
            score_synth_real = score_func(y_test_real, y_pred_synth)
            
            # 3. Train on Real, Test on Synthetic (TRTS) - if we have synthetic targets
            if len(synth_y) > 0:
                y_pred_real_synth = model_real.predict(synth_X_aligned)
                score_real_synth = score_func(synth_y, y_pred_real_synth)
            else:
                score_real_synth = None
        else:
            score_synth_real = None
            score_real_synth = None
        
        metrics['ml_utility'] = {
            'baseline_score': score_real_real,
            'tstr_score': score_synth_real,
            'trts_score': score_real_synth,
            'tstr_ratio': score_synth_real / score_real_real if score_synth_real is not None else None,
            'task_type': 'classification' if is_classification else 'regression'
        }
        
        self.results['ml_utility'] = metrics
        return metrics
    
    def privacy_and_diversity_metrics(self):
        """Calculate privacy and diversity metrics"""
        print("Computing privacy and diversity metrics...")
        
        metrics = {}
        
        # Validate data
        if (self.real_data.empty or self.synthetic_data.empty or 
            len(self.numerical_columns) == 0):
            print("Warning: Insufficient data for privacy/diversity metrics")
            return self._get_default_metrics('privacy_diversity')
        
        # Prepare numerical data only for distance calculations
        try:
            real_num = self.real_data[self.numerical_columns].dropna()
            synth_num = self.synthetic_data[self.numerical_columns].dropna()
            
            # Additional validation
            if len(real_num) < 2 or len(synth_num) < 2:
                print("Warning: Insufficient samples for privacy/diversity metrics")
                return self._get_default_metrics('privacy_diversity')
            
            # Check for valid numerical data
            real_num = real_num.select_dtypes(include=[np.number])
            synth_num = synth_num.select_dtypes(include=[np.number])
            
            if real_num.empty or synth_num.empty:
                print("Warning: No valid numerical data for distance calculations")
                return self._get_default_metrics('privacy_diversity')
            
            # Replace infinite values
            real_num = real_num.replace([np.inf, -np.inf], np.nan).dropna()
            synth_num = synth_num.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(real_num) < 2 or len(synth_num) < 2:
                print("Warning: Insufficient valid samples after cleaning")
                return self._get_default_metrics('privacy_diversity')
            
            # Standardize data for distance calculations
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_num)
            
            # Check if scaler fitted properly
            if np.any(np.isnan(real_scaled)) or np.any(np.isinf(real_scaled)):
                print("Warning: Scaling produced invalid values")
                return self._get_default_metrics('privacy_diversity')
            
            synth_scaled = scaler.transform(synth_num)
            
            # Validate scaled data
            if (real_scaled.size == 0 or synth_scaled.size == 0 or
                np.any(np.isnan(synth_scaled)) or np.any(np.isinf(synth_scaled))):
                print("Warning: Invalid scaled synthetic data")
                return self._get_default_metrics('privacy_diversity')
            
        except Exception as e:
            print(f"Warning: Error preparing data for privacy metrics: {e}")
            return self._get_default_metrics('privacy_diversity')
        
        # 1. Nearest Neighbor Distance Ratio (Privacy)
        try:
            if len(real_scaled) > 2 and len(synth_scaled) > 1:
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(2, len(real_scaled))).fit(real_scaled)
                
                # Distances from synthetic to real data
                distances_synth_real, _ = nbrs.kneighbors(synth_scaled, n_neighbors=1)
                
                # Distances within real data (for comparison)
                k_real = min(2, len(real_scaled))
                distances_real_real, _ = nbrs.kneighbors(real_scaled, n_neighbors=k_real)
                if k_real == 2:
                    distances_real_real = distances_real_real[:, 1]  # Exclude self
                else:
                    distances_real_real = distances_real_real[:, 0]
                
                # Validate distances
                if (len(distances_synth_real) > 0 and len(distances_real_real) > 0 and
                    np.mean(distances_real_real) > 0):
                    privacy_score = np.mean(distances_synth_real) / np.mean(distances_real_real)
                    
                    metrics['privacy'] = {
                        'nearest_neighbor_ratio': float(privacy_score),
                        'mean_distance_to_real': float(np.mean(distances_synth_real)),
                        'mean_distance_within_real': float(np.mean(distances_real_real))
                    }
        except Exception as e:
            print(f"Warning: Could not compute privacy metrics: {e}")
        
        # 2. Coverage (what fraction of real data space is covered by synthetic)
        try:
            if len(real_scaled) > 5 and len(synth_scaled) > 5:
                # For each real point, check if there's a synthetic point nearby
                nbrs_synth = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
                distances_real_synth, _ = nbrs_synth.kneighbors(real_scaled)
                
                # Define "covered" as within median distance of real data
                if 'privacy' in metrics and len(distances_real_real) > 0:
                    threshold = np.median(distances_real_real)
                    coverage = np.mean(distances_real_synth.flatten() <= threshold)
                    
                    metrics['coverage'] = {
                        'coverage_ratio': float(coverage),
                        'threshold_distance': float(threshold)
                    }
        except Exception as e:
            print(f"Warning: Could not compute coverage metrics: {e}")
        
        # 3. Diversity within synthetic data
        try:
            if len(synth_scaled) > 2 and len(real_scaled) > 2:
                from scipy.spatial.distance import pdist
                
                synth_distances = pdist(synth_scaled)
                real_distances = pdist(real_scaled)
                
                if (len(synth_distances) > 0 and len(real_distances) > 0 and 
                    np.mean(real_distances) > 0):
                    diversity_ratio = np.mean(synth_distances) / np.mean(real_distances)
                    
                    metrics['diversity'] = {
                        'diversity_ratio': float(diversity_ratio),
                        'mean_pairwise_distance_synth': float(np.mean(synth_distances)),
                        'mean_pairwise_distance_real': float(np.mean(real_distances))
                    }
        except Exception as e:
            print(f"Warning: Could not compute diversity metrics: {e}")
        
        # If no metrics computed, return defaults
        if not metrics:
            metrics = self._get_default_metrics('privacy_diversity')
        
        self.results['privacy_diversity'] = metrics
        return metrics
    
    def domain_specific_metrics(self):
        """Calculate domain-specific metrics (vehicular data)"""
        print("Computing domain-specific metrics...")
        
        metrics = {}
        
        # Example for vehicular data - adapt based on your specific domain
        vehicular_features = {
            'speed': ['speed', 'velocity', 'vehicle_speed'],
            'location': ['x_coord', 'y_coord', 'latitude', 'longitude'],
            'network': ['signal_strength', 'packet_loss', 'latency'],
            'traffic': ['traffic_density', 'message_frequency']
        }
        
        for category, possible_columns in vehicular_features.items():
            category_columns = [col for col in possible_columns 
                              if col in self.real_data.columns and col in self.synthetic_data.columns]
            
            if category_columns:
                category_metrics = {}
                
                for col in category_columns:
                    real_col = self.real_data[col].dropna()
                    synth_col = self.synthetic_data[col].dropna()
                    
                    if len(real_col) > 0 and len(synth_col) > 0:
                        # Range comparison
                        real_range = real_col.max() - real_col.min()
                        synth_range = synth_col.max() - synth_col.min()
                        range_ratio = synth_range / real_range if real_range > 0 else 1
                        
                        # Outlier comparison
                        real_q25, real_q75 = np.percentile(real_col, [25, 75])
                        synth_q25, synth_q75 = np.percentile(synth_col, [25, 75])
                        real_iqr = real_q75 - real_q25
                        synth_iqr = synth_q75 - synth_q25
                        
                        real_outliers = np.sum((real_col < real_q25 - 1.5 * real_iqr) | 
                                             (real_col > real_q75 + 1.5 * real_iqr)) / len(real_col)
                        synth_outliers = np.sum((synth_col < synth_q25 - 1.5 * synth_iqr) | 
                                              (synth_col > synth_q75 + 1.5 * synth_iqr)) / len(synth_col)
                        
                        category_metrics[col] = {
                            'range_ratio': range_ratio,
                            'real_outlier_rate': real_outliers,
                            'synth_outlier_rate': synth_outliers,
                            'outlier_rate_diff': abs(real_outliers - synth_outliers)
                        }
                
                metrics[category] = category_metrics
        
        self.results['domain_specific'] = metrics
        return metrics
    
    def compute_all_metrics(self):
        """Compute all evaluation metrics"""
        print("Computing comprehensive GAN evaluation metrics...")
        print("=" * 60)
        
        # Run all metric calculations
        self.statistical_similarity_metrics()
        self.distribution_comparison_metrics()
        self.machine_learning_utility_metrics()
        self.privacy_and_diversity_metrics()
        self.domain_specific_metrics()
        
        # Calculate overall scores
        self.calculate_overall_scores()
        
        return self.results
    
    def calculate_overall_scores(self):
        """Calculate overall quality scores"""
        scores = {}
        
        # Statistical Quality Score (0-1, higher is better)
        stat_metrics = self.results.get('statistical_similarity', {})
        wd_score = 1 / (1 + stat_metrics.get('wasserstein_distance', {}).get('mean', np.inf))
        ks_score = stat_metrics.get('kolmogorov_smirnov', {}).get('mean_pvalue', 0)
        js_score = 1 / (1 + stat_metrics.get('jensen_shannon_divergence', {}).get('mean', np.inf))
        
        scores['statistical_quality'] = np.mean([wd_score, ks_score, js_score])
        
        # Distribution Quality Score
        dist_metrics = self.results.get('distribution_comparison', {})
        mean_std_score = 1 / (1 + dist_metrics.get('mean_std_comparison', {}).get('mean_absolute_error', np.inf))
        corr_score = 1 / (1 + dist_metrics.get('correlation_comparison', {}).get('mean_absolute_error', np.inf))
        
        scores['distribution_quality'] = np.mean([mean_std_score, corr_score])
        
        # ML Utility Score
        ml_metrics = self.results.get('ml_utility', {})
        tstr_ratio = ml_metrics.get('ml_utility', {}).get('tstr_ratio', 0)
        if tstr_ratio is not None:
            scores['ml_utility'] = min(tstr_ratio, 1.0)  # Cap at 1.0
        else:
            scores['ml_utility'] = None
        
        # Privacy Score
        privacy_metrics = self.results.get('privacy_diversity', {})
        privacy_ratio = privacy_metrics.get('privacy', {}).get('nearest_neighbor_ratio', 1)
        scores['privacy'] = min(privacy_ratio / 2, 1.0)  # Normalize and cap
        
        # Overall Score (weighted average)
        valid_scores = [s for s in [scores['statistical_quality'], 
                                   scores['distribution_quality'], 
                                   scores.get('ml_utility'),
                                   scores['privacy']] if s is not None]
        
        if valid_scores:
            scores['overall'] = np.mean(valid_scores)
        else:
            scores['overall'] = 0.0
        
        self.results['overall_scores'] = scores
        
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("TABULAR GAN EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Real data shape: {self.real_data.shape}")
        report.append(f"Synthetic data shape: {self.synthetic_data.shape}")
        report.append(f"Numerical columns: {len(self.numerical_columns)}")
        report.append(f"Categorical columns: {len(self.categorical_columns)}")
        report.append("")
        
        # Overall Scores
        if 'overall_scores' in self.results:
            report.append("OVERALL QUALITY SCORES (0-1, higher is better)")
            report.append("-" * 50)
            scores = self.results['overall_scores']
            for metric, score in scores.items():
                if score is not None:
                    report.append(f"{metric.replace('_', ' ').title()}: {score:.4f}")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: N/A")
            report.append("")
        
        # Statistical Similarity
        if 'statistical_similarity' in self.results:
            report.append("STATISTICAL SIMILARITY METRICS")
            report.append("-" * 40)
            stat = self.results['statistical_similarity']
            
            wd = stat.get('wasserstein_distance', {})
            report.append(f"Wasserstein Distance (mean): {wd.get('mean', 'N/A'):.6f}")
            
            ks = stat.get('kolmogorov_smirnov', {})
            report.append(f"KS Test (mean p-value): {ks.get('mean_pvalue', 'N/A'):.6f}")
            
            js = stat.get('jensen_shannon_divergence', {})
            report.append(f"Jensen-Shannon Divergence: {js.get('mean', 'N/A'):.6f}")
            report.append("")
        
        # Distribution Comparison
        if 'distribution_comparison' in self.results:
            report.append("DISTRIBUTION COMPARISON")
            report.append("-" * 30)
            dist = self.results['distribution_comparison']
            
            mean_std = dist.get('mean_std_comparison', {})
            report.append(f"Mean Absolute Error: {mean_std.get('mean_absolute_error', 'N/A'):.6f}")
            report.append(f"Std Absolute Error: {mean_std.get('std_absolute_error', 'N/A'):.6f}")
            
            corr = dist.get('correlation_comparison', {})
            if corr:
                report.append(f"Correlation MAE: {corr.get('mean_absolute_error', 'N/A'):.6f}")
            report.append("")
        
        # ML Utility
        if 'ml_utility' in self.results:
            report.append("MACHINE LEARNING UTILITY")
            report.append("-" * 30)
            ml = self.results['ml_utility']['ml_utility']
            report.append(f"Task Type: {ml.get('task_type', 'N/A')}")
            report.append(f"Baseline Score: {ml.get('baseline_score', 'N/A'):.4f}")
            report.append(f"TSTR Score: {ml.get('tstr_score', 'N/A'):.4f}")
            report.append(f"TSTR Ratio: {ml.get('tstr_ratio', 'N/A'):.4f}")
            report.append("")
        
        # Privacy and Diversity
        if 'privacy_diversity' in self.results:
            report.append("PRIVACY AND DIVERSITY")
            report.append("-" * 25)
            pd_metrics = self.results['privacy_diversity']
            
            privacy = pd_metrics.get('privacy', {})
            if privacy:
                report.append(f"NN Distance Ratio: {privacy.get('nearest_neighbor_ratio', 'N/A'):.4f}")
            
            coverage = pd_metrics.get('coverage', {})
            if coverage:
                report.append(f"Coverage Ratio: {coverage.get('coverage_ratio', 'N/A'):.4f}")
            
            diversity = pd_metrics.get('diversity', {})
            if diversity:
                report.append(f"Diversity Ratio: {diversity.get('diversity_ratio', 'N/A'):.4f}")
            report.append("")
        
        # Interpretation Guide
        report.append("INTERPRETATION GUIDE")
        report.append("-" * 20)
        report.append("* Wasserstein Distance: Lower is better (0 = identical)")
        report.append("* KS Test p-value: Higher is better (>0.05 = similar)")
        report.append("* TSTR Ratio: Closer to 1.0 is better")
        report.append("* NN Distance Ratio: >1.0 indicates good privacy")
        report.append("* Coverage Ratio: Higher is better (how much real space is covered)")
        report.append("* Diversity Ratio: Close to 1.0 is optimal")
        
        return "\n".join(report)
    
    def plot_evaluation_results(self, save_path=None):
        """Create comprehensive evaluation plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GAN Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Distribution Comparison for first few columns
        ax = axes[0, 0]
        n_cols_to_plot = min(3, len(self.numerical_columns))
        for i, col in enumerate(self.numerical_columns[:n_cols_to_plot]):
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                real_data = self.real_data[col].dropna()
                synth_data = self.synthetic_data[col].dropna()
                
                ax.hist(real_data, bins=30, alpha=0.5, label=f'Real {col}', density=True)
                ax.hist(synth_data, bins=30, alpha=0.5, label=f'Synth {col}', density=True)
        
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Correlation Matrix Comparison
        if len(self.numerical_columns) > 1:
            ax = axes[0, 1]
            real_corr = self.real_data[self.numerical_columns].corr()
            synth_corr = self.synthetic_data[self.numerical_columns].corr()
            
            # Plot difference
            corr_diff = np.abs(real_corr.fillna(0) - synth_corr.fillna(0))
            im = ax.imshow(corr_diff, cmap='Reds', vmin=0, vmax=1)
            ax.set_title('Correlation Difference\n(Real vs Synthetic)')
            plt.colorbar(im, ax=ax)
        
        # 3. Overall Scores
        ax = axes[0, 2]
        if 'overall_scores' in self.results:
            scores = self.results['overall_scores']
            score_names = [k.replace('_', '\n').title() for k in scores.keys() if scores[k] is not None]
            score_values = [v for v in scores.values() if v is not None]
            
            bars = ax.bar(score_names, score_values, color=['red', 'blue', 'green', 'orange', 'purple'][:len(score_values)])
            ax.set_title('Overall Quality Scores')
            ax.set_ylabel('Score (0-1)')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, score_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Wasserstein Distances per column
        ax = axes[1, 0]
        if 'statistical_similarity' in self.results:
            wd_data = self.results['statistical_similarity']['wasserstein_distance']['per_column']
            if wd_data:
                columns = list(wd_data.keys())[:10]  # Show first 10
                distances = [wd_data[col] for col in columns]
                
                ax.bar(range(len(columns)), distances)
                ax.set_title('Wasserstein Distance per Column')
                ax.set_ylabel('Distance')
                ax.set_xticks(range(len(columns)))
                ax.set_xticklabels(columns, rotation=45)
        
        # 5. ML Utility Comparison
        ax = axes[1, 1]
        if 'ml_utility' in self.results and self.results['ml_utility']:
            ml_data = self.results['ml_utility']['ml_utility']
            if ml_data.get('baseline_score') is not None and ml_data.get('tstr_score') is not None:
                scores = [ml_data['baseline_score'], ml_data['tstr_score']]
                labels = ['Baseline\n(Real?Real)', 'TSTR\n(Synth?Real)']
                
                bars = ax.bar(labels, scores, color=['blue', 'red'])
                ax.set_title('ML Utility Comparison')
                ax.set_ylabel('Score')
                
                # Add value labels
                for bar, value in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Privacy vs Utility Trade-off
        ax = axes[1, 2]
        if 'privacy_diversity' in self.results and 'overall_scores' in self.results:
            privacy_score = self.results['overall_scores'].get('privacy')
            utility_score = self.results['overall_scores'].get('statistical_quality')
            
            if privacy_score is not None and utility_score is not None:
                ax.scatter([privacy_score], [utility_score], s=100, c='red')
                ax.set_xlabel('Privacy Score')
                ax.set_ylabel('Utility Score')
                ax.set_title('Privacy vs Utility Trade-off')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                # Add quadrant labels
                ax.text(0.8, 0.8, 'High Privacy\nHigh Utility', ha='center', va='center', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
                ax.text(0.2, 0.2, 'Low Privacy\nLow Utility', ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to {save_path}")
        
        plt.show()
        
        return fig

# Example usage function
def evaluate_gan_performance(real_data_path, synthetic_data_path, 
                           categorical_columns=None, target_column=None, 
                           output_dir=None):
    """
    Convenience function to evaluate GAN performance
    
    Args:
        real_data_path: path to real data CSV
        synthetic_data_path: path to synthetic data CSV  
        categorical_columns: list of categorical column names
        target_column: target column for supervised evaluation
        output_dir: directory to save results
    """
    
    # Load data
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    
    # Initialize evaluator
    evaluator = TabularGANEvaluator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        categorical_columns=categorical_columns,
        target_column=target_column
    )
    
    # Compute all metrics
    results = evaluator.compute_all_metrics()
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Create plots
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save report
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report)
        
        # Save plots
        fig = evaluator.plot_evaluation_results(
            save_path=os.path.join(output_dir, 'evaluation_plots.png')
        )
        
        # Save detailed results
        import json
        with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
    
    return results, evaluator

if __name__ == "__main__":
    # Example usage
    print("Tabular GAN Evaluation Metrics")
    print("This module provides comprehensive evaluation metrics for tabular GANs")
    print("Import and use evaluate_gan_performance() function or TabularGANEvaluator class")