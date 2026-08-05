#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe MD-GAN Metrics Runner
Runs the metrics calculation with proper error handling and encoding
"""

import os
import sys
import traceback

def run_metrics_safely():
    """Run metrics calculation with proper error handling"""
    
    print("=" * 80)
    print("SAFE MD-GAN METRICS RUNNER")
    print("=" * 80)
    
    try:
        # Set up environment
        import locale
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        print("? Environment setup complete")
        
        # Import required modules with error checking
        try:
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            print("? Required packages imported successfully")
        except ImportError as e:
            print(f"? Error importing required packages: {e}")
            print("Please install missing packages with: pip install numpy pandas matplotlib scikit-learn scipy seaborn")
            return 1
        
        # Check if results directory exists
        results_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"
        if not os.path.exists(results_dir):
            print(f"? Results directory not found: {results_dir}")
            print("Please ensure the NS-3 simulation has been run and generated results.")
            return 1
        
        print(f"? Results directory found: {results_dir}")
        
        # Import and run metrics calculator
        try:
            # Define the metrics calculator class inline to avoid import issues
            exec(open('metric_calculator_fixed.py').read())
            
            # Initialize calculator
            calculator = MDGANAccuracyMetrics(results_dir)
            print("? Metrics calculator initialized")
            
            # Run evaluation
            print("\n?? Starting metrics evaluation...")
            metrics = calculator.evaluate_all_metrics()
            
            if metrics:
                # Save results
                calculator.save_metrics_report(metrics)
                
                # Print summary
                print("\n" + "=" * 60)
                print("?? METRICS CALCULATION COMPLETED")
                print("=" * 60)
                
                if 'overall_scores' in metrics:
                    overall = metrics['overall_scores']
                    print(f"?? Composite Accuracy Score: {overall.get('composite_accuracy_score', 0):.4f}")
                    print(f"?? Statistical Similarity: {overall.get('average_statistical_similarity', 0):.4f}")
                    print(f"?? Generator Quality: {overall.get('average_generator_quality', 0):.4f}")
                    print(f"? Inception Score: {overall.get('average_inception_score', 0):.4f}")
                    print(f"?? Convergence Score: {overall.get('convergence_score', 0):.4f}")
                
                print(f"\n?? Results saved to: {results_dir}")
                return 0
            else:
                print("? No metrics were calculated. Check simulation data.")
                return 1
                
        except Exception as e:
            print(f"? Error during metrics calculation: {e}")
            traceback.print_exc()
            return 1
            
    except Exception as e:
        print(f"? Critical error: {e}")
        traceback.print_exc()
        return 1

def check_simulation_data():
    """Check if simulation has generated the required data"""
    
    print("\n?? CHECKING SIMULATION DATA")
    print("-" * 40)
    
    base_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN"
    results_dir = f"{base_dir}/results"
    datasets_dir = f"{base_dir}/datasets"
    
    # Check directories
    dirs_to_check = [
        results_dir,
        datasets_dir
    ]
    
    for directory in dirs_to_check:
        if os.path.exists(directory):
            print(f"? Directory exists: {directory}")
        else:
            print(f"? Directory missing: {directory}")
            return False
    
    # Check for generator results
    generator_count = 0
    for gen_id in range(1, 5):
        gen_dir = f"{results_dir}/generator_{gen_id}"
        if os.path.exists(gen_dir):
            generator_count += 1
            
            # Check for generated samples
            import glob
            samples = glob.glob(f"{gen_dir}/generated_samples_iter_*.csv")
            if samples:
                print(f"? Generator {gen_id}: {len(samples)} sample files")
            else:
                print(f"??  Generator {gen_id}: No sample files found")
    
    print(f"?? Found {generator_count}/4 generators")
    
    # Check for discriminator results
    disc_count = 0
    for disc_id in range(1, 21):
        disc_dir = f"{results_dir}/discriminator_{disc_id}"
        if os.path.exists(disc_dir):
            disc_count += 1
    
    print(f"?? Found {disc_count}/20 discriminators")
    
    # Check for dataset files
    dataset_count = 0
    for disc_id in range(1, 21):
        dataset_file = f"{datasets_dir}/discriminator_{disc_id}/data.csv"
        if os.path.exists(dataset_file):
            dataset_count += 1
    
    print(f"?? Found {dataset_count}/20 datasets")
    
    # Check for loss tracking
    loss_file = f"{results_dir}/loss_tracking.csv"
    if os.path.exists(loss_file):
        print(f"? Loss tracking file exists")
    else:
        print(f"??  Loss tracking file missing")
    
    return True

def create_sample_data():
    """Create sample data for testing if real data is not available"""
    
    print("\n?? CREATING SAMPLE DATA FOR TESTING")
    print("-" * 40)
    
    import numpy as np
    import pandas as pd
    import os
    
    base_dir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN"
    results_dir = f"{base_dir}/results"
    datasets_dir = f"{base_dir}/datasets"
    
    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Create sample datasets for discriminators
    for disc_id in range(1, 21):
        disc_dataset_dir = f"{datasets_dir}/discriminator_{disc_id}"
        os.makedirs(disc_dataset_dir, exist_ok=True)
        
        # Generate synthetic tabular data (e.g., vehicle sensor data)
        n_samples = 1000
        n_features = 10
        
        # Create realistic vehicle data
        np.random.seed(disc_id * 42)  # Different seed for each discriminator
        
        data = {
            'speed': np.random.normal(50, 15, n_samples),  # Speed in km/h
            'acceleration': np.random.normal(0, 2, n_samples),  # m/s^2
            'fuel_consumption': np.random.exponential(8, n_samples),  # L/100km
            'engine_temp': np.random.normal(90, 10, n_samples),  # Celsius
            'rpm': np.random.normal(2000, 500, n_samples),  # RPM
            'brake_pressure': np.random.exponential(0.5, n_samples),  # Bar
            'tire_pressure': np.random.normal(2.2, 0.2, n_samples),  # Bar
            'battery_voltage': np.random.normal(12.6, 0.5, n_samples),  # Volts
            'gps_accuracy': np.random.exponential(2, n_samples),  # Meters
            'signal_strength': np.random.normal(-70, 10, n_samples)  # dBm
        }
        
        df = pd.DataFrame(data)
        df.to_csv(f"{disc_dataset_dir}/data.csv", index=False)
    
    # Create sample generator results
    for gen_id in range(1, 5):
        gen_dir = f"{results_dir}/generator_{gen_id}"
        os.makedirs(gen_dir, exist_ok=True)
        
        # Generate synthetic generated samples
        np.random.seed(gen_id * 123)
        
        # Make generated data slightly different from real data
        noise_factor = 0.1 + gen_id * 0.05
        
        data = {
            'speed': np.random.normal(48 + gen_id, 15 * (1 + noise_factor), 500),
            'acceleration': np.random.normal(0.2, 2 * (1 + noise_factor), 500),
            'fuel_consumption': np.random.exponential(8 * (1 + noise_factor), 500),
            'engine_temp': np.random.normal(88 + gen_id, 10 * (1 + noise_factor), 500),
            'rpm': np.random.normal(1950 + gen_id * 50, 500 * (1 + noise_factor), 500),
            'brake_pressure': np.random.exponential(0.5 * (1 + noise_factor), 500),
            'tire_pressure': np.random.normal(2.15 + gen_id * 0.02, 0.2 * (1 + noise_factor), 500),
            'battery_voltage': np.random.normal(12.5 + gen_id * 0.05, 0.5 * (1 + noise_factor), 500),
            'gps_accuracy': np.random.exponential(2 * (1 + noise_factor), 500),
            'signal_strength': np.random.normal(-72 + gen_id, 10 * (1 + noise_factor), 500)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(f"{gen_dir}/generated_samples_iter_{gen_id}.csv", index=False)
        
        # Create sample metrics
        metrics = {
            "node_id": gen_id,
            "iteration": gen_id,
            "latency": 5.0 + gen_id * 0.5 + np.random.normal(0, 0.5),
            "cpu_usage": 2.0 + gen_id * 0.3 + np.random.normal(0, 0.2),
            "memory_usage": 150 + gen_id * 20 + np.random.normal(0, 10),
            "network_traffic": 1024 * (50 + gen_id * 10) + np.random.normal(0, 1000),
            "avg_disc_score": 0.5 + np.random.normal(0, 0.1),
            "proxy_loss": 2.0 - gen_id * 0.2 + np.random.normal(0, 0.1)
        }
        
        with open(f"{gen_dir}/metrics.json", 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
    
    # Create sample discriminator results
    for disc_id in range(1, 21):
        disc_dir = f"{results_dir}/discriminator_{disc_id}"
        os.makedirs(disc_dir, exist_ok=True)
        
        # Create sample discriminator metrics
        metrics = {
            "node_id": disc_id,
            "latency": 3.0 + (disc_id % 5) * 0.3 + np.random.normal(0, 0.2),
            "cpu_usage": 1.5 + (disc_id % 5) * 0.2 + np.random.normal(0, 0.1),
            "memory_usage": 100 + (disc_id % 5) * 15 + np.random.normal(0, 5),
            "network_traffic": 1024 * (30 + (disc_id % 5) * 5) + np.random.normal(0, 500),
            "mean_score": 0.6 + np.random.normal(0, 0.1),
            "loss": 1.5 - (disc_id % 5) * 0.1 + np.random.normal(0, 0.05)
        }
        
        with open(f"{disc_dir}/metrics.json", 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
    
    # Create sample loss tracking file
    loss_data = []
    import time
    base_time = int(time.time()) - 3600  # 1 hour ago
    
    for i in range(50):  # 50 time steps
        # Generator losses
        for gen_id in range(1, 5):
            loss_value = 3.0 - i * 0.05 + np.random.normal(0, 0.1)
            improved = "Yes" if i > 0 and np.random.random() > 0.3 else "No"
            
            loss_data.append({
                "Timestamp": base_time + i * 60,
                "NodeType": "Generator",
                "NodeID": gen_id,
                "LossValue": max(0.1, loss_value),
                "Improved": improved
            })
        
        # Discriminator losses
        for disc_id in range(1, 21):
            loss_value = 2.5 - i * 0.03 + np.random.normal(0, 0.08)
            improved = "Yes" if i > 0 and np.random.random() > 0.4 else "No"
            
            loss_data.append({
                "Timestamp": base_time + i * 60,
                "NodeType": "Discriminator", 
                "NodeID": disc_id,
                "LossValue": max(0.1, loss_value),
                "Improved": improved
            })
    
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(f"{results_dir}/loss_tracking.csv", index=False)
    
    print("? Sample data created successfully")
    return True

def main():
    """Main function"""
    
    print("?? Starting MD-GAN Metrics Analysis")
    
    # Check if simulation data exists
    if not check_simulation_data():
        print("\n??  Simulation data not found. Creating sample data for testing...")
        if not create_sample_data():
            print("? Failed to create sample data")
            return 1
        print("? Sample data created. You can now run the metrics calculation.")
    
    # Run metrics calculation
    return run_metrics_safely()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)