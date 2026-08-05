#!/usr/bin/env python3
# diagnose_data_issue.py - Find out why data has 0 columns
import os
import pandas as pd
import numpy as np
import argparse

def diagnose_node_data_issue(base_dir, node_id):
    """Diagnose data loading issues for a specific node"""
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSING DATA ISSUE FOR NODE {node_id}")
    print(f"{'='*60}")
    
    results_dir = os.path.join(base_dir, "results")
    datasets_dir = os.path.join(base_dir, "datasets")
    
    # Step 1: Check file paths
    print("1. CHECKING FILE PATHS")
    print("-" * 30)
    
    node_dir = os.path.join(results_dir, f"hybrid_node_{node_id}")
    print(f"Node directory: {node_dir}")
    print(f"Node directory exists: {os.path.exists(node_dir)}")
    
    # Real data paths
    real_data_paths = [
        os.path.join(datasets_dir, f"hybrid_node_{node_id}", "data.csv"),
        os.path.join(datasets_dir, f"discriminator_{node_id}", "data.csv")
    ]
    
    real_data_path = None
    for path in real_data_paths:
        print(f"Checking real data path: {path}")
        print(f"  Exists: {os.path.exists(path)}")
        if os.path.exists(path):
            real_data_path = path
            break
    
    # Synthetic data path
    synth_files = []
    if os.path.exists(node_dir):
        files = os.listdir(node_dir)
        print(f"Files in node directory: {files}")
        
        for file in files:
            if file.startswith("generated_samples_iter_") and file.endswith(".csv"):
                iteration = int(file.split("_")[-1].split(".")[0])
                synth_files.append((iteration, os.path.join(node_dir, file)))
    
    synth_files.sort(key=lambda x: x[0], reverse=True)
    synth_data_path = synth_files[0][1] if synth_files else None
    
    print(f"Synthetic data path: {synth_data_path}")
    print(f"Available iterations: {[f[0] for f in synth_files]}")
    
    if not real_data_path or not synth_data_path:
        print("? Missing data files!")
        return
    
    # Step 2: Raw file analysis
    print(f"\n2. RAW FILE ANALYSIS")
    print("-" * 30)
    
    print(f"Real data file: {real_data_path}")
    print(f"File size: {os.path.getsize(real_data_path)} bytes")
    
    # Peek at first few lines
    print("First 5 lines of real data file:")
    with open(real_data_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  Line {i}: {line.strip()}")
            else:
                break
    
    print(f"\nSynthetic data file: {synth_data_path}")
    print(f"File size: {os.path.getsize(synth_data_path)} bytes")
    
    print("First 5 lines of synthetic data file:")
    with open(synth_data_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  Line {i}: {line.strip()}")
            else:
                break
    
    # Step 3: DataFrame loading analysis
    print(f"\n3. DATAFRAME LOADING ANALYSIS")
    print("-" * 30)
    
    try:
        print("Loading real data...")
        real_data_original = pd.read_csv(real_data_path)
        print(f"Original real data shape: {real_data_original.shape}")
        print(f"Original real data columns: {list(real_data_original.columns)}")
        print(f"Original real data dtypes:\n{real_data_original.dtypes}")
        
        print(f"\nFirst few rows of real data:")
        print(real_data_original.head())
        
    except Exception as e:
        print(f"? Error loading real data: {e}")
        return
    
    try:
        print(f"\nLoading synthetic data...")
        synth_data_original = pd.read_csv(synth_data_path)
        print(f"Original synthetic data shape: {synth_data_original.shape}")
        print(f"Original synthetic data columns: {list(synth_data_original.columns)}")
        print(f"Original synthetic data dtypes:\n{synth_data_original.dtypes}")
        
        print(f"\nFirst few rows of synthetic data:")
        print(synth_data_original.head())
        
    except Exception as e:
        print(f"? Error loading synthetic data: {e}")
        return
    
    # Step 4: Column compatibility analysis
    print(f"\n4. COLUMN COMPATIBILITY ANALYSIS")
    print("-" * 30)
    
    real_cols = set(real_data_original.columns)
    synth_cols = set(synth_data_original.columns)
    
    print(f"Real data columns ({len(real_cols)}): {sorted(real_cols)}")
    print(f"Synthetic data columns ({len(synth_cols)}): {sorted(synth_cols)}")
    
    common_cols = real_cols & synth_cols
    real_only = real_cols - synth_cols
    synth_only = synth_cols - real_cols
    
    print(f"\nCommon columns ({len(common_cols)}): {sorted(common_cols)}")
    print(f"Real-only columns ({len(real_only)}): {sorted(real_only)}")
    print(f"Synthetic-only columns ({len(synth_only)}): {sorted(synth_only)}")
    
    if len(common_cols) == 0:
        print("? CRITICAL: No common columns between real and synthetic data!")
        print("This explains why evaluation shows 0 columns!")
        return
    
    # Step 5: Data cleaning simulation
    print(f"\n5. DATA CLEANING SIMULATION")
    print("-" * 30)
    
    # Simulate the cleaning process
    print("Simulating evaluation data cleaning process...")
    
    real_data = real_data_original.copy()
    synth_data = synth_data_original.copy()
    
    print(f"Step 1 - Starting shapes: Real {real_data.shape}, Synthetic {synth_data.shape}")
    
    # Find common columns (this is where the issue likely occurs)
    common_columns = list(set(real_data.columns) & set(synth_data.columns))
    print(f"Step 2 - Common columns found: {len(common_columns)}")
    print(f"Common columns: {common_columns}")
    
    if not common_columns:
        print("? FOUND THE ISSUE: No common columns after intersection!")
        print("Possible causes:")
        print("1. Column names don't match exactly")
        print("2. Different column naming conventions")
        print("3. Missing header in one of the files")
        print("4. Different separators or file formats")
        return
    
    real_data = real_data[common_columns]
    synth_data = synth_data[common_columns]
    print(f"Step 3 - After column selection: Real {real_data.shape}, Synthetic {synth_data.shape}")
    
    # Check for NaN/infinite values
    print(f"Step 4 - Checking for problematic values...")
    
    real_nan_cols = real_data.columns[real_data.isnull().all()].tolist()
    synth_nan_cols = synth_data.columns[synth_data.isnull().all()].tolist()
    
    print(f"Real data - all NaN columns: {real_nan_cols}")
    print(f"Synthetic data - all NaN columns: {synth_nan_cols}")
    
    # Replace infinite values
    real_data = real_data.replace([np.inf, -np.inf], np.nan)
    synth_data = synth_data.replace([np.inf, -np.inf], np.nan)
    
    # Drop all-NaN columns
    real_data = real_data.dropna(axis=1, how='all')
    synth_data = synth_data.dropna(axis=1, how='all')
    print(f"Step 5 - After dropping all-NaN columns: Real {real_data.shape}, Synthetic {synth_data.shape}")
    
    # Check remaining common columns
    final_common = list(set(real_data.columns) & set(synth_data.columns))
    print(f"Step 6 - Final common columns: {len(final_common)}")
    print(f"Final common columns: {final_common}")
    
    if len(final_common) == 0:
        print("? FOUND THE ISSUE: All columns dropped during cleaning!")
        print("This explains the 0 columns in evaluation!")
    else:
        print("? Columns should be available for evaluation")
        
        # Check data types
        real_data = real_data[final_common]
        synth_data = synth_data[final_common]
        
        numerical_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in final_common if col not in numerical_cols]
        
        print(f"Numerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")
        
        if len(numerical_cols) == 0:
            print("??  WARNING: No numerical columns for statistical evaluation!")
    
    # Step 6: Recommendations
    print(f"\n6. RECOMMENDATIONS")
    print("-" * 30)
    
    if len(common_cols) == 0:
        print("?? FIXES NEEDED:")
        print("1. Check column naming consistency")
        print("2. Verify CSV headers match exactly")
        print("3. Check for extra spaces or special characters in column names")
        print("4. Ensure both files use same separator (comma vs semicolon)")
    elif len(final_common) == 0:
        print("?? FIXES NEEDED:")
        print("1. Check for all-NaN columns in data")
        print("2. Verify data generation is working properly")
        print("3. Check for infinite values in synthetic data")
    else:
        print("? Data structure looks good!")

def fix_column_issues(base_dir, node_id):
    """Attempt to fix column issues"""
    
    print(f"\n{'='*60}")
    print(f"ATTEMPTING TO FIX COLUMN ISSUES FOR NODE {node_id}")
    print(f"{'='*60}")
    
    results_dir = os.path.join(base_dir, "results")
    datasets_dir = os.path.join(base_dir, "datasets")
    
    # Find data files
    real_data_paths = [
        os.path.join(datasets_dir, f"hybrid_node_{node_id}", "data.csv"),
        os.path.join(datasets_dir, f"discriminator_{node_id}", "data.csv")
    ]
    
    real_data_path = None
    for path in real_data_paths:
        if os.path.exists(path):
            real_data_path = path
            break
    
    node_dir = os.path.join(results_dir, f"hybrid_node_{node_id}")
    synth_files = []
    if os.path.exists(node_dir):
        for file in os.listdir(node_dir):
            if file.startswith("generated_samples_iter_") and file.endswith(".csv"):
                iteration = int(file.split("_")[-1].split(".")[0])
                synth_files.append((iteration, os.path.join(node_dir, file)))
    
    synth_files.sort(key=lambda x: x[0], reverse=True)
    synth_data_path = synth_files[0][1] if synth_files else None
    
    if not real_data_path or not synth_data_path:
        print("? Cannot fix - missing data files")
        return False
    
    try:
        # Load data
        real_data = pd.read_csv(real_data_path)
        synth_data = pd.read_csv(synth_data_path)
        
        print(f"Original shapes: Real {real_data.shape}, Synthetic {synth_data.shape}")
        
        # Fix 1: Standardize column names
        print("1. Standardizing column names...")
        
        # Remove extra spaces and standardize
        real_data.columns = real_data.columns.str.strip().str.lower().str.replace(' ', '_')
        synth_data.columns = synth_data.columns.str.strip().str.lower().str.replace(' ', '_')
        
        print(f"Real columns after standardization: {list(real_data.columns)}")
        print(f"Synthetic columns after standardization: {list(synth_data.columns)}")
        
        # Fix 2: Handle unnamed columns
        real_data = real_data.loc[:, ~real_data.columns.str.contains('^Unnamed')]
        synth_data = synth_data.loc[:, ~synth_data.columns.str.contains('^Unnamed')]
        
        # Fix 3: Ensure common columns exist
        common_cols = list(set(real_data.columns) & set(synth_data.columns))
        print(f"Common columns after cleaning: {common_cols}")
        
        if not common_cols:
            print("? Still no common columns - trying alternative fix...")
            
            # Alternative: Use positional matching if same number of columns
            if len(real_data.columns) == len(synth_data.columns):
                print("Attempting positional column matching...")
                new_columns = [f"feature_{i}" for i in range(len(real_data.columns))]
                real_data.columns = new_columns
                synth_data.columns = new_columns
                print(f"New standardized columns: {new_columns}")
            else:
                print("? Cannot fix - different number of columns")
                return False
        
        # Save fixed files
        backup_dir = os.path.join(node_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup original synthetic data
        import shutil
        shutil.copy(synth_data_path, os.path.join(backup_dir, f"original_{os.path.basename(synth_data_path)}"))
        
        # Save fixed synthetic data
        synth_data.to_csv(synth_data_path, index=False)
        print(f"? Fixed synthetic data saved with shape: {synth_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"? Error during fixing: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Diagnose data column issues')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN',
                       help='Base directory for GAN simulation')
    parser.add_argument('--node_id', type=int, help='Specific node to diagnose')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    parser.add_argument('--check_all', action='store_true', help='Check first few nodes')
    
    args = parser.parse_args()
    
    if args.node_id is not None:
        # Diagnose specific node
        diagnose_node_data_issue(args.base_dir, args.node_id)
        
        if args.fix:
            print("\nAttempting to fix issues...")
            success = fix_column_issues(args.base_dir, args.node_id)
            if success:
                print("? Fix attempt completed - try evaluation again")
            else:
                print("? Fix attempt failed")
    
    elif args.check_all:
        # Check first few nodes to find patterns
        print("Checking first 5 nodes for patterns...")
        for node_id in range(1, 6):
            diagnose_node_data_issue(args.base_dir, node_id)
            print("\n" + "="*60)
    
    else:
        print("Please specify --node_id or --check_all")
        print("Examples:")
        print("  python3 diagnose_data_issue.py --node_id 1")
        print("  python3 diagnose_data_issue.py --check_all")
        print("  python3 diagnose_data_issue.py --node_id 1 --fix")

if __name__ == "__main__":
    main()