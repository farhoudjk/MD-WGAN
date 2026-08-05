import pandas as pd
import numpy as np
import os

def diagnose_data_mismatch(real_data_path, generated_data_path):
    """Diagnose the extreme RMSE values by comparing data scales and distributions"""
    
    print("=" * 60)
    print("GAN DATA DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Load sample real data
    real_file = os.path.join(real_data_path, "discriminator_1", "data.csv")
    if os.path.exists(real_file):
        real_df = pd.read_csv(real_file)
        print(f"\n?? REAL DATA ANALYSIS (discriminator_1)")
        print("-" * 40)
        print(f"Shape: {real_df.shape}")
        print(f"Columns: {list(real_df.columns)}")
        
        # Show statistics for key features
        key_features = ['time', 'position_x', 'position_y', 'global_x', 'global_y', 
                       'vehicle_length', 'vehicle_width', 'speed', 'acceleration', 'spacing']
        
        print(f"\n?? REAL DATA STATISTICS:")
        for feature in key_features:
            if feature in real_df.columns:
                values = real_df[feature].dropna()
                print(f"{feature:15s}: min={values.min():12.3f}, max={values.max():12.3f}, "
                      f"mean={values.mean():12.3f}, std={values.std():12.3f}")
        
        # Show first few rows
        print(f"\n?? FIRST 3 ROWS OF REAL DATA:")
        for feature in key_features[:5]:
            if feature in real_df.columns:
                print(f"{feature:15s}: {real_df[feature].iloc[:3].tolist()}")
    
    # Load sample generated data
    gen_file = os.path.join(generated_data_path, "generator_1", "generated_samples_iter_1.csv")
    if not os.path.exists(gen_file):
        # Try different iteration numbers
        for i in [1, 10, 25, 50, 75]:
            test_file = os.path.join(generated_data_path, "generator_1", f"generated_samples_iter_{i}.csv")
            if os.path.exists(test_file):
                gen_file = test_file
                break
    
    if os.path.exists(gen_file):
        gen_df = pd.read_csv(gen_file)
        print(f"\n?? GENERATED DATA ANALYSIS (generator_1)")
        print("-" * 40)
        print(f"Shape: {gen_df.shape}")
        print(f"Columns: {list(gen_df.columns)}")
        
        # Column mapping
        column_mapping = {
            'time': 3, 'position_x': 4, 'position_y': 5, 'global_x': 6, 'global_y': 7,
            'vehicle_length': 8, 'vehicle_width': 9, 'speed': 11, 'acceleration': 12, 'spacing': 16
        }
        
        print(f"\n?? GENERATED DATA STATISTICS (mapped to real features):")
        for feature, col_idx in column_mapping.items():
            if col_idx in gen_df.columns:
                values = gen_df[col_idx].dropna()
                print(f"{feature:15s}: min={values.min():12.3f}, max={values.max():12.3f}, "
                      f"mean={values.mean():12.3f}, std={values.std():12.3f}")
        
        # Show first few rows
        print(f"\n?? FIRST 3 ROWS OF GENERATED DATA (mapped):")
        for feature, col_idx in list(column_mapping.items())[:5]:
            if col_idx in gen_df.columns:
                print(f"{feature:15s}: {gen_df[col_idx].iloc[:3].tolist()}")
        
        # Direct comparison
        print(f"\n??  SCALE COMPARISON:")
        print("-" * 40)
        
        if os.path.exists(real_file):
            for feature, col_idx in column_mapping.items():
                if feature in real_df.columns and col_idx in gen_df.columns:
                    real_mean = real_df[feature].mean()
                    gen_mean = gen_df[col_idx].mean()
                    ratio = abs(gen_mean / real_mean) if real_mean != 0 else float('inf')
                    
                    print(f"{feature:15s}: Real={real_mean:12.3f}, Gen={gen_mean:12.3f}, "
                          f"Ratio={ratio:8.2f}x")
                    
                    if ratio > 1000 or ratio < 0.001:
                        print(f"                    ??  EXTREME SCALE DIFFERENCE!")
        
        # Check for obvious issues
        print(f"\n?? POTENTIAL ISSUES:")
        print("-" * 40)
        
        issues_found = []
        
        # Check if generated data has extreme values
        for col in gen_df.select_dtypes(include=[np.number]).columns:
            values = gen_df[col].dropna()
            if len(values) > 0:
                if abs(values.max()) > 1e6 or abs(values.min()) > 1e6:
                    issues_found.append(f"Column {col}: Extreme values (max={values.max():.2e}, min={values.min():.2e})")
                
                if values.std() > 1e6:
                    issues_found.append(f"Column {col}: Extremely high variance (std={values.std():.2e})")
        
        # Check if real data has extreme values  
        if os.path.exists(real_file):
            for col in real_df.select_dtypes(include=[np.number]).columns:
                values = real_df[col].dropna()
                if len(values) > 0:
                    if abs(values.max()) > 1e6 or abs(values.min()) > 1e6:
                        issues_found.append(f"Real {col}: Extreme values (max={values.max():.2e}, min={values.min():.2e})")
        
        if issues_found:
            for issue in issues_found:
                print(f"??  {issue}")
        else:
            print("? No obvious extreme value issues detected")
        
        # Recommendations
        print(f"\n?? RECOMMENDATIONS:")
        print("-" * 40)
        print("1. Check if data needs normalization/standardization")
        print("2. Verify coordinate systems match (global vs local)")
        print("3. Check if generated data is in correct units")
        print("4. Consider using relative RMSE instead of absolute")
        print("5. Verify column mapping is correct")
        
        return real_df, gen_df
    
    else:
        print(f"? Generated data file not found: {gen_file}")
        return None, None

def calculate_relative_rmse(real_df, gen_df):
    """Calculate relative RMSE (RMSE / mean of real data) for better interpretation"""
    
    column_mapping = {
        'time': 3, 'position_x': 4, 'position_y': 5, 'global_x': 6, 'global_y': 7,
        'vehicle_length': 8, 'vehicle_width': 9, 'speed': 11, 'acceleration': 12, 'spacing': 16
    }
    
    print(f"\n?? RELATIVE RMSE ANALYSIS:")
    print("-" * 40)
    
    for feature, col_idx in column_mapping.items():
        if feature in real_df.columns and col_idx in gen_df.columns:
            real_values = real_df[feature].dropna()
            gen_values = gen_df[col_idx].dropna()
            
            if len(real_values) > 0 and len(gen_values) > 0:
                # Sample same number of points
                n_samples = min(len(real_values), len(gen_values), 1000)
                real_sample = real_values.sample(n=n_samples, random_state=42)
                gen_sample = gen_values.sample(n=n_samples, random_state=42)
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((real_sample.values - gen_sample.values) ** 2))
                
                # Calculate relative RMSE
                real_mean = abs(real_sample.mean())
                relative_rmse = (rmse / real_mean * 100) if real_mean > 0 else float('inf')
                
                print(f"{feature:15s}: RMSE={rmse:12.3e}, Relative RMSE={relative_rmse:6.1f}%")

# Main execution
if __name__ == "__main__":
    real_data_path = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets"
    generated_data_path = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results"
    
    real_df, gen_df = diagnose_data_mismatch(real_data_path, generated_data_path)
    
    if real_df is not None and gen_df is not None:
        calculate_relative_rmse(real_df, gen_df)