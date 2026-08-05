import pandas as pd

# Specify the input and output file paths
txt_file = '/home/yorku/Downloads/ns-allinone-3.43/ns-3.43/GANs/WGAN/trajectories-0750am-0805am.txt'
csv_file = '/home/yorku/Downloads/ns-allinone-3.43/ns-3.43/raw_traffic_data.csv'

# Define column names clearly matching your dataset
columns = [
    'vehicle_id', 'frame_id', 'total_frames', 'time', 'position_x', 'position_y',
    'global_x', 'global_y', 'vehicle_length', 'vehicle_width', 'vehicle_class',
    'speed', 'acceleration', 'lane_id', 'preceding_vehicle_id',
    'following_vehicle_id', 'spacing'
]

# Read the TXT file assuming whitespace-separated values
data = pd.read_csv(txt_file, delim_whitespace=True, header=None, names=columns, engine='python')

# Save the data into CSV file without additional commas
data.to_csv(csv_file, index=False)

print(f"Dataset successfully converted and saved to: {csv_file}")
