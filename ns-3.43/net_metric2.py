import os
from scapy.all import rdpcap, IP, Raw
import sys
from collections import defaultdict
import datetime

def calculate_network_metrics(pcap_files):
    """
    Analyzes a list of pcap files to calculate total network traffic,
    data payload, network overhead ratio, E2E latency, and per-second metrics.

    Args:
        pcap_files (list): A list of file paths to the pcap files.

    Returns:
        dict: A dictionary containing all the calculated metrics.
    """
    total_network_bytes = 0
    total_data_bytes = 0
    total_packets = 0
    
    # Dictionary to track packet flows for latency calculation
    # Key: a unique identifier for the packet (e.g., IP ID)
    # Value: a dict with 'start' and 'end' timestamps
    packet_flows = {}
    
    # Dictionary to track network traffic by the second of the simulation
    # Key: integer representing the second of the simulation
    # Value: total network bytes for that second
    traffic_by_second = defaultdict(int)

    # Track the first and last timestamps to calculate total simulation duration
    first_timestamp = float('inf')
    last_timestamp = 0

    print("Starting analysis of PCAP files...")
    # Check if the list of pcap files is empty
    if not pcap_files:
        print("No .pcap files found in the current directory.")
        return {
            'total_packets': 0, 'total_network_bytes': 0, 'total_data_bytes': 0,
            'overhead_ratio': 0, 'average_e2e_latency': 0, 'total_duration': 0,
            'network_bytes_per_sec': 0, 'data_bytes_per_sec': 0,
            'overhead_per_sec': 0, 'cumulative_traffic_data': []
        }

    for file_name in pcap_files:
        if not os.path.exists(file_name):
            print(f"Warning: File not found - {file_name}. Skipping...")
            continue

        try:
            # rdpcap reads all packets from the specified pcap file.
            packets = rdpcap(file_name)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}. Skipping...")
            continue
        
        # Iterate over each packet in the file.
        try:
            for pkt in packets:
                # Update first and last timestamps
                if pkt.time < first_timestamp:
                    first_timestamp = pkt.time
                if pkt.time > last_timestamp:
                    last_timestamp = pkt.time
                
                total_packets += 1
                
                # The total size of the packet on the wire is its length.
                pkt_size = len(pkt)
                total_network_bytes += pkt_size
                
                # We need to find the actual data payload.
                if IP in pkt and Raw in pkt:
                    total_data_bytes += len(pkt[Raw].load)
                elif IP in pkt and len(pkt[IP].payload) > 0:
                    # Fallback for cases where 'Raw' layer is not explicitly present
                    total_data_bytes += len(pkt[IP].payload)

                # E2E Latency Calculation (using IP ID for a simple flow identifier)
                if IP in pkt:
                    packet_id = pkt[IP].id
                    timestamp = pkt.time
                    
                    if packet_id not in packet_flows:
                        packet_flows[packet_id] = {'start': timestamp, 'end': timestamp}
                    else:
                        # Update the end time with the latest timestamp seen for this packet ID
                        packet_flows[packet_id]['end'] = timestamp
                
                # Calculate traffic by second of the simulation
                if first_timestamp != float('inf'):
                    time_in_seconds = int(pkt.time - first_timestamp)
                    traffic_by_second[time_in_seconds] += pkt_size
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting gracefully.")
            sys.exit(0)

    print("Analysis complete.")
    
    # Calculate the average E2E latency
    total_latency = 0
    completed_flows = 0
    for flow_id, timestamps in packet_flows.items():
        # A 'completed flow' for this simple method is a packet seen more than once
        if timestamps['end'] > timestamps['start']:
            latency = timestamps['end'] - timestamps['start']
            total_latency += latency
            completed_flows += 1
            
    average_latency = 0
    if completed_flows > 0:
        average_latency = total_latency / completed_flows

    # Calculate total duration of the simulation
    total_duration = 0
    if first_timestamp != float('inf') and last_timestamp > first_timestamp:
        total_duration = last_timestamp - first_timestamp

    # Calculate per-second metrics
    network_bytes_per_sec = 0
    data_bytes_per_sec = 0
    if total_duration > 0:
        network_bytes_per_sec = total_network_bytes / total_duration
        data_bytes_per_sec = total_data_bytes / total_duration
        
    overhead_per_sec = network_bytes_per_sec - data_bytes_per_sec

    # Calculate the overhead ratio.
    overhead_ratio = 0
    if total_data_bytes > 0:
        overhead_ratio = (total_network_bytes - total_data_bytes) / total_data_bytes
    
    # Create the cumulative traffic data list
    cumulative_traffic_data = []
    current_cumulative_bytes = 0
    for second in sorted(traffic_by_second.keys()):
        current_cumulative_bytes += traffic_by_second[second]
        cumulative_traffic_data.append({'time_s': second, 'cumulative_bytes': current_cumulative_bytes})

    results = {
        'total_packets': total_packets,
        'total_network_bytes': total_network_bytes,
        'total_data_bytes': total_data_bytes,
        'overhead_ratio': overhead_ratio,
        'average_e2e_latency': average_latency,
        'total_duration': total_duration,
        'network_bytes_per_sec': network_bytes_per_sec,
        'data_bytes_per_sec': data_bytes_per_sec,
        'overhead_per_sec': overhead_per_sec,
        'cumulative_traffic_data': cumulative_traffic_data
    }
    
    return results

def generate_csv(data_list):
    """
    Generates a CSV string from a list of dictionaries.
    """
    if not data_list:
        return "Time (s),Cumulative Traffic (Bytes)\n"
    
    headers = "Time (s),Cumulative Traffic (Bytes)\n"
    rows = [f"{item['time_s']},{item['cumulative_bytes']}" for item in data_list]
    return headers + "\n".join(rows)

if __name__ == '__main__':
    # Automatically find all files ending in .pcap in the current directory
    pcap_file_list = [f for f in os.listdir('.') if f.endswith('.pcap')]
    
    metrics = calculate_network_metrics(pcap_file_list)
    csv_output = generate_csv(metrics['cumulative_traffic_data'])

    # Convert the byte counts to KB and MB for readability.
    total_network_kb = metrics['total_network_bytes'] / 1024
    total_network_mb = metrics['total_network_bytes'] / (1024 * 1024)
    total_data_kb = metrics['total_data_bytes'] / 1024
    total_data_mb = metrics['total_data_bytes'] / (1024 * 1024)
    network_kb_per_sec = metrics['network_bytes_per_sec'] / 1024
    data_kb_per_sec = metrics['data_bytes_per_sec'] / 1024
    overhead_kb_per_sec = metrics['overhead_per_sec'] / 1024

    print("\n=================================")
    print("       Analysis Results")
    print("=================================")
    print(f"Total Packets Transmitted: {metrics['total_packets']}")
    print(f"Total Simulation Duration: {metrics['total_duration']:.4f} seconds")
    print("---------------------------------")
    print("Network Traffic:")
    print(f"  Total Network Traffic: {total_network_kb:.2f} KB ({total_network_mb:.2f} MB)")
    print(f"  Total Data Payload: {total_data_kb:.2f} KB ({total_data_mb:.2f} MB)")
    print(f"  Traffic Rate: {network_kb_per_sec:.2f} KB/s")
    print("---------------------------------")
    print("Network Overhead:")
    print(f"  Overhead Ratio: {metrics['overhead_ratio']:.4f}")
    print(f"  Overhead Rate: {overhead_kb_per_sec:.2f} KB/s")
    print("---------------------------------")
    print("Latency:")
    print(f"  Average E2E Latency: {metrics['average_e2e_latency']:.6f} seconds")
    print("---------------------------------")
    print("Cumulative Traffic Data (CSV Format):")
    print(csv_output)
    print("=================================\n")
