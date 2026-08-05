import os
from scapy.all import rdpcap, IP, Raw

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
    
    # Track the first and last timestamps to calculate total simulation duration
    first_timestamp = float('inf')
    last_timestamp = 0

    print("Starting analysis of PCAP files...")
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
        for pkt in packets:
            total_packets += 1
            
            # Update first and last timestamps
            if pkt.time < first_timestamp:
                first_timestamp = pkt.time
            if pkt.time > last_timestamp:
                last_timestamp = pkt.time
            
            # The total size of the packet on the wire is its length.
            # This includes all headers (Ethernet, IP, TCP/UDP, etc.)
            total_network_bytes += len(pkt)
            
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

    results = {
        'total_packets': total_packets,
        'total_network_bytes': total_network_bytes,
        'total_data_bytes': total_data_bytes,
        'overhead_ratio': overhead_ratio,
        'average_e2e_latency': average_latency,
        'total_duration': total_duration,
        'network_bytes_per_sec': network_bytes_per_sec,
        'data_bytes_per_sec': data_bytes_per_sec,
        'overhead_per_sec': overhead_per_sec
    }
    
    return results

if __name__ == '__main__':
    # List all 20 of your pcap files for analysis.
    pcap_file_list = [
        "hybrid-gan-node-1-0-0.pcap", "hybrid-gan-node-2-1-0.pcap",
        "hybrid-gan-node-3-2-0.pcap", "hybrid-gan-node-4-3-0.pcap",
        "hybrid-gan-node-5-4-0.pcap", "hybrid-gan-node-6-5-0.pcap",
        "hybrid-gan-node-7-6-0.pcap", "hybrid-gan-node-8-7-0.pcap",
        "hybrid-gan-node-9-8-0.pcap", "hybrid-gan-node-10-9-0.pcap",
        "hybrid-gan-node-11-10-0.pcap", "hybrid-gan-node-12-11-0.pcap",
        "hybrid-gan-node-13-12-0.pcap", "hybrid-gan-node-14-13-0.pcap",
        "hybrid-gan-node-15-14-0.pcap", "hybrid-gan-node-16-15-0.pcap",
        "hybrid-gan-node-17-16-0.pcap", "hybrid-gan-node-18-17-0.pcap",
        "hybrid-gan-node-19-18-0.pcap", "hybrid-gan-node-20-19-0.pcap"
    ]
    
    metrics = calculate_network_metrics(pcap_file_list)

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
    print("=================================\n")
