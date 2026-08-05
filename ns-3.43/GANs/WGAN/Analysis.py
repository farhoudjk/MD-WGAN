#!/usr/bin/env python3

import sys
import os
import pyshark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants based on your simulation
NUM_GENERATORS = 2
NUM_DISCRIMINATORS = 10
TOTAL_GAN_NODES = NUM_GENERATORS + NUM_DISCRIMINATORS

def extract_node_info(filename):
    """Extract node type and ID from the PCAP filename."""
    try:
        parts = os.path.basename(filename).split('-')
        if 'gen' in parts[3]:
            node_type = 'generator'
            node_id = int(parts[3][3:])
        elif 'disc' in parts[3]:
            node_type = 'discriminator'
            node_id = int(parts[3][4:])
        else:
            node_type = 'unknown'
            node_id = -1
        return node_type, node_id
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return 'unknown', -1

def analyze_pcap_files(pcap_files):
    # Create a dictionary to track all GAN nodes and their IPs
    gan_nodes = {}
    ip_to_node_info = {}
    
    # Initialize data structures
    packets = []
    flows = defaultdict(lambda: {'packets': 0, 'bytes': 0, 'first_time': float('inf'), 
                                 'last_time': 0, 'latencies': [], 'is_gan': False})
    
    # First pass: identify all GAN nodes and their IPs
    print("First pass: Identifying GAN nodes...")
    for pcap_file in pcap_files:
        node_type, node_id = extract_node_info(pcap_file)
        if node_type in ['generator', 'discriminator']:
            # This is a GAN node PCAP
            print(f"Processing {pcap_file} ({node_type} {node_id})...")
            
            try:
                # Just read a few packets to get the node's IP
                cap = pyshark.FileCapture(pcap_file, display_filter='ip', keep_packets=False)
                
                # Look through first few packets to find the node's IP
                for i, packet in enumerate(cap):
                    if hasattr(packet, 'ip'):
                        src_ip = packet.ip.src
                        # If this IP is assigned to a GAN node and not already recorded
                        if src_ip not in ip_to_node_info:
                            gan_nodes[src_ip] = {'type': node_type, 'id': node_id}
                            ip_to_node_info[src_ip] = f"{node_type}-{node_id}"
                            print(f"  Identified {node_type} {node_id} with IP {src_ip}")
                            break
                    if i >= 10:  # Only check first 10 packets
                        break
                
                cap.close()
            except Exception as e:
                print(f"Error in first pass for {pcap_file}: {e}")
    
    # Write the GAN nodes to a file for future reference
    with open('gan_nodes.txt', 'w') as f:
        for ip in gan_nodes:
            f.write(f"{ip}\n")
    print(f"Identified {len(gan_nodes)} GAN nodes, saved to gan_nodes.txt")
    
    # Second pass: analyze all traffic
    print("\nSecond pass: Analyzing traffic...")
    for pcap_file in pcap_files:
        print(f"Processing {pcap_file}...")
        
        # Extract node type and ID
        node_type, node_id = extract_node_info(pcap_file)
        
        # Parse PCAP file
        try:
            cap = pyshark.FileCapture(pcap_file)
            
            for packet in cap:
                try:
                    # Skip non-IP packets
                    if not hasattr(packet, 'ip'):
                        continue
                    
                    # Extract packet info
                    timestamp = float(packet.frame_info.time_epoch)
                    src_ip = packet.ip.src
                    dst_ip = packet.ip.dst
                    
                    # Determine if it's UDP (for port info) or just use IP
                    if hasattr(packet, 'udp'):
                        src_port = int(packet.udp.srcport)
                        dst_port = int(packet.udp.dstport)
                        protocol = 'UDP'
                    elif hasattr(packet, 'tcp'):
                        src_port = int(packet.tcp.srcport)
                        dst_port = int(packet.tcp.dstport)
                        protocol = 'TCP'
                    else:
                        # For non-UDP/TCP, use protocol number as port
                        protocol = packet.ip.proto
                        src_port = 0
                        dst_port = 0
                    
                    # Get packet length (prefer udp/tcp length if available)
                    if hasattr(packet, 'udp'):
                        length = int(packet.udp.length)
                    elif hasattr(packet, 'tcp') and hasattr(packet.tcp, 'len'):
                        length = int(packet.tcp.len)
                    else:
                        length = int(packet.ip.len)
                    
                    # Check if this is GAN traffic
                    is_gan = src_ip in gan_nodes or dst_ip in gan_nodes
                    
                    # Create flow key (in both directions)
                    flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                    
                    # Update flow statistics
                    flows[flow_key]['packets'] += 1
                    flows[flow_key]['bytes'] += length
                    flows[flow_key]['first_time'] = min(flows[flow_key]['first_time'], timestamp)
                    flows[flow_key]['last_time'] = max(flows[flow_key]['last_time'], timestamp)
                    flows[flow_key]['is_gan'] = is_gan
                    
                    # Source node info for better reporting
                    src_node_type = 'unknown'
                    src_node_id = -1
                    if src_ip in gan_nodes:
                        src_node_type = gan_nodes[src_ip]['type']
                        src_node_id = gan_nodes[src_ip]['id']
                    
                    # Destination node info
                    dst_node_type = 'unknown'
                    dst_node_id = -1
                    if dst_ip in gan_nodes:
                        dst_node_type = gan_nodes[dst_ip]['type']
                        dst_node_id = gan_nodes[dst_ip]['id']
                    
                    # Store packet info
                    packets.append({
                        'timestamp': timestamp,
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'src_port': src_port,
                        'dst_port': dst_port,
                        'protocol': protocol,
                        'length': length,
                        'is_gan': is_gan,
                        'src_node_type': src_node_type,
                        'src_node_id': src_node_id,
                        'dst_node_type': dst_node_type,
                        'dst_node_id': dst_node_id
                    })
                    
                except Exception as e:
                    print(f"  Error processing packet: {e}")
                    continue
            
            cap.close()
        except Exception as e:
            print(f"Error processing file {pcap_file}: {e}")
    
    # Convert packets to DataFrame
    df = pd.DataFrame(packets)
    
    # Calculate flow statistics
    flow_stats = []
    for flow_key, stats in flows.items():
        src_ip, dst_ip, src_port, dst_port, protocol = flow_key
        
        # Calculate duration and throughput
        duration = stats['last_time'] - stats['first_time']
        if duration <= 0:
            duration = 0.001  # Avoid division by zero
        
        throughput = (stats['bytes'] * 8) / (duration * 1000)  # Kbps
        
        # Get source and destination node info if available
        src_node_info = ip_to_node_info.get(src_ip, 'unknown')
        dst_node_info = ip_to_node_info.get(dst_ip, 'unknown')
        
        # Create flow record
        flow_stats.append({
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'packets': stats['packets'],
            'bytes': stats['bytes'],
            'duration': duration,
            'throughput_kbps': throughput,
            'is_gan': stats['is_gan'],
            'src_node': src_node_info,
            'dst_node': dst_node_info
        })
    
    # Convert to DataFrame
    df_flows = pd.DataFrame(flow_stats)
    
    # Group packets by different criteria for analysis
    if not df.empty:
        # 1. Calculate overall network statistics
        total_bytes = df['length'].sum()
        gan_bytes = df[df['is_gan']]['length'].sum()
        background_bytes = total_bytes - gan_bytes
        
        total_packets = len(df)
        gan_packets = len(df[df['is_gan']])
        background_packets = total_packets - gan_packets
        
        # 2. Calculate generator-discriminator specific statistics
        # Create dataframes for specific traffic types
        df_gen_to_disc = df[
            (df['src_node_type'] == 'generator') & 
            (df['dst_node_type'] == 'discriminator')
        ]
        
        df_disc_to_gen = df[
            (df['src_node_type'] == 'discriminator') & 
            (df['dst_node_type'] == 'generator')
        ]
        
        df_gen_to_gen = df[
            (df['src_node_type'] == 'generator') & 
            (df['dst_node_type'] == 'generator')
        ]
        
        # Get latency estimates by looking at communication patterns
        # Calculate latency for discriminator responses (time between discriminator receiving
        # a packet from a generator and sending a response back)
        latency_stats = []
        
        # Get each generator ID
        for gen_id in range(1, NUM_GENERATORS + 1):
            # For each discriminator associated with this generator
            disc_range = range(1, 6) if gen_id == 1 else range(6, 11)
            for disc_id in disc_range:
                # Find traffic from generator to this discriminator
                gen_to_disc = df[
                    (df['src_node_type'] == 'generator') & 
                    (df['src_node_id'] == gen_id) &
                    (df['dst_node_type'] == 'discriminator') & 
                    (df['dst_node_id'] == disc_id)
                ]
                
                # Find traffic from this discriminator back to generator
                disc_to_gen = df[
                    (df['src_node_type'] == 'discriminator') & 
                    (df['src_node_id'] == disc_id) &
                    (df['dst_node_type'] == 'generator') & 
                    (df['dst_node_id'] == gen_id)
                ]
                
                # If we have both types of traffic, estimate latency
                if not gen_to_disc.empty and not disc_to_gen.empty:
                    # Sort by timestamp
                    gen_to_disc = gen_to_disc.sort_values('timestamp')
                    disc_to_gen = disc_to_gen.sort_values('timestamp')
                    
                    # For each packet from generator to discriminator
                    response_times = []
                    for _, gen_packet in gen_to_disc.iterrows():
                        # Find the first response after this packet
                        possible_responses = disc_to_gen[
                            disc_to_gen['timestamp'] > gen_packet['timestamp']
                        ]
                        
                        if not possible_responses.empty:
                            # Take the first response
                            response = possible_responses.iloc[0]
                            response_time = response['timestamp'] - gen_packet['timestamp']
                            
                            # Only count reasonable response times (< 10 seconds)
                            if response_time < 10:
                                response_times.append(response_time)
                    
                    # Calculate average response time if we have data
                    if response_times:
                        avg_latency = sum(response_times) / len(response_times)
                        min_latency = min(response_times)
                        max_latency = max(response_times)
                        
                        latency_stats.append({
                            'generator_id': gen_id,
                            'discriminator_id': disc_id,
                            'samples': len(response_times),
                            'avg_latency_ms': avg_latency * 1000,  # Convert to ms
                            'min_latency_ms': min_latency * 1000,
                            'max_latency_ms': max_latency * 1000
                        })
        
        # Create latency DataFrame
        df_latency = pd.DataFrame(latency_stats)
        
        # 3. Output results
        print("\n=== Network Analysis Results ===")
        print(f"\nOverall Network Statistics:")
        print(f"Total packets: {total_packets}")
        print(f"Total bytes: {total_bytes}")
        print(f"GAN traffic: {gan_packets} packets, {gan_bytes} bytes ({gan_bytes/total_bytes*100:.2f}% of total)")
        print(f"Background traffic: {background_packets} packets, {background_bytes} bytes ({background_bytes/total_bytes*100:.2f}% of total)")
        
        # Traffic breakdown by node type
        print("\nTraffic Breakdown:")
        print(f"Generator ? Discriminator: {len(df_gen_to_disc)} packets, {df_gen_to_disc['length'].sum()} bytes")
        print(f"Discriminator ? Generator: {len(df_disc_to_gen)} packets, {df_disc_to_gen['length'].sum()} bytes")
        print(f"Generator ? Generator: {len(df_gen_to_gen)} packets, {df_gen_to_gen['length'].sum()} bytes")
        
        # Latency statistics if available
        if not df_latency.empty:
            print("\nEstimated Latency Statistics:")
            print(f"Average latency across all pairs: {df_latency['avg_latency_ms'].mean():.2f} ms")
            print(f"Min latency: {df_latency['min_latency_ms'].min():.2f} ms")
            print(f"Max latency: {df_latency['max_latency_ms'].max():.2f} ms")
        
        # 4. Save results to CSV files
        # Overall network statistics
        with open('network_overhead.csv', 'w') as f:
            f.write("TrafficType,Packets,Bytes,Percentage\n")
            f.write(f"GAN Traffic,{gan_packets},{gan_bytes},{gan_bytes/total_bytes*100:.2f}\n")
            f.write(f"Background Traffic,{background_packets},{background_bytes},{background_bytes/total_bytes*100:.2f}\n")
            f.write(f"Total,{total_packets},{total_bytes},100.0\n")
        
        # Flow statistics
        df_flows.to_csv('flow_statistics.csv', index=False)
        
        # Latency statistics
        if not df_latency.empty:
            df_latency.to_csv('latency_statistics.csv', index=False)
        
        # 5. Create visualizations
        # Network overhead pie chart
        plt.figure(figsize=(8, 6))
        plt.pie([gan_bytes, background_bytes], 
                labels=['GAN Traffic', 'Background Traffic'],
                autopct='%1.1f%%', startangle=90)
        plt.title('Network Traffic Distribution (Bytes)')
        plt.savefig('network_overhead_bytes.png')
        
        # Packet distribution pie chart
        plt.figure(figsize=(8, 6))
        plt.pie([gan_packets, background_packets], 
                labels=['GAN Traffic', 'Background Traffic'],
                autopct='%1.1f%%', startangle=90)
        plt.title('Network Traffic Distribution (Packets)')
        plt.savefig('network_overhead_packets.png')
        
        # GAN traffic breakdown
        if not df_gen_to_disc.empty or not df_disc_to_gen.empty or not df_gen_to_gen.empty:
            plt.figure(figsize=(10, 6))
            traffic_types = ['Generator ? Discriminator', 'Discriminator ? Generator', 'Generator ? Generator']
            traffic_bytes = [df_gen_to_disc['length'].sum(), df_disc_to_gen['length'].sum(), df_gen_to_gen['length'].sum()]
            
            plt.bar(traffic_types, traffic_bytes)
            plt.ylabel('Bytes')
            plt.title('GAN Traffic Breakdown')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('gan_traffic_breakdown.png')
        
        # Latency visualization
        if not df_latency.empty:
            plt.figure(figsize=(12, 6))
            
            # Sort by discriminator ID for clearer visualization
            df_latency_sorted = df_latency.sort_values(['generator_id', 'discriminator_id'])
            
            # Create labels
            labels = [f"G{row['generator_id']}-D{row['discriminator_id']}" 
                     for _, row in df_latency_sorted.iterrows()]
            
            plt.bar(labels, df_latency_sorted['avg_latency_ms'])
            plt.errorbar(labels, df_latency_sorted['avg_latency_ms'],
                         yerr=[(df_latency_sorted['avg_latency_ms'] - df_latency_sorted['min_latency_ms']),
                              (df_latency_sorted['max_latency_ms'] - df_latency_sorted['avg_latency_ms'])],
                         fmt='o', color='r', ecolor='black', capsize=5)
            
            plt.ylabel('Latency (ms)')
            plt.title('Average Communication Latency between Generators and Discriminators')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('latency_statistics.png')
        
        print("\nAnalysis complete. Results saved to CSV files and visualizations generated.")
    else:
        print("No packets found for analysis.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pcap_file1> <pcap_file2> ...")
        sys.exit(1)
    
    pcap_files = sys.argv[1:]
    analyze_pcap_files(pcap_files)