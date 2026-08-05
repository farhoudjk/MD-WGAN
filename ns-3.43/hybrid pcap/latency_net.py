import sys
from scapy.all import rdpcap, IP, UDP, Raw
from collections import defaultdict
import numpy as np

# Constants from NS-3 code
NUM_HYBRID_NODES = 20
BASE_IP = "10.1.1."
NODE_IPS = [f"{BASE_IP}{i+1}" for i in range(NUM_HYBRID_NODES)]  # 10.1.1.1 to 10.1.1.20
PORT = 9
ROUND_INTERVAL = 15.0  # From Interval=Seconds(15.0)

def load_pcap_files():
    """Load all 20 PCAP files."""
    pcap_data = {}
    for i in range(1, NUM_HYBRID_NODES + 1):
        pcap_name = f"hybrid-gan-node-{i}-{i-1}-0.pcap"
        try:
            pcap_data[f'node_{i}'] = rdpcap(pcap_name)
        except FileNotFoundError:
            print(f"Warning: PCAP file {pcap_name} not found.")
            continue
    print(f"Loaded {len(pcap_data)} PCAP files.")
    return pcap_data

def parse_hybrid_payload(payload):
    """Parse payload like 'GEN:1.23,DISC:4.56' to (gen_loss, disc_loss)."""
    try:
        parts = payload.split(',')
        gen_part = parts[0].split(':')[1]
        disc_part = parts[1].split(':')[1]
        return float(gen_part), float(disc_part)
    except (IndexError, ValueError):
        return None, None

def match_packets(pcap_data):
    """Group packets into rounds and collect sends/receives."""
    rounds = defaultdict(lambda: {'sends': [], 'receives': []})
    packet_log = []

    for node_id in range(1, NUM_HYBRID_NODES + 1):
        node_key = f'node_{node_id}'
        if node_key not in pcap_data:
            continue
        node_ip = NODE_IPS[node_id - 1]
        for pkt in pcap_data[node_key]:
            if IP not in pkt or UDP not in pkt:
                packet_log.append(f"Node {node_id}: Non-IP/UDP packet at {pkt.time:.3f}s")
                continue
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
            if sport != PORT and dport != PORT:
                packet_log.append(f"Node {node_id}: Wrong port {sport}/{dport} at {pkt.time:.3f}s")
                continue
            time = pkt.time
            round_id = int(time // ROUND_INTERVAL)
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            if pkt.haslayer(Raw):
                raw_payload = pkt[Raw].load
                payload = raw_payload.decode('utf-8', errors='ignore').strip('\x00')
            else:
                payload = ''
                packet_log.append(f"Node {node_id}: No Raw layer at {time:.3f}s (sport/dport: {sport}/{dport})")
                continue
            gen_loss, disc_loss = parse_hybrid_payload(payload)
            if gen_loss is None or disc_loss is None:
                packet_log.append(f"Node {node_id}: Invalid payload '{payload}' at {time:.3f}s (raw: {raw_payload!r})")
                continue
            packet_log.append(f"Node {node_id}: Valid packet src={src_ip} dst={dst_ip} gen_loss={gen_loss:.4f} disc_loss={disc_loss:.4f} at {time:.3f}s (sport/dport: {sport}/{dport})")
            
            # If outgoing (send from this node)
            if src_ip == node_ip and dst_ip in NODE_IPS:
                rounds[round_id]['sends'].append({
                    'time': time,
                    'src_id': node_id,
                    'dst_ip': dst_ip,
                    'gen_loss': gen_loss,
                    'disc_loss': disc_loss
                })
            # If incoming (receive at this node)
            elif dst_ip == node_ip and src_ip in NODE_IPS:
                rounds[round_id]['receives'].append({
                    'time': time,
                    'src_ip': src_ip,
                    'dst_id': node_id,
                    'gen_loss': gen_loss,
                    'disc_loss': disc_loss
                })

    # Save packet log
    with open("packet_log.txt", "w") as f:
        f.write("\n".join(packet_log))
    print(f"Packet log saved to packet_log.txt")

    return rounds

def calculate_round_latency(rounds):
    """Calculate E2E network latency per round across all pairs."""
    round_latencies = {}
    max_feedback_delay = 30.0  # Allow up to 2 rounds later (15s * 2)
    for round_id, data in rounds.items():
        latencies = []
        for send in data['sends']:
            send_time = float(send['time'])
            src_id = send['src_id']
            dst_ip = send['dst_ip']
            dst_id = NODE_IPS.index(dst_ip) + 1
            # Find corresponding receive at dst in same or next rounds
            matching_receives = []
            for r_id in range(round_id, round_id + 3):  # Check current + 2 rounds
                if r_id in rounds:
                    matching_receives.extend([
                        recv for recv in rounds[r_id]['receives']
                        if recv['dst_id'] == dst_id and recv['src_ip'] == NODE_IPS[src_id - 1] and float(recv['time']) > send_time and float(recv['time']) <= send_time + max_feedback_delay
                    ])
            if matching_receives:
                receive = min(matching_receives, key=lambda x: float(x['time']))
                latency = float(receive['time']) - send_time
                latencies.append(latency)
                print(f"Round {round_id}: Matched send from node {src_id} to node {dst_id} (time {send_time:.3f}) to receive (time {receive['time']:.3f}), latency {latency:.6f}s")
        round_latencies[round_id] = {
            'avg_latency': float(np.mean(latencies)) if latencies else 0.0,
            'num_pairs': len(latencies),
            'latencies': latencies
        }
    return round_latencies

def main():
    print("Loading PCAP files...")
    pcap_data = load_pcap_files()

    print("Matching packets into rounds...")
    rounds = match_packets(pcap_data)

    print("Calculating per-round latencies...")
    round_latencies = calculate_round_latency(rounds)

    print("\nPer-Round E2E Network Latencies (in simulated seconds):")
    if not round_latencies:
        print("No rounds found with matching packets. Check packet_log.txt for details.")
    else:
        for round_id in sorted(round_latencies.keys()):
            latency_info = round_latencies[round_id]
            print(f"Round {round_id} (~{round_id*15}s to {(round_id+1)*15}s):")
            print(f"  Average Latency: {latency_info['avg_latency']:.6f}s")
            print(f"  Number of Sample-Feedback Pairs: {latency_info['num_pairs']}")
            print(f"  Individual Latencies: {latency_info['latencies']}")

if __name__ == "__main__":
    main()