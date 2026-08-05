import sys
from scapy.all import rdpcap, IP, UDP, Raw
from collections import defaultdict
import numpy as np

def load_pcap_files(gen_pcap, disc_pcap_list):
    """Load PCAP files for generator and discriminators."""
    pcap_data = {'generator': rdpcap(gen_pcap)}
    for i, disc_pcap in enumerate(disc_pcap_list, 1):
        pcap_data[f'discriminator_{i}'] = rdpcap(disc_pcap)
    print(f"Loaded PCAPs: generator ({len(pcap_data['generator'])} packets), "
          f"discriminators {[len(pcap_data[f'discriminator_{i}']) for i in range(1, len(disc_pcap_list) + 1)]}")
    return pcap_data

def match_packets(pcap_data, gen_ip="10.1.1.1", disc_ips=["10.1.1.5", "10.1.1.6", "10.1.1.7", "10.1.1.8", "10.1.1.9"], port=9):
    """Group packets into rounds and match sample/feedback packets."""
    round_interval = 20.0  # Round duration in seconds
    rounds = defaultdict(lambda: {'samples': [], 'feedbacks': []})
    packet_log = []

    # Process generator PCAP
    for pkt in pcap_data['generator']:
        if IP not in pkt or UDP not in pkt:
            packet_log.append(f"Generator: Non-IP/UDP packet at {pkt.time:.3f}s")
            continue
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
        if sport != port and dport != port:
            packet_log.append(f"Generator: Wrong port {sport}/{dport} at {pkt.time:.3f}s")
            continue
        time = pkt.time
        round_id = int(time // round_interval)
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        if pkt.haslayer(Raw):
            raw_payload = pkt[Raw].load
            payload = raw_payload.decode('utf-8', errors='ignore').strip('\x00')
        else:
            payload = ''
            packet_log.append(f"Generator: No Raw layer at {time:.3f}s (sport/dport: {sport}/{dport})")
            continue
        try:
            loss_value = float(payload)
            packet_log.append(f"Generator: Valid packet src={src_ip} dst={dst_ip} loss={loss_value:.4f} at {time:.3f}s (sport/dport: {sport}/{dport})")
            if src_ip == gen_ip and dst_ip in disc_ips:
                rounds[round_id]['samples'].append({
                    'time': time,
                    'dst_ip': dst_ip,
                    'loss': loss_value,
                    'disc_id': disc_ips.index(dst_ip) + 1
                })
            elif dst_ip == gen_ip and src_ip in disc_ips:
                rounds[round_id]['feedbacks'].append({
                    'time': time,
                    'src_ip': src_ip,
                    'loss': loss_value,
                    'disc_id': disc_ips.index(src_ip) + 1
                })
        except ValueError:
            packet_log.append(f"Generator: Invalid payload '{payload}' at {time:.3f}s (sport/dport: {sport}/{dport}) (raw: {raw_payload!r})")

    # Process discriminator PCAPs
    for i, disc_ip in enumerate(disc_ips, 1):
        for pkt in pcap_data[f'discriminator_{i}']:
            if IP not in pkt or UDP not in pkt:
                packet_log.append(f"Discriminator {i}: Non-IP/UDP packet at {pkt.time:.3f}s")
                continue
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
            if sport != port and dport != port:
                packet_log.append(f"Discriminator {i}: Wrong port {sport}/{dport} at {pkt.time:.3f}s")
                continue
            time = pkt.time
            round_id = int(time // round_interval)
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            if pkt.haslayer(Raw):
                raw_payload = pkt[Raw].load
                payload = raw_payload.decode('utf-8', errors='ignore').strip('\x00')
            else:
                payload = ''
                packet_log.append(f"Discriminator {i}: No Raw layer at {time:.3f}s (sport/dport: {sport}/{dport})")
                continue
            try:
                loss_value = float(payload)
                packet_log.append(f"Discriminator {i}: Valid packet src={src_ip} dst={dst_ip} loss={loss_value:.4f} at {time:.3f}s (sport/dport: {sport}/{dport})")
                if src_ip == gen_ip and dst_ip == disc_ip:
                    rounds[round_id]['samples'].append({
                        'time': time,
                        'dst_ip': dst_ip,
                        'loss': loss_value,
                        'disc_id': i
                    })
                elif src_ip == disc_ip and dst_ip == gen_ip:
                    rounds[round_id]['feedbacks'].append({
                        'time': time,
                        'src_ip': src_ip,
                        'loss': loss_value,
                        'disc_id': i
                    })
            except ValueError:
                packet_log.append(f"Discriminator {i}: Invalid payload '{payload}' at {time:.3f}s (sport/dport: {sport}/{dport}) (raw: {raw_payload!r})")

    # Save packet log
    with open("packet_log.txt", "w") as f:
        f.write("\n".join(packet_log))
    print(f"Packet log saved to packet_log.txt")

    return rounds

def calculate_round_latency(rounds):
    """Calculate E2E network latency per round."""
    round_latencies = {}
    max_feedback_delay = 40.0  # Allow feedback up to 2 rounds later
    for round_id, data in rounds.items():
        latencies = []
        for sample in data['samples']:
            sample_time = float(sample['time'])  # Convert to float
            disc_id = sample['disc_id']
            # Look for feedback in same or next round
            matching_feedbacks = []
            for r_id in range(round_id, round_id + 2):  # Check current and next round
                if r_id in rounds:
                    matching_feedbacks.extend([
                        fb for fb in rounds[r_id]['feedbacks']
                        if fb['disc_id'] == disc_id and float(fb['time']) > sample_time and float(fb['time']) <= sample_time + max_feedback_delay
                    ])
            if matching_feedbacks:
                feedback = min(matching_feedbacks, key=lambda x: float(x['time']))
                latency = float(feedback['time']) - sample_time
                latencies.append(latency)
                print(f"Round {round_id}: Matched sample (disc {disc_id}, time {sample_time:.3f}) to feedback (time {feedback['time']:.3f}), latency {latency:.6f}s")
        round_latencies[round_id] = {
            'avg_latency': float(np.mean(latencies)) if latencies else 0.0,
            'num_pairs': len(latencies),
            'latencies': latencies
        }
    return round_latencies

def main():
    # PCAP files
    gen_pcap = "dsrc-highway-gan-gen1-0-0.pcap"
    disc_pcap_list = [
        "dsrc-highway-gan-disc-1-4-0.pcap",
        "dsrc-highway-gan-disc-2-5-0.pcap",
        "dsrc-highway-gan-disc-3-6-0.pcap",
        "dsrc-highway-gan-disc-4-7-0.pcap",
        "dsrc-highway-gan-disc-5-8-0.pcap"
    ]

    print("Loading PCAP files...")
    pcap_data = load_pcap_files(gen_pcap, disc_pcap_list)

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
            print(f"Round {round_id} (~{round_id*20}s to {(round_id+1)*20}s):")
            print(f"  Average Latency: {latency_info['avg_latency']:.6f}s")
            print(f"  Number of Sample-Feedback Pairs: {latency_info['num_pairs']}")
            print(f"  Individual Latencies: {latency_info['latencies']}")

if __name__ == "__main__":
    main()