from scapy.all import rdpcap, UDP, IP
from collections import defaultdict

pcap_file = "/home/ubuntu/ns-allinone-3.43/ns-3.43/dsrc-highway-gan-gen2-1-0.pcap"  # Change this to your actual file
packets = rdpcap(pcap_file)

send_times = defaultdict(list)
recv_times = defaultdict(list)

# Build timestamp dictionaries based on IP/UDP ports
for pkt in packets:
    if IP in pkt and UDP in pkt:
        ip = pkt[IP]
        udp = pkt[UDP]

        # Assume client sends first (request), server replies (response)
        key = (ip.src, ip.dst, udp.sport, udp.dport)
        reverse_key = (ip.dst, ip.src, udp.dport, udp.sport)

        send_times[key].append(pkt.time)
        recv_times[reverse_key].append(pkt.time)

# Match send and receive times based on same flow (client?server then server?client)
latencies = []

for key in send_times:
    sends = send_times[key]
    responses = recv_times.get(key, [])

    pair_count = min(len(sends), len(responses))
    for i in range(pair_count):
        rtt = responses[i] - sends[i]
        if rtt > 0:
            latencies.append(rtt)

# Display results
if latencies:
    avg_latency = sum(latencies) / len(latencies)
    print(f"Matched pairs: {len(latencies)}")
    print(f"Average RTT (Round Trip Time): {avg_latency*1000:.2f} ms")
    print(f"Min RTT: {min(latencies)*1000:.2f} ms")
    print(f"Max RTT: {max(latencies)*1000:.2f} ms")
else:
    print("? No valid latency pairs found. Try a different pcap or ensure Echo traffic exists.")
