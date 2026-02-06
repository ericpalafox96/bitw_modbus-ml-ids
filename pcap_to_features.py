from scapy.all import rdpcap, TCP, Raw
import pandas as pd
import numpy as np
import hashlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pcap", required=True)
parser.add_argument("--window", type=float, default=0.5)
parser.add_argument("--label", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

PCAP_FILE = args.pcap
WINDOW_SIZE = args.window
LABEL = args.label
OUT_FILE = args.out

packets = rdpcap(PCAP_FILE)

# Filter Modbus TCP packets (port 502)
filtered = []
for pkt in packets:
    if pkt.haslayer(TCP):
        sport = int(pkt[TCP].sport)
        dport = int(pkt[TCP].dport)
        if sport == 502 or dport == 502:
            filtered.append(pkt)

if len(filtered) == 0:
    raise SystemExit("No Modbus packets found")

packets = filtered

pkt_times = np.array([float(pkt.time) for pkt in packets])
pkt_sizes = np.array([len(pkt) for pkt in packets])

# Hash payloads for replay detection
payload_hashes = []
for pkt in packets:
    if pkt.haslayer(Raw):
        payload_hashes.append(hashlib.md5(bytes(pkt[Raw].load)).hexdigest())
    else:
        payload_hashes.append(None)

start_time = pkt_times[0]
end_time = pkt_times[-1]

rows = []
current = start_time

while current < end_time:
    mask = (pkt_times >= current) & (pkt_times < current + WINDOW_SIZE)
    idx = np.where(mask)[0]

    if idx.size == 0:
        current += WINDOW_SIZE
        continue

    sizes = pkt_sizes[idx]
    times = pkt_times[idx]

    # Inter-arrival times
    if times.size > 1:
        inter_arrivals = np.diff(times)
    else:
        inter_arrivals = np.array([0.0])

    # Duplicate payload ratio (replay signal)
    window_hashes = [payload_hashes[i] for i in idx if payload_hashes[i]]
    dup_ratio = 0.0
    if len(window_hashes) > 1:
        dup_ratio = 1.0 - (len(set(window_hashes)) / len(window_hashes))

    # --- NEW: Write-based features (command injection signal) ---
    write_count = 0
    write_regs = set()

    for i in idx:
        pkt = packets[i]
        if pkt.haslayer(Raw):
            payload = bytes(pkt[Raw].load)

            # Modbus/TCP: MBAP header is 7 bytes, function code is byte 7
            if len(payload) >= 8:
                func_code = payload[7]

                # 0x06 = Write Single Register, 0x10 = Write Multiple Registers
                if func_code in (6, 16):
                    write_count += 1

                    # Register address is next two bytes (works for 0x06 and 0x10)
                    if len(payload) >= 10:
                        reg = int.from_bytes(payload[8:10], byteorder="big")
                        write_regs.add(reg)

    total_pkts = int(idx.size)
    write_ratio = write_count / total_pkts if total_pkts > 0 else 0.0
    unique_write_regs = len(write_regs)
    # -----------------------------------------------------------

    row = {
        "packet_count": total_pkts,
        "bytes_total": int(sizes.sum()),
        "packet_size_mean": float(sizes.mean()),
        "packet_size_std": float(sizes.std()),
        "iat_mean": float(inter_arrivals.mean()),
        "iat_std": float(inter_arrivals.std()),
        "dup_payload_ratio": float(dup_ratio),

        # NEW columns:
        "write_ratio": float(write_ratio),
        "unique_write_regs": int(unique_write_regs),

        "label": LABEL
    }

    rows.append(row)
    current += WINDOW_SIZE

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)
print("Saved:", OUT_FILE)
