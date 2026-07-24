import time
from substrateinterface import SubstrateInterface

def init_node_connection(url="ws://127.0.0.1:9944"):
    print(f"[AUDITOR LOG]: Establishing handshake with ledger at {url}...")
    try:
        substrate = SubstrateInterface(url=url)
        print("[AUDITOR LOG]: Connection active. Monitoring block state updates...")
        return substrate
    except Exception as e:
        print(f"[AUDITOR LOG]: Cannot reach node. Re-trying in 5s... Details: {e}")
        return None

def parse_blockchain_vector(raw_val):
    """Cleanly converts raw vector bytes/hex strings back into human-readable text strings"""
    if not raw_val:
        return None

    # Handle hex strings returned by Substrate storage queries
    if isinstance(raw_val, str) and raw_val.startswith('0x'):
        try:
            return bytes.fromhex(raw_val[2:]).decode('utf-8')
        except ValueError:
            return raw_val
    
    # Handle explicit byte arrays or lists of integers
    if isinstance(raw_val, (bytes, bytearray)):
        return raw_val.decode('utf-8')
    elif isinstance(raw_val, list):
        try:
            return bytes(raw_val).decode('utf-8')
        except ValueError:
            return str(raw_val)
    
    return str(raw_val)

def main():
    substrate = None
    last_command = None

    # Target identity key string for our hardware network
    test_device_id = "drone_01"
    # Encode key to a byte list to perfectly match the pallet's Vec<u8> storage mapping
    encoded_device_key = list(test_device_id.encode('utf-8'))

    # Target endpoint variable (change to Server IP when deploying to production)
    node_url = "ws://127.0.0.1:9944"

    while True:
        if not substrate:
            substrate = init_node_connection(url=node_url)
            if not substrate:
                time.sleep(5.0)
                continue

        try:
            # Query our custom DeviceCommands storage map inside the template module
            result = substrate.query(
                module='Template',
                storage_function='DeviceCommands',
                params=[encoded_device_key]
            )

            if result:
                current_command = parse_blockchain_vector(result.value)

                # Filter condition: Only print when a brand new telemetry string arrives
                if current_command and current_command != last_command:
                    
                    # 🛠️ PHASE 4: DETECT AND ROUTE NAO TELEMETRY PREFIXES
                    if current_command.startswith("MANUAL_NAO_"):
                        print(f"\n[OTA ROBOTIC AUDIT]: NAO Unit Event Finalized!")
                        print(f"   • Identity  : {test_device_id} (NAO Component)")
                        print(f"   • Payload   : '{current_command}'")
                    
                    # ROUTE STANDARD DRONE TELEMETRY PREFIXES
                    else:
                        print(f"\n[OTA FLIGHT AUDIT]: Drone Unit Event Finalized!")
                        print(f"   • Identity  : {test_device_id}")
                        print(f"   • Payload   : '{current_command}'")
                        
                    last_command = current_command
        
        except Exception as e:
            print(f"[READ EXCEPTION]: Pipeline state dropped: {e}")
            substrate = None # Forces network auto-reconnection loop if server restarts

        time.sleep(1.0)

if __name__ == "__main__":
    main()
