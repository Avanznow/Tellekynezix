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

    if isinstance(raw_val, str) and raw_val.startswith('0x'):
        try:
            return bytes.fromhex(raw_val[2:]).decode('utf-8')
        except ValueError:
            return raw_val
    
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
    last_hash = None
    last_prediction = None

    test_device_id = "drone_01"
    encoded_device_key = list(test_device_id.encode('utf-8'))
    node_url = "ws://127.0.0.1:9944"

    while True:
        if not substrate:
            substrate = init_node_connection(url=node_url)
            if not substrate:
                time.sleep(5.0)
                continue

        try:
            # -----------------------------------------------------------------
            # MONITOR CHANNEL 1: Manual Controls & Actions (Your Original Logic)
            # -----------------------------------------------------------------
            result_cmd = substrate.query(
                module='Template',
                storage_function='DeviceCommands',
                params=[encoded_device_key]
            )

            if result_cmd:
                current_command = parse_blockchain_vector(result_cmd.value)
                if current_command and current_command != last_command:
                    if current_command.startswith("MANUAL_NAO_"):
                        print(f"\n[OTA ROBOTIC AUDIT]: NAO Unit Event Finalized!")
                        print(f"   • Identity  : {test_device_id} (NAO Component)")
                        print(f"   • Payload   : '{current_command}'")
                    else:
                        print(f"\n[OTA FLIGHT AUDIT]: Drone Unit Event Finalized!")
                        print(f"   • Identity  : {test_device_id}")
                        print(f"   • Payload   : '{current_command}'")
                    last_command = current_command

            # -----------------------------------------------------------------
            # NEW MONITOR CHANNEL 2: Secure Brainwave File Hashing Anchors
            # -----------------------------------------------------------------
            result_hash = substrate.query(
                module='Template',
                storage_function='BrainwaveRegistry',
                params=[encoded_device_key]
            )

            if result_hash:
                current_hash = parse_blockchain_vector(result_hash.value)
                if current_hash and current_hash != last_hash:
                    print(f"\n[SECURE BLOCK DATA CRYPTO ANCHOR]: Brainwave File Transmitted!")
                    print(f"   • Identity  : {test_device_id}")
                    print(f"   • SHA-256   : {current_hash}")
                    last_hash = current_hash

            # -----------------------------------------------------------------
            # NEW MONITOR CHANNEL 3: AI/ML Machine Learning Predictions
            # -----------------------------------------------------------------
            result_pred = substrate.query(
                module='Template',
                storage_function='BrainwavePredictions',
                params=[encoded_device_key]
            )

            if result_pred:
                current_prediction = parse_blockchain_vector(result_pred.value)
                if current_prediction and current_prediction != last_prediction:
                    print(f"\n[DECENTRALIZED MODEL PREDICTION]: AI Model Processing Completed!")
                    print(f"   • Identity  : {test_device_id}")
                    print(f"   • Prediction: '{current_prediction}'")
                    last_prediction = current_prediction
        
        except Exception as e:
            print(f"[READ EXCEPTION]: Pipeline state dropped: {e}")
            substrate = None

        time.sleep(1.0)

if __name__ == "__main__":
    main()
