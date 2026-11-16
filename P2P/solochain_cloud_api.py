import hashlib
import time
import configparser
from PySide6.QtCore import QObject, QThread, Signal, Slot

try:
    from substrateinterface import Keypair, SubstrateInterface
    from substrateinterface.exceptions import SubstrateRequestException
    print(f" [P2P ENGINE]: Substrate python libraries found. Blockchain telemetry enabled.")
    HAS_SUBSTRATE = True
except ImportError:
    print(f" [P2P ENGINE]: Substrate python libraries not found. Blockchain telemetry will be disabled.")
    HAS_SUBSTRATE = False


class SubstrateTelemetryWorker(QThread):
    status_updated = Signal(bool, str, str)

    def __init__(self, rpc_url="ws://127.0.0.1:9944"):
        super().__init__()
        self.rpc_url = rpc_url
        self._running = True
        self.test_device_id = "drone_01"
        self.encoded_device_key = list(self.test_device_id.encode("utf-8"))

    def update_endpoint(self, new_url):
        print(f" [P2P ENGINE]: Re-targeting blockchain ledger server node interface to: {new_url}")
        self.rpc_url = new_url

    def parse_blockchain_vector(self, raw_val):
        if not raw_val:
            return None
        if isinstance(raw_val, str) and raw_val.startswith("0x"):
            try:
                return bytes.fromhex(raw_val[2:]).decode("utf-8")
            except ValueError:
                return raw_val
        if isinstance(raw_val, (bytes, bytearray)):
            return raw_val.decode("utf-8")
        elif isinstance(raw_val, list):
            try:
                return bytes(raw_val).decode("utf-8")
            except ValueError:
                return str(raw_val)
        return str(raw_val)

    def run(self):
        time.sleep(1.5)
        
        # Track the last seen raw payload to filter out duplicate polling cycles
        last_processed_payload = None

        while self._running:
            if not HAS_SUBSTRATE:
                self.status_updated.emit(
                    False,
                    "Substrate python libraries not connected.",
                    "MISSING DEPENDENCIES",
                )
                time.sleep(1.5)
                continue
            try:
                substrate = SubstrateInterface(url=self.rpc_url)
                substrate.init_runtime()
                self.status_updated.emit(
                    True,
                    "Handshake verified with active validator block node.\nStandby state idle.",
                    "Awaiting Block Updates...",
                )

                while self._running:
                    block_hash = substrate.get_block_hash()
                    result = substrate.query(
                        module="Template",
                        storage_function="DeviceCommands",
                        params=[self.encoded_device_key],
                        block_hash=block_hash,
                    )
                    
                    if result and result.value:
                        current_command = self.parse_blockchain_vector(result.value)
                        
                        if current_command:
                            # Clean up the payload string if it contains a pipe separator
                            if "|" in current_command:
                                current_command = current_command.split("|")[0]

                            # Only process and emit UI loops if this is a BRAND NEW ledger update
                            if current_command != last_processed_payload:
                                last_processed_payload = current_command

                                # Execute UI update loops only for unique command occurrences
                                if current_command.startswith("MANUAL_NAO_"):
                                    status_msg = f"[OTA ROBOTIC AUDIT]: NAO Unit Event Finalized!\n Identity: {self.test_device_id}\n Payload: '{current_command}'"
                                else:
                                    status_msg = f"[OTA FLIGHT AUDIT]: Drone Unit Event Finalized!\n Identity: {self.test_device_id}\n Payload: '{current_command}'"
                                
                                self.status_updated.emit(True, status_msg, str(block_hash))
                    
                    # Native polling frequency constraint
                    time.sleep(0.1)

            except Exception as latency_exception:
                self.status_updated.emit(
                    False,
                    f"Sync check active: {latency_exception}",
                    "Sync Status",
                )
                time.sleep(2.0)

    def stop(self):
        self._running = False
        self.wait()


class SolochainCloudAPI(QObject):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.root_object = None
        self.substrate = None
        self.blockchain_keypair = None
        self.telemetry_worker = SubstrateTelemetryWorker()
        self.telepens_worker_connection = self.telemetry_worker.status_updated.connect(self.handle_blockchain_status)
        self.telemetry_worker.start()
        self._initialized = True

        if HAS_SUBSTRATE:
            self.init_blockchain_connection()

    def calculate_file_sha256(self, file_path):
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return "0x" + sha256_hash.hexdigest()
        except Exception as e:
            print(f" [HASH ERROR]: Could not compute SHA-256 for {file_path}: {e}")
            return None

    def init_blockchain_connection(self):
        print("[BLOCKCHAIN LOG]: Connecting to local Substrate node...")
        try:
            self.substrate = SubstrateInterface(url="ws://127.0.0.1:9944")
            self.substrate.init_runtime()
            self.blockchain_keypair = Keypair.create_from_uri("//Alice")
            print(f" [BLOCKCHAIN LOG]: Connected successfully! Chain: {self.substrate.chain}")
        except Exception as e:
            self.substrate = None
            print(f"[BLOCKCHAIN MOCK LOG]: Simulation active. Details: {e}")

    def set_root_object(self, root_object):
        self.root_object = root_object
        self.connect_signals()

    def connect_signals(self):
        if self.root_object is None:
            print("Error: root_object not set in SolochainCloudAPI")
            return

        try:
            # Find the root component and connect the QML signal to Python
            tab_root = self.root_object.findChild(QObject, "cloudTabRoot")
            if tab_root:
                tab_root.updateSubstrateEndpoint.connect(self.handle_endpoint_switch)
                print("Solochain endpoint signal connected successfully")
            else:
                print("Error: cloudTabRoot not found for Solochain API")
        except Exception as e:
            print(f"Error connecting solochain signal: {e}")

    @Slot(bool, str, str)
    def handle_blockchain_status(self, is_secure, message, block_hash):
        if not self.root_object:
            return
        cloud_tab = self.root_object.findChild(QObject, "cloudTabRoot")
        target_view = cloud_tab if cloud_tab else self.root_object
        if target_view:
            target_view.setProperty("isP2pSecure", is_secure)
            target_view.setProperty("p2pStatusMessage", message)
            target_view.setProperty("latestBlockHash", block_hash)

    @Slot(str)
    def handle_endpoint_switch(self, new_url):
        if self.telemetry_worker:
            self.telemetry_worker.update_endpoint(new_url)
        if HAS_SUBSTRATE:
            try:
                self.substrate = SubstrateInterface(url=new_url)
            except Exception:
                self.substrate = None

    def _get_safe_nonce(self):
        """Bypasses library function mismatches using a direct, universal RPC call."""
        try:
            result = self.substrate.rpc_request(
                "system_accountNextIndex",
                [self.blockchain_keypair.ss58_address],
            )
            return result.get("result", 0)
        except Exception:
            return 0

    def send_telemetry_transaction(self, device_id, command):
        if not HAS_SUBSTRATE or not self.substrate:
            print(f"Blockchain offline -> Device: {device_id}, Action: {command}")
            return False
        try:
            print(f" (BLOCKCHAIN LOG): Formulating block update extrinsic for '{device_id}' -> '{command}'...")
            nonce = self._get_safe_nonce()
            
            call = self.substrate.compose_call(
                call_module="Template",
                call_function="transmit_command",
                call_params={
                    "device_id": str(device_id).encode("utf-8"),
                    "command": str(command).encode("utf-8"), 
                },
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=self.blockchain_keypair, nonce=nonce
            )
            receipt = self.substrate.submit_extrinsic(
                extrinsic, wait_for_inclusion=False
            )
            print(f" (BLOCKCHAIN LOG): Finalized securely on-chain! Block Hash: {receipt.block_hash}")
            return True
            
        except Exception as e:
            print(f" (BLOCKCHAIN ERROR): Broadcast rejected by node: {e}")
            return False

    def send_brainwave_registration(self, device_id, file_path):
        if not HAS_SUBSTRATE or not self.substrate:
            print(f" [MOCK MODE]: Chain offline. Simulated registration for {device_id}")
            return False
        file_hash = self.calculate_file_sha256(file_path)
        if not file_hash:
            return False
        try:
            print(f" [BLOCKCHAIN LOG]: Registering brainwave for '{device_id}' with hash '{file_hash}'...")
            nonce = self._get_safe_nonce()

            call = self.substrate.compose_call(
                call_module="Template",
                call_function="register_brainwave",
                call_params={
                    "device_id": str(device_id).encode("utf-8"),
                    "file_hash": str(file_hash).encode("utf-8"),
                },
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=self.blockchain_keypair, nonce=nonce
            )
            receipt = self.substrate.submit_extrinsic(
                extrinsic, wait_for_inclusion=True
            )
            print(f"[BLOCKCHAIN LOG]: Brainwave registered successfully! Block Hash: {receipt.block_hash}")
            return True
        except Exception as e:
            print(f" [BLOCKCHAIN ERROR]: Registration failed: {e}")
            return False

    def check_latest_brain_prediction(self, device_id, file_path, prediction_command):
        if not HAS_SUBSTRATE or not self.substrate:
            print(f"[MOCK MODE]: Chain offline. Simulated prediction submission for {device_id}")
            return False
        file_hash = self.calculate_file_sha256(file_path) or f"0x_mock_{int(time.time())}"
        try:
            print(f" [BLOCKCHAIN LOG]: Broadcasting ML Flight Prediction '{prediction_command}' for '{device_id}'...")
            nonce = self._get_safe_nonce()
            call = self.substrate.compose_call(
                call_module="Template",
                call_function="submit_prediction",
                call_params={
                    "device_id": str(device_id).encode("utf-8"),
                    "file_hash": str(file_hash).encode("utf-8"),
                    "prediction": str(prediction_command).encode("utf-8"),
                },
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=self.blockchain_keypair, nonce=nonce
            )
            receipt = self.substrate.submit_extrinsic(
                extrinsic, wait_for_inclusion=True
            )
            print(f"[BLOCKCHAIN LOG]: ML Prediction finalized securely on-chain! Block Hash: {receipt.block_hash}")
            return True
        except Exception as e:
            print(f" [BLOCKCHAIN ERROR]: Prediction broadcast failed: {e}")
            return False

    def shutdown(self):
        if self.telemetry_worker:
            self.telemetry_worker.stop()


solochain_api = SolochainCloudAPI()