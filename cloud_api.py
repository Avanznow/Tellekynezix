import time
import configparser
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Slot, QThread, Signal

# BLOCKCHAIN ADDITION: Import the Substrate connection engine
try:
    from substrateinterface import SubstrateInterface, Keypair
    from substrateinterface.exceptions import SubstrateRequestException
    HAS_SUBSTRATE = True
except ImportError:
    HAS_SUBSTRATE = False

# Fallback stub for fileTransfer if imported externally in your codebase
try:
    from sftp import fileTransfer 
except ImportError:
    try:
        from your_transfer_module import fileTransfer
    except ImportError:
        fileTransfer = None


class SubstrateTelemetryWorker(QThread):
    """
    Background worker execution thread tracking custom pallet block state mutations.
    Communicates across thread boundaries via standard Qt Signals to prevent GUI freezing.
    """
    status_updated = Signal(bool, str, str) # Parameters: (is_secure, status_message, block_hash)

    def __init__(self, rpc_url="ws://127.0.0.1:9944"):
        super().__init__()
        self.rpc_url = rpc_url
        self._running = True
        self.test_device_id = "drone_01"
        self.encoded_device_key = list(self.test_device_id.encode('utf-8'))

    def update_endpoint(self, new_url):
        """Thread-safe runtime mutation target allowing on-the-fly network endpoint switches"""
        print(f"[P2P ENGINE]: Re-targeting blockchain ledger server node interface to: {new_url}")
        self.rpc_url = new_url

    def parse_blockchain_vector(self, raw_val):
        """Standardized decoding utility ensuring clean structural parsing of vector types from custom storage map"""
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

    def run(self):
        print("[P2P ENGINE]: Waiting for QML layout tree stabilization buffer...")
        # SAFETY HOOK: Yield execution loop briefly to allow Vulkan drivers to fully initialize the QML Canvas
        time.sleep(1.5)
        
        print("[P2P ENGINE]: Spawning asynchronous active validation status tracking routine...")
        while self._running:
            if not HAS_SUBSTRATE:
                self.status_updated.emit(False, "Substrate python libraries uninstalled.", "MISSING DEPENDENCIES")
                time.sleep(5.0)
                continue
                
            try:
                # Initialize external socket pipeline tracking targeting lab setup configs
                substrate = SubstrateInterface(url=self.rpc_url)
                substrate.init_runtime()
                
                # Signal successful localized or server network peer synchronization state
                self.status_updated.emit(True, "Handshake verified with active validator block node.\nStandby state idle.", "Awaiting Block Updates...")
                
                while self._running:
                    try:
                        # Extract the head state block signature hash parameter dynamically
                        block_hash = substrate.get_block_hash()
                        
                        # Query your custom FRAME pallet StorageMap tracking runtime handshakes
                        result = substrate.query(
                            module='Template',
                            storage_function='DeviceCommands',
                            params=[self.encoded_device_key],
                            block_hash=block_hash
                        )
                        
                        if result and result.value:
                            current_command = self.parse_blockchain_vector(result.value)
                            
                            # 💡 NEW INTELLIGENT LOG ROUTER: Converts raw blockchain updates directly into telemetry_inspector log text formats
                            if current_command:
                                if current_command.startswith("MANUAL_NAO_"):
                                    status_msg = f"◆ [OTA ROBOTIC AUDIT]: NAO Unit Event Finalized!\n   • Identity  : {self.test_device_id} (NAO Component)\n   • Payload   : '{current_command}'"
                                else:
                                    status_msg = f"◆ [OTA FLIGHT AUDIT]: Drone Unit Event Finalized!\n   • Identity  : {self.test_device_id}\n   • Payload   : '{current_command}'"
                            else:
                                status_msg = "Handshake connection active. No data packets in block header."
                        else:
                            status_msg = "Handshake connection active. No data packets in block header."
                            
                        self.status_updated.emit(True, status_msg, str(block_hash))
                            
                    except SubstrateRequestException as latency_exception:
                        # Graceful Degradation Handling: Intercept latency bounds and switch visual state to orange warning flags
                        self.status_updated.emit(False, f"Latency or synchronization delay detected: {latency_exception}", "Sync Warning...")
                    
                    time.sleep(1.0)
                    
            except Exception as network_fault:
                # Catch sudden physical node server disconnect crash occurrences or network drops
                fail_msg = f"Node unreachable at target endpoint address: {self.rpc_url}. Re-trying pipeline binding..."
                self.status_updated.emit(False, fail_msg, "FAULT / RECONNECTING")
                
                # Fallback non-blocking sleep counter protecting interface clicks from locking up
                for _ in range(5):
                    if not self._running:
                        break
                    time.sleep(1.0)

    def stop(self):
        self._running = False
        self.wait()


class CloudAPI(QObject):
    _instance = None # Class-level variable to enforce singleton protection

    def __new__(cls, *args, **kwargs):
        """Enforces a strict global singleton pattern to block duplicate thread allocation races"""
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
        self.telemetry_worker = None
        self._initialized = True
        
        if HAS_SUBSTRATE:
            self.init_blockchain_connection()

    def init_blockchain_connection(self):
        """Intuitively connects your GUI to your running Rust engine backend"""
        print("[BLOCKCHAIN LOG]: Connecting to local Substrate node...")
        try:
            # Initialize core functional baseline transmission interface parameters
            self.substrate = SubstrateInterface(url="ws://127.0.0.1:9944")
            self.substrate.init_runtime()
            self.blockchain_keypair = Keypair.create_from_uri('//Alice')
            print(f"[BLOCKCHAIN LOG]: Connected successfully! Chain: {self.substrate.chain} | Runtime: {self.substrate.runtime_version}")
        except Exception as e:
            self.substrate = None
            print(f"[BLOCKCHAIN MOCK LOG]: Could not reach node, but simulation is active. Details: {e}")

    def set_root_object(self, root_object):
        self.root_object = root_object

    def connect_buttons(self):
        if self.root_object is None:
            print("Error: root_object not set in CloudAPI")
            return
        try:
            # Secure children mappings using safe conditionals to prevent null-pointer segfaults
            btn_save = self.root_object.findChild(QObject, "saveConfigButton")
            if btn_save: btn_save.clicked.connect(self.save_config)
            
            btn_load = self.root_object.findChild(QObject, "loadConfigButton")
            if btn_load: btn_load.clicked.connect(self.load_config)
            
            btn_clear = self.root_object.findChild(QObject, "clearConfigButton")
            if btn_clear: btn_clear.clicked.connect(self.clear_config)
            
            btn_upload = self.root_object.findChild(QObject, "uploadButton")
            if btn_upload: btn_upload.clicked.connect(self.upload)
            
            btn_pk = self.root_object.findChild(QObject, "privateKeyDirButton")
            if btn_pk: btn_pk.clicked.connect(self.browse_private_key_dir)
            
            btn_src = self.root_object.findChild(QObject, "sourceDirButton")
            if btn_src: btn_src.clicked.connect(self.browse_source_dir)
            
            btn_tgt = self.root_object.findChild(QObject, "targetDirButton")
            if btn_tgt: btn_tgt.clicked.connect(self.browse_target_dir)
            
            # Wire up the new blockchain input routing fields
            cloud_tab = self.root_object.findChild(QObject, "cloudTabRoot")
            target_view = cloud_tab if cloud_tab else self.root_object
            
            if target_view:
                try:
                    target_view.updateSubstrateEndpoint.connect(self.handle_endpoint_switch)
                    print("Cloud API and Substrate endpoint network listeners bound successfully")
                except AttributeError:
                    print("(BRIDGE WARNING): Base layout bindings processing independently.")
            
            # SAFE POSITIONING: Only launch the background thread worker AFTER all UI buttons are completely bound
            if not self.telemetry_worker:
                self.telemetry_worker = SubstrateTelemetryWorker()
                self.telemetry_worker.status_updated.connect(self.handle_blockchain_status)
                self.telemetry_worker.start()
            print("Cloud API buttons connected successfully")
        except Exception as e:
            print(f"Error connecting cloud API buttons: {e}")

    @Slot(bool, str, str)
    def handle_blockchain_status(self, is_secure, message, block_hash):
        """Crosses thread boundaries safely to map incoming node data back onto QML visual elements"""
        if not self.root_object:
            return
        cloud_tab = self.root_object.findChild(QObject, "cloudTabRoot")
        target_view = cloud_tab if cloud_tab else self.root_object
        if target_view:
            # Thread-safe updates directly matching declarative QML parameters
            target_view.setProperty("isP2pSecure", is_secure)
            target_view.setProperty("p2pStatusMessage", message)
            target_view.setProperty("latestBlockHash", block_hash)

    @Slot(str)
    def handle_endpoint_switch(self, new_url):
        """Intercepts input entries made from laboratory workstation screens and adjusts system configs"""
        if self.telemetry_worker:
            self.telemetry_worker.update_endpoint(new_url)
        if HAS_SUBSTRATE:
            try:
                self.substrate = SubstrateInterface(url=new_url)
                print(f"(TX TUNNEL): Realigned base transmitter module to: {new_url}")
            except Exception:
                self.substrate = None

    @Slot()
    def browse_private_key_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                input_field = self.root_object.findChild(QObject, "privateKeyDirInput")
                if input_field: input_field.setProperty("text", str(file_paths[0]))

    @Slot()
    def browse_source_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                input_field = self.root_object.findChild(QObject, "sourceDirInput")
                if input_field: input_field.setProperty("text", str(file_paths[0]))

    @Slot()
    def browse_target_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                input_field = self.root_object.findChild(QObject, "targetDirInput")
                if input_field: input_field.setProperty("text", str(file_paths[0]))

    @Slot()
    def save_config(self):
        selected_file, _ = QFileDialog.getSaveFileName(None, "Save config file", "", "INI Files (*.ini)")
        if selected_file:
            if not selected_file.endswith(".ini"): selected_file += ".ini"
            with open(selected_file, 'w') as configfile:
                h_in = self.root_object.findChild(QObject, "hostInput")
                u_in = self.root_object.findChild(QObject, "usernameInput")
                pk_in = self.root_object.findChild(QObject, "privateKeyDirInput")
                chk_in = self.root_object.findChild(QObject, "ignoreHostKeyCheckbox")
                src_in = self.root_object.findChild(QObject, "sourceDirInput")
                tgt_in = self.root_object.findChild(QObject, "targetDirInput")
                self.config['data'] = {
                    "-HOST-": h_in.property("text") if h_in else "",
                    "-USERNAME-": u_in.property("text") if u_in else "",
                    "-PRIVATE_KEY-": pk_in.property("text") if pk_in else "",
                    "-IGNORE_HOST_KEY-": chk_in.property("checked") if chk_in else True,
                    "-SOURCE-": src_in.property("text") if src_in else "",
                    "-TARGET-": tgt_in.property("text") if tgt_in else "",
                }
                self.config.write(configfile)

    @Slot()
    def load_config(self):
        selected_file, _ = QFileDialog.getOpenFileName(None, "Load config file", "", "INI Files (*.ini)")
        try:
            if selected_file:
                self.config.read(selected_file)
                h_in = self.root_object.findChild(QObject, "hostInput")
                if h_in: h_in.setProperty("text", self.config["data"]["-HOST-"])
                u_in = self.root_object.findChild(QObject, "usernameInput")
                if u_in: u_in.setProperty("text", self.config["data"]["-USERNAME-"])
                pk_in = self.root_object.findChild(QObject, "privateKeyDirInput")
                if pk_in: pk_in.setProperty("text", self.config["data"]["-PRIVATE_KEY-"])
                chk_in = self.root_object.findChild(QObject, "ignoreHostKeyCheckbox")
                if chk_in: chk_in.setProperty("checked", self.config["data"]["-IGNORE_HOST_KEY-"].lower() in ("true"))
                src_in = self.root_object.findChild(QObject, "sourceDirInput")
                if src_in: src_in.setProperty("text", self.config["data"]["-SOURCE-"])
                tgt_in = self.root_object.findChild(QObject, "targetDirInput")
                if tgt_in: tgt_in.setProperty("text", self.config["data"]["-TARGET-"])
        except Exception as e:
            QMessageBox.critical(None, "Loading failed", "Error: " + str(e))

    @Slot()
    def clear_config(self):
        h_in = self.root_object.findChild(QObject, "hostInput")
        if h_in: h_in.setProperty("text", "")
        u_in = self.root_object.findChild(QObject, "usernameInput")
        if u_in: u_in.setProperty("text", "")
        pk_in = self.root_object.findChild(QObject, "privateKeyDirInput")
        if pk_in: pk_in.setProperty("text", "")
        chk_in = self.root_object.findChild(QObject, "ignoreHostKeyCheckbox")
        if chk_in: chk_in.setProperty("checked", True)
        src_in = self.root_object.findChild(QObject, "sourceDirInput")
        if src_in: src_in.setProperty("text", "")
        tgt_in = self.root_object.findChild(QObject, "targetDirInput")
        if tgt_in: tgt_in.setProperty("text", "/home/")

    def send_telemetry_transaction(self, device_id, command):
        if not HAS_SUBSTRATE or not self.substrate:
            print(f" (MOCK PRINT LOG): Chain offline or uninitialized, but telemetry intercepted -> Device: {device_id}, Action: {command}")
            return False
        try:
            print(f"(BLOCKCHAIN LOG): Formulating block update extrinsic for '{device_id}' -> '{command}'...")
            nonce = self.substrate.get_account_nonce(self.blockchain_keypair.ss58_address)
            call = self.substrate.compose_call(
                call_module='Template',
                call_function='transmit_command',
                call_params={
                    'device_id': str(device_id),
                    'command': str(command)
                }
            )
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.blockchain_keypair,
                nonce=nonce
            )
            receipt = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
            print(f"(BLOCKCHAIN LOG): Finalized securely on-chain! Block Hash: {receipt.block_hash}")
            return True
        except Exception as e:
            print(f"(BLOCKCHAIN ERROR): Broadcast rejected by node: {e}")
            return False

    @Slot()
    def upload(self):
        if not fileTransfer:
            QMessageBox.critical(None, "Configuration Error", "SFTP File Transfer module missing from system pathway.")
            return
        try:
            h_in = self.root_object.findChild(QObject, "hostInput")
            u_in = self.root_object.findChild(QObject, "usernameInput")
            pk_in = self.root_object.findChild(QObject, "privateKeyDirInput")
            pw_in = self.root_object.findChild(QObject, "passwordInput")
            chk_in = self.root_object.findChild(QObject, "ignoreHostKeyCheckbox")
            svrcon = fileTransfer(
                h_in.property("text") if h_in else "",
                u_in.property("text") if u_in else "",
                pk_in.property("text") if pk_in else "",
                pw_in.property("text") if pw_in else "",
                chk_in.property("checked") if chk_in else True
            )
            src_in = self.root_object.findChild(QObject, "sourceDirInput")
            tgt_in = self.root_object.findChild(QObject, "targetDirInput")
            source_dir = src_in.property("text") if src_in else ""
            target_dir = tgt_in.property("text") if tgt_in else ""
            if source_dir and target_dir:
                svrcon.transfer(source_dir, target_dir)
                print("(BLOCKCHAIN LOG): Registration triggered by GUI upload button click event.")
                self.send_telemetry_transaction(device_id="drone_01", command=f"Transfer_Complete_To_{target_dir}")
                QMessageBox.information(None, "Upload complete", "File uploaded successfully via SFTP!")
            else:
                QMessageBox.critical(None, "Upload failed", "Please ensure that all fields have been filled!")
        except Exception as e:
            print("\n(HOME WORKFLOW DETECTED): University Lab Server is unreachable from home.")
            print(f"Network error detailed: {e}")
            try:
                tgt_in = self.root_object.findChild(QObject, "targetDirInput")
                target_dir = tgt_in.property("text") if tgt_in else "/home/mock_target"
            except Exception:
                target_dir = "/home/mock_target"
            print("(FALLBACK): Rerouting action directly to your local Substrate node...")
            success = self.send_telemetry_transaction(device_id="drone_01", command=f"Home_Simulation_{target_dir}")
            if success:
                QMessageBox.information(None, "Home Simulation Success", "SFTP failed, but command registered on Substrate!")
            else:
                QMessageBox.warning(None, "Pipeline Error", "Both SFTP and Substrate transactions failed.")

    def shutdown(self):
        """Clean destructor termination block safely freeing thread sockets during exit events"""
        if self.telemetry_worker:
            self.telemetry_worker.stop()


# Instantiate single global API bridge container object instance for pipeline integrations
cloud_api = CloudAPI()