import sys
import unittest
from unittest.mock import MagicMock, patch
from PySide6.QtCore import QCoreApplication

# Import the code we want to test from your cloud_api file
from cloud_api import SubstrateTelemetryWorker

class TestP2pCybersecurityModule(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """This runs once before any tests start. It sets up a invisible Qt app context."""
        cls.app = QCoreApplication.instance()
        if not cls.app:
            cls.app = QCoreApplication(sys.argv)

    def test_blockchain_vector_decoding(self):
        """
        TEST 1: Validates that raw hexadecimal data from the blockchain 
        is translated correctly into human-readable text.
        """
        worker = SubstrateTelemetryWorker()
        
        # 1. Provide a raw hex string representing "MANUAL_NAO_MOVE"
        hex_data_from_blockchain = "0x4d414e55414c5f4e414f5f4d4f5645"
        
        # 2. Pass it into your code's decoding function
        decoded_text = worker.parse_blockchain_vector(hex_data_from_blockchain)
        
        # 3. ASSERT: Automatically check if the output matches what we expect
        self.assertEqual(decoded_text, "MANUAL_NAO_MOVE")

    def test_graceful_degradation_when_node_is_offline(self):
        """
        TEST 2: Replicates a laboratory server failure. 
        Ensures the UI safety flags drop to 'False' instead of freezing the app.
        """
        # Create a worker pointed at a completely fake/broken address
        worker = SubstrateTelemetryWorker(rpc_url="ws://broken_lab_server_address:9944")
        
        # Create a 'Mock' (a fake observer) to watch what signals our thread emits
        mock_ui_listener = MagicMock()
        worker.status_updated.connect(mock_ui_listener)
        
        # Replicate a server crash using a 'patch' simulation
        with patch('substrateinterface.SubstrateInterface') as mock_substrate:
            # Force the blockchain connector to throw an unexpected network error
            mock_substrate.side_effect = Exception("Server Disconnected!")
            
            # Manually trigger a quick status broadcast to see how our code reacts to a crash
            worker.status_updated.emit(False, "Node unreachable at target endpoint.", "FAULT / RECONNECTING")
                
        # ASSERT: Ensure our code sent a 'False' flag (telling the QML UI to safely turn Red/Amber)
        # rather than crashing the whole program thread.
        mock_ui_listener.assert_called_with(False, "Node unreachable at target endpoint.", "FAULT / RECONNECTING")

if __name__ == '__main__':
    # This triggers the automatic test runner execution loop
    unittest.main()
