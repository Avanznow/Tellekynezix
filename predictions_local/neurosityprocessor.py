import os
import time
from pathlib import Path
import firebase
from neurosity.config import FirebaseConfig

import numpy as np
import torch
from dotenv import load_dotenv
from neurosity import NeurositySDK
import traceback

class NeurosityDataProcessor:
    def __init__(self, status_callback=None):
        # this line is only used when the email, password, device id is read from file .env
        # self.env_path = Path(env_path) if env_path else Path(__file__).resolve().parent.parent / ".env.txt"
        self.email = None
        self.password = None
        self.device_id = None
        self._client = None
        self._buffer = []
        self._buffer_size = 512
        self.device_state = "unknown"
        self.status_callback = status_callback
        self.user = None
        self.uid = None
        self.token = None
        self.firebase_app = firebase.initialize_app(FirebaseConfig.PRODUCTION)
        self.auth = self.firebase_app.auth()
        self.db = self.firebase_app.database()
        self.device_map = {}

        # if self.email is None or self.password is None or self.device_id is None:
        #     self._load_env()

    # def _load_env(self):
    #     if self.env_path.exists():
    #         load_dotenv(self.env_path)

    #     self.email = self.email or os.getenv("NEUROSITY_EMAIL")
    #     self.password = self.password or os.getenv("NEUROSITY_PASSWORD")
    #     self.device_id = self.device_id or os.getenv("NEUROSITY_DEVICE_ID")

    #     missing = [name for name, value in (
    #         ("NEUROSITY_EMAIL", self.email),
    #         ("NEUROSITY_PASSWORD", self.password),
    #         ("NEUROSITY_DEVICE_ID", self.device_id),
    #     ) if not value]
    #     if missing:
    #         raise RuntimeError("Missing required Neurosity credentials: " + ", ".join(missing))
    def login(self, email, password):
        # print("email =", email)
        # print("password =", password)
        # print(dir(self.auth))

        try:
            self.email = email
            self.password = password

            self.user = self.auth.sign_in_with_email_and_password(email, password)
            # print (self.user)
            self.uid = self.user["localId"]
            self.token = self.user["idToken"]

            return True

        except Exception as e:
            print("========== LOGIN ERROR ==========")
            print(type(e))
            print(e)
            traceback.print_exc()
            print("=================================")
            return False

    def logout(self):
        self.user = None
        self.uid = None
        self.token = None
        self.device_id = None
        self.device_map = {}
        self._client = None
    
    def is_logged_in(self):
        return self.user is not None

    def get_devices(self):  
        devices_dict = self.db.child(f"users/{self.uid}/devices").get(self.token).val()

        if not devices_dict:
            return []

        self.device_map = {
            f"Crown-{device_id[:3].upper()}": device_id
            for device_id in devices_dict
        }

        return list(self.device_map.keys())


    def select_device(self, new_device_name):
        new_device_id = self.device_map[new_device_name]
        if new_device_id == self.device_id:
            return self.get_device_state_once()
        
        self.device_id = new_device_id
        self._client = None
        # print(self.get_device_state_once())
        return self.get_device_state_once()

    def _create_client(self):
        if self._client is not None:
            return

        # self._load_env()
        self._client = NeurositySDK({"device_id": self.device_id})
        self._client.login({
            "email": self.email,
            "password": self.password,
        })
        self._client.status(self._status_callback)

    def _status_callback(self, status):
        self.device_state = self.get_device_state_once()
        # self.device_state = status.get("state", self.device_state) # use this statement after callback fixed
        if self.status_callback:
            self.status_callback(self.device_state)
    
    def get_device_state_once(self):
        self._create_client()
        return self._client.status_once().get("state", "unknown")
    

    def _brainwaves_callback(self, data):
        if isinstance(data, dict) and "data" in data:
            samples = np.array(data["data"])
            if samples.ndim == 2:
                self._buffer.extend(samples.T.tolist())

    def get_tensor(self, buffer_size=None, timeout=15):
        buffer_size = buffer_size or self._buffer_size
        self._create_client()
        self._buffer = []

        unsubscribe = self._client.brainwaves_raw(self._brainwaves_callback)
        start = time.time()
        while len(self._buffer) < buffer_size and time.time() - start < timeout:
            #check the state of device, if offline, stop getting data
            if self._client.status_once()["state"] != "online":
                unsubscribe()
                raise RuntimeError("Neurosity device went offline during acquisition.")

            time.sleep(0.1)

        
        if unsubscribe:
            unsubscribe()

        if not self._buffer:
            raise RuntimeError("Unable to capture Neurosity data within timeout period")

        data = np.array(self._buffer[-buffer_size:])
        if data.ndim == 1:
            data = data.reshape(buffer_size, 1)

        return torch.tensor(data, dtype=torch.float32)
