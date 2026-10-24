# Avatar-Tellekynezix Substrate Solochain

## Architecture Overview

The legacy centralized VPS framework has been augmented by a decentralized **Substrate Solochain Node**:
* **Substrate Runtime:** Contains a custom `template` pallet managing decentralized ledger access controls.
* **Storage Arrays:** Features `DeviceCommands`, a secure on-chain map logging explicit payload matrices mapped directly to hashed device public keys (`encoded_device_key`).
* **Python API Interface:** A specialized GUI loop utilizes `substrate-interface` to securely poll runtime blocks and handle local device instructions.

---

## Prerequisites & Environment Setup

Before starting compilation, ensure your machine has the necessary Rust toolchain and foundational system dependencies installed.

### 1. System Dependencies (Ubuntu/Linux)
```bash
sudo apt update
sudo apt install -y cmake build-essential git libssl-dev pkg-config clang curl
```

### 2. Rust & WebAssembly Setup
Substrate runtime compilation relies on specific stable and nightly WASM toolchain profiles:
```bash
# Install Rustup toolchain manager
curl --proto '=https' --tlsv1.2 -sSf https://rustup.rs | sh
source $HOME/.cargo/env

# Configure standard WASM compilation target
rustup default stable
rustup update
rustup update nightly
rustup target add wasm32-unknown-unknown --toolchain nightly
```
NOTE: Refer to this article from the Polkadot SDK documentation for setup instructions on other Operating Systems:
https://docs.polkadot.com/parachains/install-polkadot-sdk/

---

## Cloning and Initial Setup

Clone the Substrate Node Template:
```bash
# Clone the repository
git clone https://github.com/paritytech/polkadot-sdk-solochain-template.git
cd polkadot-sdk-solochain-template
```

---

## Building and Running the Substrate Node

### 1. Update `lib.rs` with Custom Pallets & Compile
Replace the template pallet file `pallets/template/src/lib.rs` with your custom file located at `P2P/solochain-template/lib.rs`. Then, build the substrate node:
```bash
cargo build --release
```

### 2. Launch Local Development Network
Boot up a clean, single-node local validator dev chain with immediate transaction finality:
```bash
./target/release/node-template --dev
```
The local blockchain is now active, exposing an RPC interface endpoint locally at `ws://127.0.0.1:9944` or `http://127.0.0.1:9944`.

---

## Setting Up the Client Side (Python Client)

Once the local validator node is successfully broadcasting blocks, set up and run the localized device client framework:

### 1. Configure the Virtual Environment
Navigate back to your project root folder and execute:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Application Interface
Launch the primary GUI execution handler. It will instantly initialize an edge handshake configuration to hook directly into the local `SubstrateInterface` runtime tracking loop:
```bash
python GUI5.py
```

### 3. Run the Telemetry Inspector
Open a separate terminal window, activate your virtual environment, and run the inspector to audit live blocks being added to the blockchain ledger:
```bash
cd P2P
python telemetry_inspector.py
```
