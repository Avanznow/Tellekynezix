// replace the contents of `pallets/template/src/lib.rs` with this custom pallet logic for implementing Solochain in Avatar-Telekynezix P2P network

// We make sure this pallet uses `no_std` for compiling to Wasm.
#![cfg_attr(not(feature = "std"), no_std)]

// Re-export pallet items so that they can be accessed from the crate namespace.
pub use pallet::*;

// Bring in Vector support safely for no_std environments
extern crate alloc;

// RUNTIME ALIGNMENT: Create the SubstrateWeight structure expected by runtime/src/configs/mod.rs
pub mod weights {
	use frame_support::weights::Weight;

	pub trait WeightInfo {
		fn do_something() -> Weight { Weight::default() }
		fn cause_error() -> Weight { Weight::default() }
	}
	impl WeightInfo for () {}

	pub struct SubstrateWeight<T>(core::marker::PhantomData<T>);
	impl<T: frame_system::Config> WeightInfo for SubstrateWeight<T> {}
	impl<T> SubstrateWeight<T> {
		pub fn transmit_command() -> Weight { Weight::default() }
		pub fn register_brainwave() -> Weight { Weight::default() }
		pub fn submit_prediction() -> Weight { Weight::default() }
	}
}

// All pallet logic is defined in its own module and must be annotated by the `pallet` attribute.
#[frame_support::pallet]
pub mod pallet {
	// Import various useful types required by all FRAME pallets.
	use frame_support::pallet_prelude::*;
	use frame_system::pallet_prelude::*;
	use alloc::vec::Vec;

	#[pallet::pallet]
	pub struct Pallet<T>(_);

	/// The pallet's configuration trait.
	#[pallet::config]
	pub trait Config: frame_system::Config {
		/// Uses strict Polkadot SDK event trait bindings
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
		
		/// RUNTIME ALIGNMENT: Add the missing WeightInfo association expected by configs/mod.rs
		type WeightInfo: super::weights::WeightInfo;
	}

	/// BLOCKCHAIN ADDITION: Custom storage mapping for connected devices
	#[pallet::storage]
	#[pallet::unbounded]
	pub(super) type DeviceCommands<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		Vec<u8>, // Key: device_id
		Vec<u8>, // Value: command string
		OptionQuery,
	>;

	/// READ BRAIN STORAGE (input): maps a device/laptop ID to its raw EEG file hash fingerprint
	#[pallet::storage]
	#[pallet::unbounded]
	pub(super) type BrainwaveRegistry<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		Vec<u8>, // Key: device_id
		Vec<u8>, // Value: SHA-128 EEG file hash fingerprint
		OptionQuery,
	>;
	
	/// READ BRAIN STORAGE (output): maps a device/laptop ID to its calculated ML flight command prediction
	#[pallet::storage]
	#[pallet::unbounded]
	pub(super) type BrainwavePredictions<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		Vec<u8>, // Key: device_id
		Vec<u8>, // Value: Predicted flight command string
		OptionQuery,
	>;

	/// Events that functions in this pallet can emit.
	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Over-the-air notification dispatched whenever a new transaction is processed
		CommandIssued { device_id: Vec<u8>, command: Vec<u8> },
		BrainwaveRegistered { device_id: Vec<u8>, file_hash: Vec<u8> },
		BrainwavePredicted { device_id: Vec<u8>, prediction: Vec<u8> },
	}

	#[pallet::error]
	pub enum Error<T> {
		NoneValue,
		StorageOverflow,
	}

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Custom transaction endpoint used by the Python GUI interface 
		#[pallet::call_index(0)]
		#[pallet::weight(Weight::default())]
		pub fn transmit_command(
			origin: OriginFor<T>,
			device_id: Vec<u8>,
			command: Vec<u8>,
		) -> DispatchResult {
			// Ensure signature authentication from the transaction origin (e.g., Alice)
			let _sender = ensure_signed(origin)?;

			// Bind the inputs inside the global ledger map storage dictionary
			<DeviceCommands<T>>::insert(&device_id, &command);

			// Broadcast an event receipt out across the network nodes
			Self::deposit_event(Event::CommandIssued { device_id, command });

			Ok(())
		}
		
		/// EXTRINSIC 1: Triggered by the Python GUI interface to register a new EEG file hash fingerprint for a device
		#[pallet::call_index(1)]
		#[pallet::weight(Weight::default())]
		pub fn register_brainwave(
			origin: OriginFor<T>,
			device_id: Vec<u8>,
			file_hash: Vec<u8>,
		) -> DispatchResult {
			// Ensure signature authentication from the transaction origin (e.g., Alice)
			let _sender = ensure_signed(origin)?;

			// Bind the inputs inside the global ledger map storage dictionary
			<BrainwaveRegistry<T>>::insert(&device_id, &file_hash);

			// Broadcast an event receipt out across the network nodes
			Self::deposit_event(Event::BrainwaveRegistered { device_id, file_hash });

			Ok(())
		}
		
		/// EXTRINSIC 2: Triggered by the server after the ML model has predicted a flight command based on the EEG file hash fingerprint
		#[pallet::call_index(2)]
		#[pallet::weight(Weight::default())]
		pub fn submit_prediction(
			origin: OriginFor<T>,
			device_id: Vec<u8>,
			file_hash: Vec<u8>,
			prediction: Vec<u8>,
		) -> DispatchResult {
			// Ensure signature authentication from the computing node
			let _sender = ensure_signed(origin)?;

			// Bind the prediction to the device ID
			<BrainwavePredictions<T>>::insert(&device_id, &prediction);

			// Broadcast an event receipt out across the network nodes
			Self::deposit_event(Event::BrainwavePredicted { device_id, prediction });

			Ok(())
		}
	}
}
