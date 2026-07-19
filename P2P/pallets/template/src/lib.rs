// We make sure this pallet uses `no_std` for compiling to Wasm.
#![cfg_attr(not(feature = "std"), no_std)]

// Re-export pallet items so that they can be accessed from the crate namespace.
pub use pallet::*;

// Bring in Vector support safely for no_std environments
extern crate alloc;

// 💡 RUNTIME ALIGNMENT: Create an empty weights module to satisfy the runtime bindings
pub mod weights {
	pub trait WeightInfo {}
	impl WeightInfo for () {}
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
		
		/// 💡 RUNTIME ALIGNMENT: Add the missing WeightInfo association expected by configs/mod.rs
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

	/// Events that functions in this pallet can emit.
	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Over-the-air notification dispatched whenever a new transaction is processed
		CommandIssued { device_id: Vec<u8>, command: Vec<u8> },
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
	}
}
