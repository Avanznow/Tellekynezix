"""Plain-language explanations of quality rules, shared by the dashboard and the GUI."""


def reason_text(config):
    """Plain-language explanation of every rule reason, using the active thresholds."""
    return {
        "flat_channel": "Flat signal: no change between samples (electrode disconnected, shorted or channel off)",
        "adc_saturation": f"Saturated: at least {100 * config.max_rail_fraction:g}% of samples at the ADC rail "
                          f"(±{config.rail_uv:,.0f} µV)",
        "excessive_amplitude": f"Peak-to-peak above {config.max_ptp_uv:,.0f} µV",
        "excessive_sample_jump": f"Jump above {config.max_jump_uv:,.0f} µV between consecutive samples",
        "excessive_line_noise": f"Mains ({config.mains_hz} Hz) power above {100 * config.max_line_ratio:g}% of the spectrum",
        "nonfinite_eeg": "Missing or non-numeric samples",
        "out_of_range_eeg": "Values outside any physical EEG range",
        "packet_discontinuity": "Missing packets: the hardware sample counter skipped",
        "acquisition_gap": f"Gap longer than {config.max_arrival_gap_seconds:g} s between arrivals",
        "nonmonotonic_time": "Timestamps go backwards",
        "nonadvancing_time": "Timestamps never advance",
        "insufficient_samples": f"Shorter than {config.min_seconds:g} s",
        "mains_band_not_resolvable": "Mains frequency cannot be resolved at this sampling rate",
        "learned_quality_rejection": "Rejected by the trained quality model",
        "awaiting_quality_recovery": "Waiting for enough consecutive good windows after a bad one",
        "continuity_unverified": "Packet counter or timestamps unavailable, so continuity cannot be checked",
        "repeated_or_reversed_capture": "The same capture was replayed",
    }


def explain(reason, config):
    """One reason code (or free-text withholding reason) as a sentence for people."""
    return reason_text(config).get(reason, reason.replace("_", " "))
