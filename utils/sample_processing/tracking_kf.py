import numpy as np
from dataclasses import dataclass
from .tracking import (
    TrackingSignalState,
    TrackingSignalParameters,
    TrackingLoopParameters,
    SignalTrackingOutputs,
)
from .correlation import correlate__delay


def track_signal(
    uptime_seconds: float,
    sample_block: np.ndarray,
    signal_params: TrackingSignalParameters,
    signal_state: TrackingSignalState,  # *IO
    loop_params: TrackingLoopParameters,
    signal_outputs: SignalTrackingOutputs,  # *IO
) -> None:
    # Unpack values
    code_seq = signal_params.code_seq
    nominal_code_rate_chips_per_sec = signal_params.nominal_code_rate_chips_per_sec
    code_length_chips = signal_params.code_length_chips
    carrier_freq_hz = signal_params.carrier_freq_hz

    carrier_phase_cycles = signal_state.carrier_phase_cycles
    doppler_freq_hz = signal_state.doppler_freq_hz
    code_phase_chips = signal_state.code_phase_seconds * nominal_code_rate_chips_per_sec
    adjusted_code_rate_chips_per_sec = nominal_code_rate_chips_per_sec * (1 + doppler_freq_hz / carrier_freq_hz)

    # Propagate carrier and code phases to start of block
    time_delta = uptime_seconds - signal_state.uptime_epoch_seconds
    carrier_phase_cycles += doppler_freq_hz * time_delta
    code_phase_chips += adjusted_code_rate_chips_per_sec * time_delta

    # Perform correlation
    # SLOW VERSION:
    # First perform carrier wipeoff
    # phi = 2.0 * np.pi * (carrier_phase_cycles + doppler_freq_hz * sample_block_time_arr)
    # wipeoff_samples = sample_block * np.exp(-1j * phi)
    # # Next perform code correlation (early, prompt, late)
    # code_phase_chips = code_phase_chips
    # prompt_code_phase_chips = code_phase_chips + sample_block_time_arr * adjusted_code_rate_chips_per_sec
    # corr_values = np.zeros(3, dtype=complex)
    # EPL_chip_spacing = loop_params.EPL_chip_spacing
    # for i_bin, chip_bin_offset in enumerate([EPL_chip_spacing, 0.0, -EPL_chip_spacing]):
    #     offset_code_phase_chips = prompt_code_phase_chips + chip_bin_offset
    #     chip_indices = np.floor(offset_code_phase_chips).astype(int) % code_length_chips
    #     signal_replica = code_seq[chip_indices] 
    #     corr_values[i_bin] = np.sum(wipeoff_samples * np.conj(signal_replica))
    # early, prompt, late = corr_values
    
    # FAST VERSION:
    EPL_chip_spacing = loop_params.EPL_chip_spacing
    early, prompt, late = correlate__delay(
        sample_block,
        loop_params.samp_rate,
        carrier_phase_cycles,
        doppler_freq_hz,
        code_seq,
        code_length_chips,
        adjusted_code_rate_chips_per_sec,
        code_phase_chips,
        3,
        EPL_chip_spacing,
        -EPL_chip_spacing,
    )

    # Estimate state errors from correlator outputs
    # Costas (arctan) discriminator for carrier phase error
    delta_theta = np.arctan(prompt.imag / prompt.real) / (2.0 * np.pi)
    delta_eta = EPL_chip_spacing * (np.abs(early) - np.abs(late)) / (np.abs(early) + np.abs(late))

    # Apply loop filter to errors
    # DLL
    filt_code_phase_error_chips = loop_params.DLL_filter_coeff * delta_eta
    # PLL
    filt_carr_phase_error_cycles = loop_params.PLL_filter_coeffs[0] * delta_theta
    filt_doppler_freq_error_hz = loop_params.PLL_filter_coeffs[1] * delta_theta

    # Update signal state
    carrier_phase_cycles += filt_carr_phase_error_cycles
    doppler_freq_hz += filt_doppler_freq_error_hz
    code_phase_chips += filt_code_phase_error_chips

    signal_state.uptime_epoch_seconds = uptime_seconds
    signal_state.carrier_phase_cycles = carrier_phase_cycles
    signal_state.doppler_freq_hz = doppler_freq_hz
    signal_state.code_phase_seconds = code_phase_chips / nominal_code_rate_chips_per_sec

    # Store outputs
    output_idx = signal_outputs.output_index
    signal_outputs.uptime_seconds[output_idx] = uptime_seconds
    signal_outputs.carr_phase_errors_cycles[output_idx] = delta_theta
    signal_outputs.code_phase_errors_chips[output_idx] = delta_eta
    signal_outputs.early_corr[output_idx] = early
    signal_outputs.prompt_corr[output_idx] = prompt
    signal_outputs.late_corr[output_idx] = late
    signal_outputs.carr_phase_cycles[output_idx] = carrier_phase_cycles
    signal_outputs.doppler_freq_hz[output_idx] = doppler_freq_hz
    signal_outputs.code_phase_seconds[output_idx] = signal_state.code_phase_seconds
    signal_outputs.output_index += 1