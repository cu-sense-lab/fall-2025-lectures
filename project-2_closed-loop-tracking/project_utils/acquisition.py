import os
import logging
import numpy as np
from typing import Tuple, Optional
from gnss_tools.signals.gps_l1ca import (
    get_GPS_L1CA_code_sequence,
    CODE_LENGTH as L1CA_CODE_LENGTH,
    CODE_RATE as L1CA_CODE_RATE,
    CARRIER_FREQ as L1CA_CARRIER_FREQ,
)
from scipy.constants import speed_of_light

import project_utils.sample_utils as sample_utils

L1CA_DATA_SYMBOL_RATE = 50


def sample_GPS_L1CA_code(
    prn: int,
    chips: np.ndarray,
    data_bits: Optional[np.ndarray] = None,
    data_bit_start_chip: float = 0,
):
    """
    Samples and returns the code sequence for the given PRN and code phase
    sequence `chips`
    ----------------------------------------------------------------------------
    """
    code_seq = get_GPS_L1CA_code_sequence(prn)
    code_samples = code_seq[chips.astype(int) % L1CA_CODE_LENGTH]
    if data_bits is not None:
        data_phase = (
            (chips - data_bit_start_chip) * L1CA_DATA_SYMBOL_RATE / L1CA_CODE_RATE
        )
        data_samples = data_bits[data_phase.astype(int) % len(data_bits)]
        code_samples = (code_samples + data_samples) % 2
    return code_samples


def sample_GPS_L1CA_signal(
    prn: int,
    time: np.ndarray,
    chip0: float,
    chip_rate: float,
    phi0: float,
    phi_rate: float,
    phi_accel: float = 0,
    data_bits: Optional[np.ndarray] = None,
    data_bit_start_chip: float = 0,
):
    """
    Samples and returns the complex baseband signal sequence given the PRN and signal
    phase parameters.
    ----------------------------------------------------------------------------
    """
    chips = chip0 + chip_rate * time
    phi = phi0 + phi_rate * time + 0.5 * phi_accel * time**2
    code_samples = sample_GPS_L1CA_code(
        prn, chips, data_bits=data_bits, data_bit_start_chip=data_bit_start_chip
    )
    C = 1 - 2 * code_samples
    return C * np.exp(1j * phi)


def coarse_acquire_GPS_L1CA_signal(
    samples: np.ndarray,
    samp_rate: float,
    center_freq: float,
    prn: int,
    L_coh: float = 1,
    N_ncoh: int = 1,
    vrel_max: float = 1000,
    vrel_min: float = -1000,
    return_correlation: bool = False,
    **kwargs
):
    """
    Acquires the Doppler frequency and code phase paramters of a signal.
    --------------------------------------------------------------------------------------------------------------------
    `samples` -- the sample buffer; acquisition will be at the start of the buffer
    `samp_rate` -- the buffer sampling rate
    `center_freq` -- the center frequency of the front end
    `prn` -- satellite PRN
    `L_coh` -- the number of code_periods to coherently integrate
    `N_ncoh` -- the number of coherent blocks to noncoherently integrate
    `vrel_max` -- the maximum relative velocity (m/s), used for generating Doppler search bins
    `vrel_max` -- the minimum relative velocity (m/s), used for generating Doppler search bins
    `return_correlation` -- whether to include the correlation matrix in output

    Returns:
    `outputs` -- dict with `doppler`, `code_phase`, `snr`, `cn0
    """
    T_coh = L_coh * L1CA_CODE_LENGTH / L1CA_CODE_RATE
    N_coh = int(T_coh * samp_rate)
    N = N_ncoh * N_coh

    dopp_max = vrel_max * L1CA_CARRIER_FREQ / speed_of_light
    dopp_min = vrel_min * L1CA_CARRIER_FREQ / speed_of_light
    doppler_bins = np.arange(dopp_min, dopp_max, 1 / T_coh)

    # no data bit handling in coarse acquisition -- may necessitate non-coherent integration for certain coherent integration periods
    correlation = np.zeros((len(doppler_bins), N_coh))
    fft_blocks = np.fft.fft(samples[:N].reshape((N_ncoh, N_coh)), axis=1)
    samp_var = np.var(samples.real[:N])

    time = np.arange(N_coh) / samp_rate
    for i, doppler in enumerate(doppler_bins):
        code_rate = L1CA_CODE_RATE * (1 + doppler / L1CA_CARRIER_FREQ)
        phi_rate = 2 * np.pi * (L1CA_CARRIER_FREQ - center_freq + doppler)
        s_ref = sample_GPS_L1CA_signal(prn, time, 0, code_rate, 0, phi_rate)
        correlation[i] = np.sum(
            abs(np.fft.ifft(np.conj(fft_blocks) * np.fft.fft(s_ref)[None, :])), axis=0
        )

    # Compute number of samples in one code period
    N_1cp = int(L1CA_CODE_LENGTH / L1CA_CODE_RATE * samp_rate)
    corr = correlation[:, :N_1cp]
    dopp_bin, sample_bin = np.unravel_index(corr.argmax(), corr.shape)
    max_val = corr[dopp_bin, sample_bin]
    noise_var = (np.var(corr) * corr.size - max_val**2) / (
        corr.size - 1
    )  # TODO better way of computing noise var?
    snr = max_val**2 / noise_var
    doppler = doppler_bins[dopp_bin]

    # Calculate chip phase from sample phase
    code_phase = sample_bin / N_1cp * L1CA_CODE_LENGTH
    cnr = snr / (N_ncoh * T_coh)

    outputs = {
        "snr": snr,
        "cnr": cnr,
        "doppler": doppler,
        "code_phase": code_phase,
        "noise_var": noise_var,
    }
    if return_correlation:
        outputs["correlation"] = corr
        outputs["doppler_bins"] = doppler_bins
        outputs["code_phase_bins"] = np.arange(N_1cp) / samp_rate * L1CA_CODE_RATE

    return outputs


def fine_acquire_GPS_L1CA_signal(
    samples, samp_rate, center_freq, prn, code_phase_acq, doppler_acq, L_coh, N_blks
):
    """
    This function is performed after coarse acquisition of the L1CA signal.  It refines the Doppler estimate and
    estimates the carrier phase offset.
    --------------------------------------------------------------------------------------------------------------------
    """
    code_rate = L1CA_CODE_RATE * (1 + doppler_acq / L1CA_CARRIER_FREQ)
    N_samples = N_blks * L_coh * L1CA_CODE_LENGTH / code_rate * samp_rate

    first_sample_index = int(
        (L1CA_CODE_LENGTH - code_phase_acq) / code_rate * samp_rate
    )
    samples = samples[first_sample_index:]

    phi_rate = 2 * np.pi * (L1CA_CARRIER_FREQ - center_freq + doppler_acq)
    N = int(L_coh * L1CA_CODE_LENGTH / code_rate * samp_rate)

    samples = samples[: N * N_blks]
    time = np.arange(N * N_blks) / samp_rate
    s_ref = sample_GPS_L1CA_signal(prn, time, 0, code_rate, 0, phi_rate)
    correlation_blocks = (samples * np.conj(s_ref)).reshape((N_blks, N)).sum(axis=-1)

    delta_phase = np.angle(correlation_blocks[1:] / correlation_blocks[:-1])
    indices = np.where((delta_phase - np.mean(delta_phase)) ** 2 < np.var(delta_phase))[0]
    slope = np.mean(delta_phase[indices])
    doppler_delta = slope * samp_rate / (N * 2 * np.pi)
    phases = np.unwrap(np.angle(correlation_blocks))
    t_blks = np.arange(N_blks) * N / samp_rate
    phase_slope, phi0 = np.polyfit(t_blks, phases, 1)

    outputs = {
        "correlation": correlation_blocks,
        "phases": phases,
        "t_blks": t_blks,
        "phi0": phi0,
        "doppler_delta": doppler_delta,
        "doppler": doppler_acq + doppler_delta,
    }
    return outputs


def acquire_GPS_L1CA_data_bit_phase(
    samples: np.ndarray, samp_rate: float, center_freq: float, prn: int, code_phase_acq: float, doppler_acq: float, N_blks: int=500
):
    """
    This function is performed after coarse acquisition (and optionally, after fine acquisition) of the L1CA signal.
    It finds the location of navigation data bit edges to provide a more absolute code phase estimate so that tracking
    or other processes can know when potential bit transitions happen.
    --------------------------------------------------------------------------------------------------------------------
    """
    L_coh = 1
    # There are 20 1ms blocks per data bit, so N_blks needs to be a multiple of 20
    assert N_blks % 20 == 0

    code_rate = L1CA_CODE_RATE * (1 + doppler_acq / L1CA_CARRIER_FREQ)
    N_samples = N_blks * L1CA_CODE_LENGTH / code_rate * samp_rate

    first_sample_index = int(
        (L1CA_CODE_LENGTH - code_phase_acq) / code_rate * samp_rate
    )
    samples = samples[first_sample_index:]

    phi_rate = 2 * np.pi * (L1CA_CARRIER_FREQ - center_freq + doppler_acq)
    generate_reference = lambda time: sample_GPS_L1CA_signal(
        prn, time, 0, code_rate, 0, phi_rate
    )

    N = int(L_coh * L1CA_CODE_LENGTH / code_rate * samp_rate)
    samples = samples[: N * N_blks]
    time = np.arange(N * N_blks) / samp_rate
    s_ref = sample_GPS_L1CA_signal(prn, time, 0, code_rate, 0, phi_rate)
    correlation_blocks = (samples * np.conj(s_ref)).reshape((N_blks, N)).sum(axis=-1)

    corr = []
    for i in range(20):
        coh = np.sum(np.roll(correlation_blocks, i).reshape((-1, 20)), axis=1)
        corr.append(sum(abs(coh)))

    data_bit_phase = np.argmax(corr)
    # I know it seems weird at first but this is correct
    new_code_phase = code_phase_acq + (data_bit_phase - 1) * L1CA_CODE_LENGTH

    outputs = {
        "shift_correlation": corr,
        "data_bit_phase": data_bit_phase,
        "code_phase": new_code_phase,
    }
    return outputs


def acquire_GPS_L1CA_signal(
    filepath: os.PathLike,
    samp_rate: float,
    center_freq: float,
    sample_params: sample_utils.SampleParameters,
    prn: int,
    start_sample: int,
    c_acq_L_coh: float=5,
    c_acq_N_ncoh: int=2,
    f_acq_L_coh: float=1,
    f_acq_N_blks: int=40,
    print_results=True,
):
    """
    Helper function to run all three acquisition functions

    Returns:
        `c_acq, f_acq, n_acq` -- coarse acquisition, fine acquisition, and data bit alignment results
    """
    # Get signal block for acquisition
    sample_loader = sample_utils.SampleLoader(sample_params, samp_rate, max_buffer_duration_ms=1000)

    num_samples = int(1 * samp_rate)
    with open(filepath, "rb") as f:
        sample_block = sample_loader.load_samples(f, start_sample, num_samples)

    c_acq = coarse_acquire_GPS_L1CA_signal(
        sample_block,
        samp_rate,
        center_freq,
        prn,
        c_acq_L_coh,
        c_acq_N_ncoh,
        vrel_max=1000,
        vrel_min=-1000,
        return_correlation=True,
    )

    # Fine acquire
    f_acq = fine_acquire_GPS_L1CA_signal(
        sample_block,
        samp_rate,
        center_freq,
        prn,
        c_acq["code_phase"],
        c_acq["doppler"],
        f_acq_L_coh,
        f_acq_N_blks,
    )

    # Nav bit synchronization
    n_acq = acquire_GPS_L1CA_data_bit_phase(
        sample_block,
        samp_rate,
        center_freq,
        prn,
        c_acq["code_phase"],
        f_acq["doppler"],
        N_blks=500,
    )

    if print_results:
        print(
            "Code Phase: {0:3.3f} chips \tDoppler Freq: {1:3.3f} \t C/N0: {2:3.3f}".format(
                c_acq["code_phase"], c_acq["doppler"], 10 * np.log10(c_acq["cnr"])
            )
        )
        print(
            "Phi0: {0:3.3f} rad \tDoppler Freq: {1:3.3f} \t Dopp. Delta: {2:3.3f}".format(
                f_acq["phi0"], f_acq["doppler"], f_acq["doppler_delta"]
            )
        )
        print("Data Bit Phase: {0:3.3f}".format(n_acq["data_bit_phase"]))

    return c_acq, f_acq, n_acq
