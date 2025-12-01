import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


# There are various parameters that go into the coarse acquisition process.
# We partition these parameters into two groups:
# 1. Parameters that are common to all signals; we call thsese the `AcquisitionProcessorParameters`.
# 2. Parameters that are specific to each signal; we call these the `SignalAcquisitionParameters`.
#
# The coarse acquisiton result includes references to both of these parameter groups, as well as
# the estimated signal state, SNR, and correlation array.


@dataclass
class AcquisitionConfiguration:
    replica_duration_ms: int
    num_blocks: int
    block_step_ms: int
    sample_rate: int

    def __post_init__(self):
        self.replica_length_samples = int(
            self.sample_rate * self.replica_duration_ms / 1000
        )
        self.replica_time_arr = (
            np.arange(self.replica_length_samples) / self.sample_rate
        )

        # self.block_size_samples = 1 << (self.replica_length_samples - 1).bit_length()
        self.block_size_samples = self.replica_length_samples
        self.block_duration_seconds = self.block_size_samples / self.sample_rate
        self.fft_resolution = 1 / self.block_duration_seconds
        assert self.num_blocks > 0
        self.acq_total_duration_ms = (
            self.num_blocks - 1
        ) * self.block_step_ms + self.replica_duration_ms
        self.total_num_samples = int(
            self.acq_total_duration_ms / 1000 * self.sample_rate
        )

@dataclass
class AcqSignalCodeParameters:
    rate_chips_per_sec: float
    length_chips: int
    sequence: np.ndarray[np.int8]

import utils.signals.gps_l1ca as gps_l1ca
GPS_L1CA_ACQ_CODE_PARAMS = {
    f"G{prn:02}": AcqSignalCodeParameters(
        code_rate_chips_per_sec=gps_l1ca.CODE_RATE,
        code_length_chips=gps_l1ca.CODE_LENGTH,
        code_sequence=gps_l1ca.get_GPS_L1CA_code_sequence(prn),
    ) for prn in range(1, 33)
}

@dataclass
class AcquisitionResult:
    pass


@dataclass
class SignalReplicaCacheEntry:
    replica_samples: np.ndarray
    replica_fft: np.ndarray


class AcquisitionProcessor_BPSK:
    """
    Performs coarse acquisition for a set of BPSK signals.

    Acquisition parameters are stored for each signal in the `acq_params` dictionary.

    Pre-computed replicas and FFTs are stored for each signal in the `replica_cache` dictionary.
    The values in the cache entries must be recomputed when acquisition parameters change or when the sampling rate changes.
    """

    def __init__(
        self,
        acq_config: AcquisitionConfiguration,
        code_parameters: Dict[str, AcqSignalCodeParameters],
    ):
        self.acq_config = acq_config
        self.code_parameters = code_parameters
        self.replica_cache_entry_dict: Dict[str, SignalReplicaCacheEntry] = {}

        # We internally store all acquisition results for caching and debug purposes
        self.acq_results_dict: Dict[str, AcquisitionResult] = {}


    def process_sample_block(
        self,
        sample_block: np.ndarray[np.complex64]
    ) -> bool:
        
        if (
            len(sample_block)
            < self.acq_config.acq_total_duration_ms / 1000 * self.acq_config.sample_rate
        ):
            logging.error(
                "Insufficient samples to perform acquisition"
            )  # change to logger
            return False

        M = self.acq_config.num_blocks
        N = self.acq_config.block_size_samples
        samples = sample_block[: M * N].reshape((M, N))
        samples_var = np.var(
            samples[0, :]
        )  # <- Estimate of raw sample variance, can be used for noise stats
        samples_fft = np.fft.fft(samples, axis=1)

        for signal_id, code_params in self.code_parameters.items():

            # Check if there is a cached replica for this signal
            if signal_id in self.replica_cache_entry_dict:
                replica_entry = self.replica_cache_entry_dict[signal_id]
                replica_samples_fft = replica_entry.replica_fft
            else:
                replica_samples = np.zeros(
                    self.acq_config.block_size_samples, dtype=np.complex64
                )
                chips_arr = 0.0 + self.acq_config.replica_time_arr * code_params.rate_chips_per_sec
                chip_indices = chips_arr.astype(int) % code_params.length_chips
                replica_samples[:self.acq_config.replica_length_samples] = code_params.sequence[chip_indices].astype(float)
                replica_samples_fft = np.fft.fft(replica_samples)
                replica_entry = SignalReplicaCacheEntry(
                    replica_samples, replica_samples_fft
                )
                self.replica_cache_entry_dict[signal_id] = replica_entry

            min_doppler_fft_bin = int(
                sig_acq_params.search_doppler_min_hz / self.acq_config.fft_resolution
            )
            max_doppler_fft_bin = int(
                sig_acq_params.search_doppler_max_hz / self.acq_config.fft_resolution
            )
            doppler_search_bins = range(min_doppler_fft_bin, max_doppler_fft_bin)
            correlation = np.zeros(
                (len(doppler_search_bins), self.acq_config.block_size_samples)
            )

            for i, roll in enumerate(doppler_search_bins):
                corr = (
                    1 / (N * samples_var) * np.fft.ifft(
                        np.conj(np.roll(samples_fft, -roll, axis=1)) * replica_samples_fft[None, :]
                    )
                )
                correlation[i] = np.sum(
                    np.abs(corr) ** 2, axis=0
                )  # <- noise should be chi-squared with M deg of freedom
                # but we don't really know the noise distribution because correlation sidelobes are mixed in if signal is strong
                # therefore, first find peak, then estimate noise distr, the compute threshold

            # Compute number of samples in one code period
            num_samples_per_code_period = int(
                code_params.length / code_params.rate * self.acq_config.sample_rate
            )
            correlation = correlation[:, :num_samples_per_code_period]

            # Find acquisition peak
            dopp_bin, sample_bin = np.unravel_index(
                correlation.argmax(), correlation.shape
            )
            max_val = correlation[dopp_bin, sample_bin]

            # Notes on noise distribution:
            #  N noise samples are summed to produce Gaussian noise with `N * samp_var` variance
            #  The magnitude of those complex noise samples is added `M` times
            #  The noise cells in the resulting correlation array have chi-squared distribution with
            #  `2 * M` degrees of freedom.  We will approximate that the cell containing our signal
            #  peak has magnitude `M * A**2 + noise`
            # TODO: is there a better way of computing noise var?  should fit to chi squared?
            # import numpy as np

            # import scipy.stats

            # scipy.stats.chi2()
            # scipy.stats.chi2.ppf()

            # false alarm rate of alpha = 0.00001
            # raw sample noise std of sigma
            # coherent gain of N samples, leads to amplitude N*A and noise std of sqrt(N) * sigma
            # normalize this result to get signal peak of sqrt(N) * A / sigma and noise std of 1
            # non-coherent integration of M blocks of squared amplitude leads to noise bins with chi2 with 2*M degrees of freedom
            # and signal peak of N * A**2 / sigma**2
            # but the main point really is that, under than normalization scheme, the CFAR threshold is related to CDF of chi2

            noise_var = (np.var(correlation) * correlation.size - max_val**2) / (
                correlation.size - 1
            )

            # Estimate SNR and CNR
            acq_snr = max_val**2 / noise_var
            acq_snr_db = 10 * np.log10(acq_snr)
            # noise_mean_est = M * np.sqrt(N)
            # A_est = np.sqrt(max_val - noise_mean_est)
            # cnr_est = A_est**2 / samples_var * self.samp_rate

            # Calculate Doppler from dopp bin offset
            acq_doppler_hz = (
                min_doppler_fft_bin + dopp_bin
            ) * self.acq_config.fft_resolution

            # Calculate code phase from sample phase
            # TODO: when should this include Doppler effect on code phase?
            #  did this calculation -- almost never, but double check for higher frequencies..
            # acq_code_phase_chips = sample_bin * sig_params.mod.code.rate / samp_rate
            # acq_code_phase: GTime = GTime.from_float_seconds(acq_code_phase_chips / sig_params.mod.code.rate)
            acq_code_phase_seconds = sample_bin / self.acq_config.sample_rate

            acq_corr_result = CorrelationResult(
                correlation,
                min_doppler_fft_bin * self.acq_config.fft_resolution,
                self.acq_config.fft_resolution,
                0.0,
                1
                / self.acq_config.sample_rate,  # TODO: is this right for doppler compression/expansion case?
            )

            acq_result = AcquisitionResult_BPSK(
                signal_id,
                block_start_epoch_uptime,
                acq_doppler_hz,
                acq_code_phase_seconds,
                acq_snr_db,
                acq_corr_result,
                self.acq_config,
                sig_acq_params,
            )
            
            self.acq_results_dict[signal_id] = acq_result

            # First check if SNR passed acq threshold, and if not, continue
            if acq_snr_db < sig_acq_params.snr_threshold_db:
                # print(f"Failed to acquire: {signal_id}: {acq_result}")
                # logging.log(2, f"Failed to acquire: {signal_id}: {acq_result}")
                continue

            # logging.log(2, f"Acquired: {signal_id}: {acq_result}")
            print(f"Acquired: {signal_id}: {acq_result}")
            # Otherwise, append the acquisition result to the acquired signals list
            self._acquired_signals.append(acq_result)

        return StatusCode.OKAY

