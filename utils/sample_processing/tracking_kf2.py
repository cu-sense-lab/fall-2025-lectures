import numpy as np
from dataclasses import dataclass
from .tracking import (
    TrackingSignalParameters,
)
from .correlation import correlate__delay
import scipy.linalg

speed_of_light = 299792458.0  # m/s

class KFSignalState:
    """Kalman filter state for a tracked signal.
    [code_phase_seconds, carrier_phase_rad, doppler_rad_per_sec, accel_rad_per_sec2]
    """

    def __init__(
            self,
            uptime_epoch_seconds: float,
            code_phase_seconds: float,
            carrier_phase_rad: float,
            doppler_rad_per_sec: float,
            initial_code_phase_uncertainty: float = 1e-6, # seconds
            initial_carrier_phase_uncertainty: float = np.pi,  # radians
            initial_doppler_uncertainty: float = 100.0,  # rad/s
    ):
        self.uptime_epoch_seconds = uptime_epoch_seconds
        self.code_phase_seconds = code_phase_seconds
        self.carrier_phase_rad = carrier_phase_rad
        self.doppler_rad_per_sec = doppler_rad_per_sec

        self.state_uncertainty_covar = np.diag([
            initial_code_phase_uncertainty**2,
            initial_carrier_phase_uncertainty**2,
            initial_doppler_uncertainty**2,
        ])

    @property
    def state_vector(self) -> np.ndarray:
        return np.array([
            self.code_phase_seconds,
            self.carrier_phase_rad,
            self.doppler_rad_per_sec,
        ])

def construct_R_matrix(
        cn0_dbhz: float,
        nominal_integration_time_sec: float,
        B_DLL_hz: float
) -> np.ndarray:
    T = nominal_integration_time_sec
    # Convert C/N0 from dB-Hz to linear
    cn0 = 10 ** (cn0_dbhz / 10.0)
    # Compute noise variance for code phase and carrier phase measurements
    sigma2_code_phase = B_DLL_hz / cn0 * (1 + 2 / T / cn0)
    sigma2_carrier_phase = 1 / (2 * T * cn0) * (1 + 1 / (2 * T * cn0))
    return np.diag([sigma2_code_phase, sigma2_carrier_phase])


def construct_Q_matrix(
        q_eta: float,
        q_b: float,
        q_d: float,
        q_a: float,
        omega_c: float,
        dt: float,
) -> np.ndarray:
    # start from continuous-time model
    # A = np.array([
    #     [0, 0, 1/omega_c, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1],
    #     [0, 0, 0, 0]
    # ])
    # Q = np.array([
    #     [q_eta + q_b, q_b / omega_c, 0, 0],
    #     [q_b / omega_c, omega_c**2 * q_b, 0, 0],
    #     [0, 0, omega_c**2 * q_d, 0],
    #     [0, 0, 0, omega_c**2 / speed_of_light**2 * q_a]
    # ])
    # N = 4
    A = np.array([
        [0, 0, 1/omega_c],
        [0, 0, 1],
        [0, 0, 0]
    ])
    Q = np.array([
        [q_eta + q_b, q_b / omega_c, 0],
        [q_b / omega_c, omega_c**2 * q_b, 0],
        [0, 0, omega_c**2 * q_d],
    ])
    N = 3
    F = np.zeros((2 * N, 2 * N))
    F[:N, :N] = -A
    F[:N, N:] = Q
    F[N:, N:] = A.T
    G = scipy.linalg.expm(F * dt)
    A_d = G[N:, N:].T
    Q_d = A_d @ G[0:N, N:2 * N]
    return Q_d


@dataclass
class KFLoopParameters:
    update_period_ms: int
    integration_period_ms: int
    carrier_freq_hz: float
    q_eta: float
    q_b: float
    q_d: float
    q_a: float
    samp_rate: float
    nominal_cn0_dbhz: float
    EPL_chip_spacing: float = 0.5  # chips

    def __post_init__(self):
        integration_period = self.integration_period_ms * 1e-3
        block_length_samples = int(self.samp_rate * integration_period)
        self.block_sample_time_arr = np.arange(block_length_samples) / self.samp_rate
        dt = self.update_period_ms * 1e-3
        self.update_time_delta = dt
        omega_c = 2.0 * np.pi * self.carrier_freq_hz
        self.Q = construct_Q_matrix(
            self.q_eta,
            self.q_b,
            self.q_d,
            self.q_a,
            omega_c=omega_c,
            dt=dt
        )
        self.F = np.array([
            [1, 0, dt / omega_c],
            [0, 1, dt],
            [0, 0, 1],
        ])
        self.H = np.array([
            [1, 0, 0],
            [0, 1, dt / 2]
        ])
        self.R = construct_R_matrix(
            self.nominal_cn0_dbhz,
            nominal_integration_time_sec=dt,
            B_DLL_hz=1.0  # Placeholder, could be parameterized
        )
        self.I_state = np.eye(3)


@dataclass
class SignalTrackingOutputs:

    def __init__(self, output_capacity: int):
        self.uptime_seconds = np.zeros(output_capacity, dtype=float)
        self.carr_phase_errors_cycles = np.zeros(output_capacity, dtype=float)
        self.code_phase_errors_chips = np.zeros(output_capacity, dtype=float)
        self.early_corr = np.zeros(output_capacity, dtype=complex)
        self.prompt_corr = np.zeros(output_capacity, dtype=complex)
        self.late_corr = np.zeros(output_capacity, dtype=complex)
        self.code_phase_seconds = np.zeros(output_capacity, dtype=float)
        self.carr_phase_cycles = np.zeros(output_capacity, dtype=float)
        self.doppler_freq_hz = np.zeros(output_capacity, dtype=float)

        self.code_phase_uncertainty_seconds = np.zeros(output_capacity, dtype=float)
        self.carr_phase_uncertainty_cycles = np.zeros(output_capacity, dtype=float)
        self.doppler_uncertainty_hz = np.zeros(output_capacity, dtype=float)

        self.state_delta = np.zeros((output_capacity, 3), dtype=float)
        self.kalman_gain = np.zeros((output_capacity, 3, 2), dtype=float)
        self.state_error_covar = np.zeros((output_capacity, 3, 3), dtype=float)
        self.meas_vector = np.zeros((output_capacity, 2), dtype=float)

        self.output_index = 0

def track_signal(
    uptime_seconds: float,
    sample_block: np.ndarray,
    signal_params: TrackingSignalParameters,
    signal_state: KFSignalState,  # *IO
    loop_params: KFLoopParameters,
    signal_outputs: SignalTrackingOutputs,  # *IO
) -> None:
    # Unpack values
    code_seq = signal_params.code_seq
    nominal_code_rate_chips_per_sec = signal_params.nominal_code_rate_chips_per_sec
    code_length_chips = signal_params.code_length_chips
    carrier_freq_hz = signal_params.carrier_freq_hz
    omega_c = 2.0 * np.pi * carrier_freq_hz
    EPL_chip_spacing = loop_params.EPL_chip_spacing

    adjusted_code_rate_sec_per_sec = (1 + signal_state.doppler_rad_per_sec / omega_c)

    # Propagate carrier and code phases to start of block
    # x_k_minus = F @ x_k_plus
    time_delta = uptime_seconds - signal_state.uptime_epoch_seconds
    signal_state.code_phase_seconds += adjusted_code_rate_sec_per_sec * time_delta
    signal_state.carrier_phase_rad += signal_state.doppler_rad_per_sec * time_delta

    F = loop_params.F
    Q = loop_params.Q
    P = signal_state.state_uncertainty_covar
    P = F @ P @ F.T + Q

    # Perform correlation
    early, prompt, late = correlate__delay(
        sample_block,
        loop_params.samp_rate,
        signal_state.carrier_phase_rad / (2.0 * np.pi),
        signal_state.doppler_rad_per_sec / (2.0 * np.pi),
        code_seq,
        code_length_chips,
        adjusted_code_rate_sec_per_sec * nominal_code_rate_chips_per_sec,
        signal_state.code_phase_seconds * nominal_code_rate_chips_per_sec,
        3,
        EPL_chip_spacing,
        -EPL_chip_spacing,
    )

    # Let t be RX time and t_sys be system time

    # t_tx(t) = t_sys(t) + Dt_tx(t)
    # t_sys(t) = t - Dt_rx(t)

    # signal is transmitted x(t) = t_tx(t)
    # signal is received; at antenna x(t) = t_tx(t - tau(t))
    # signal is sampled; eta(t_k) = t_tx(t_k - tau(t_k)) = t_rx(t_k) - rho(t_k) / c
    # phi(t_k) = -rho(t_k) * 2 * pi / c

    # in order to simulate x[k], first identify t_k
    # t_k = t_rx(t_0) + k * dt - Dt_rx(t_k)


    # I have two time coordinates, t_1 and t_2
    # I have a function f(t_1)
    # Determine 

    # x[k] = t_rx[k] - rho[k] / c

    # t_rx(t_k) = t_rx(t_0) + k * dt 
    #           = t_0 + Dt_rx(t_0) + k * dt
    #           = t_k + Dt_rx(t_k)
    # t_k = t_rx(t_0) + k * dt - Dt_rx(t_k)

    # eta(t) = t_tx(t - tau(t)) = t + Dt_tx(t-tau) - tau(t)
    #                           = t_rx(t) - rho(t) / c

    # phi(t) = t_tx(t - tau(t)) - t_rx(t) = -rho(t) / c

    # x(t) = A(t) * C(t_rx(t) - rho(t)) * exp(-j * rho(t) *2*pi/c)

    # Estimate state errors from correlator outputs
    # Define measurement vector for state delta
    # dy_k = H @ dx_k_minus + v_k
    # Costas (arctan) discriminator for carrier phase error
    delta_theta = np.arctan(prompt.imag / prompt.real)
    # delta_eta = EPL_chip_spacing / nominal_code_rate_chips_per_sec * (np.abs(early) - np.abs(late)) / (np.abs(early) + np.abs(late))
    delta_eta = EPL_chip_spacing / nominal_code_rate_chips_per_sec * (np.abs(early) - np.abs(late)) / (np.abs(early) + np.abs(late) + 2 * np.abs(prompt))
    meas_vector = np.array([delta_eta, delta_theta])

    # Apply loop filter to errors
    P_prior = signal_state.state_uncertainty_covar
    H = loop_params.H
    R = loop_params.R
    K = P_prior @ H.T @ scipy.linalg.inv(H @ P_prior @ H.T + R)
    state_delta = K @ meas_vector

    # P_post = (np.eye(4) - K @ H) @ P_prior
    I = loop_params.I_state
    P_post = (I - K @ H) @ P_prior @ (I - K @ H).T + K @ R @ K.T
    # check symmetry
    P_post = 0.5 * (P_post + P_post.T)

    # Update signal state
    # signal_state.code_phase_seconds += state_delta[0]
    # signal_state.carrier_phase_rad += state_delta[1]
    # signal_state.doppler_rad_per_sec += state_delta[2]

    signal_state.carrier_phase_rad += delta_theta * 0
    signal_state.doppler_rad_per_sec += delta_theta * 100

    signal_state.state_uncertainty_covar[...] = P_post

    signal_state.uptime_epoch_seconds = uptime_seconds

    # Store outputs
    output_idx = signal_outputs.output_index
    signal_outputs.uptime_seconds[output_idx] = uptime_seconds
    signal_outputs.carr_phase_errors_cycles[output_idx] = delta_theta
    signal_outputs.code_phase_errors_chips[output_idx] = delta_eta
    signal_outputs.early_corr[output_idx] = early
    signal_outputs.prompt_corr[output_idx] = prompt
    signal_outputs.late_corr[output_idx] = late
    signal_outputs.code_phase_seconds[output_idx] = signal_state.code_phase_seconds
    signal_outputs.carr_phase_cycles[output_idx] = signal_state.carrier_phase_rad / (2.0 * np.pi)
    signal_outputs.doppler_freq_hz[output_idx] = signal_state.doppler_rad_per_sec / (2.0 * np.pi)

    signal_outputs.code_phase_uncertainty_seconds[output_idx] = np.sqrt(P_post[0, 0])
    signal_outputs.carr_phase_uncertainty_cycles[output_idx] = np.sqrt(P_post[1, 1]) / (2.0 * np.pi)
    signal_outputs.doppler_uncertainty_hz[output_idx] = np.sqrt(P_post[2, 2]) / (2.0 * np.pi)

    signal_outputs.state_delta[output_idx, :] = state_delta
    signal_outputs.kalman_gain[output_idx, ...] = K
    signal_outputs.state_error_covar[output_idx, ...] = P_post
    signal_outputs.meas_vector[output_idx, :] = meas_vector
    
    signal_outputs.output_index += 1