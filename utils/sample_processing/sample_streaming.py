from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class SampleParameters:
    """
    ------------------------------------------------------------------------------------------------
    `bit_depth` -- int, number of bits per numeric value (so an entire complex sample require `2*bit_depth` bits)
    `is_complex` -- bool, whether the samples are complex-valued (versus real-valued)
    `is_integer` -- bool, whether the numeric types are integer (versus float)
    `is_signed` -- bool (default True), whether the numeric types are signed (versus unsigned)
    `is_i_msb` -- bool (default True), whether the I-component occupies the most significant bits of the sample stream
        E.g. for 4-bit samples, specifies whether the I component is in the most significant nibble
    `bytes_per_sample` -- int, number of bytes per sample
    """

    bit_depth: int
    is_complex: bool
    is_integer: bool
    is_signed: bool = True
    is_i_lsb: bool = True

    @property
    def bytes_per_sample(self) -> int:
        if self.is_complex:
            return (self.bit_depth * 2) // 8  # Use integer division
        return self.bit_depth // 8  # Use integer division

    @staticmethod
    def from_dict(sample_parameters_dict: Dict[str, int | bool]) -> "SampleParameters":
        """
        Create a SampleParameters object from a dictionary.
        """
        return SampleParameters(
            bit_depth=sample_parameters_dict["bit_depth"],
            is_complex=sample_parameters_dict["is_complex"],
            is_integer=sample_parameters_dict["is_integer"],
            is_signed=sample_parameters_dict.get("is_signed", True),
            is_i_lsb=sample_parameters_dict.get("is_i_lsb", True),
        )


def get_numpy_dtype(
    is_integer: bool, is_signed: bool, bit_depth: int
) -> Optional[np.dtype]:

    match (is_integer, is_signed, bit_depth):
        case (True, True, 8):
            return np.int8
        case (True, True, 16):
            return np.int16
        case (True, True, 32):
            return np.int32
        # case (True, True, 64):
        #     return np.int64
        case (True, False, 8):
            return np.uint8
        case (True, False, 16):
            return np.uint16
        case (True, False, 32):
            return np.uint32
        case (True, False, 64):
            return np.uint64
        case (False, True, 16):
            return np.float16
        case (False, True, 32):
            return np.float32
        # case (False, True, 64):  # Note: we cannot deal with 64-bit sample components since we restrict ourselves to 32-bit sample buffers
        #     return np.float64
        case (_, _, _):
            return None


def compute_sample_array_size_bytes(
    num_samples: int,
    component_bit_depth: int,
    is_complex: bool,
) -> int:
    """
    Compute the number of bytes needes to store `num_samples` samples.
    This function works even for samples with bit depths less than 8 bits.
    """
    bits_per_sample = component_bit_depth * (2 if is_complex else 1)
    buffer_size_bytes = (num_samples * bits_per_sample) // 8
    if buffer_size_bytes * 8 != (num_samples * bits_per_sample):
        buffer_size_bytes += 1
    return buffer_size_bytes


def convert_to_complex64_samples(
    input_bytes: memoryview,
    output_sample_array: np.ndarray[int, np.complex64],
    sample_params: SampleParameters,
) -> None:
    """
    Convert raw bytes to complex64 samples (each component is float32).
    The input bytes are assumed to be in the format specified by the parameters.
    The output bytes are stored in numpy complex64 format.
    Complex samples are stored in I/Q interleaved format.
    """
    # Unpack parameters
    bit_depth = sample_params.bit_depth
    is_complex = sample_params.is_complex
    is_integer = sample_params.is_integer
    is_signed = sample_params.is_signed
    is_i_lsb = sample_params.is_i_lsb
    
    # Determine sample component dtype; will be None for non-standard bit depths
    input_component_dtype = get_numpy_dtype(is_integer, is_signed, bit_depth)

    # Compute output size in bytes (sanity check)
    if bit_depth >= 8:
        # Use integer division to compute component size
        bytes_per_input_component = bit_depth // 8  
        bytes_per_input_sample = (
            2 * bytes_per_input_component if is_complex else bytes_per_input_component
        )
        num_input_samples = len(input_bytes) // bytes_per_input_sample
        assert (num_input_samples == len(output_sample_array))
    else:
        # For sub-byte samples, we always pack into bytes
        sample_components_per_byte = 8 // bit_depth
        samples_per_byte = (
            sample_components_per_byte // 2 if is_complex else sample_components_per_byte
        )
        num_input_samples = len(input_bytes) * samples_per_byte
        assert (num_input_samples == len(output_sample_array))

    # Need to view as float32 (instead of complex64) so that we can handle I/Q ordering and real samples
    if input_component_dtype is not None:
        raw_sample_array = np.frombuffer(input_bytes, dtype=input_component_dtype)
        if is_complex:
            if is_i_lsb:
                output_sample_array.real = raw_sample_array[0::2]
                output_sample_array.imag = raw_sample_array[1::2]
            else:
                output_sample_array.real = raw_sample_array[1::2]
                output_sample_array.imag = raw_sample_array[0::2]
        else:
            output_sample_array.real = raw_sample_array
            output_sample_array.imag = 0
    elif bit_depth == 4:
        sample_byte_array = np.frombuffer(input_bytes, dtype=np.byte)
        if is_signed:
            comp1 = (sample_byte_array << 4).view(np.int8) >> 4
            comp2 = (sample_byte_array << 0).view(np.int8) >> 4
        else:
            comp1 = (sample_byte_array << 4).view(np.uint8) >> 4
            comp2 = (sample_byte_array << 0).view(np.uint8) >> 4
        if is_complex:
            if is_i_lsb:
                output_sample_array.real = comp1
                output_sample_array.imag = comp2
            else:
                output_sample_array.real = comp2
                output_sample_array.imag = comp1
        else:
            output_sample_array[0::2] = comp1
            output_sample_array[1::2] = comp2

    elif bit_depth == 2:
        sample_byte_array = np.frombuffer(input_bytes, dtype=np.byte)
        # 0xC0 0x30 0x0C 0x03
        if is_signed:
            comp1 = ((sample_byte_array & 0x03) << 6).view(np.int8) >> 6
            comp2 = ((sample_byte_array & 0x0C) << 4).view(np.int8) >> 6
            comp3 = ((sample_byte_array & 0x30) << 2).view(np.int8) >> 6
            comp4 = ((sample_byte_array & 0xC0) << 0).view(np.int8) >> 6
        else:
            comp1 = ((sample_byte_array & 0x03) << 6).view(np.uint8) >> 6
            comp2 = ((sample_byte_array & 0x0C) << 4).view(np.uint8) >> 6
            comp3 = ((sample_byte_array & 0x30) << 2).view(np.uint8) >> 6
            comp4 = ((sample_byte_array & 0xC0) << 0).view(np.uint8) >> 6
        if is_complex:
            if is_i_lsb:
                output_sample_array.real[0::2] = comp1
                output_sample_array.imag[0::2] = comp2
                output_sample_array.real[1::2] = comp3
                output_sample_array.imag[1::2] = comp4
            else:
                output_sample_array.real[0::2] = comp2
                output_sample_array.imag[0::2] = comp1
                output_sample_array.real[1::2] = comp4
                output_sample_array.imag[1::2] = comp3
        else:
            output_sample_array[0::4] = comp1
            output_sample_array[1::4] = comp2
            output_sample_array[2::4] = comp3
            output_sample_array[3::4] = comp4
    else:
        raise NotImplemented()
