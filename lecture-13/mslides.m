% Generate code for reading 16-bit signed integers from raw bytes
raw_bytes = uint8([0x34, 0x12, 0x78, 0x56, 0xFF, 0x7F, 0x00, 0x80]); % Example raw byte data

% Raw bytes can be typecast directly to int16
int16_result_native = typecast(raw_bytes, 'int16');

% Example for when machine endianness needs to be swapped
int16_result_swapped = swapbytes(typecast(raw_bytes, 'int16'));





% Generate code for reading 4-bit signed integers from raw bytes
raw_bytes = uint8([0x12, 0x34, 0x56, 0x78]); % Example raw byte data for 4-bit integers

int8_arr = typecast(raw_bytes, 'int8'); % Typecast to int8 first
% Note: MATLAB does not have arithmetic shift operations
real = (int8_arr & 0xF0) / 16; % Extract high nibble
imag = ((int8_arr << 4) & 0xF0) / 16; % Extract low nibble



% Generate code for reading 8-bit signed integers from raw bytes
% Then separate into real and imaginary parts
raw_bytes = uint8([0x12, 0x34, 0x56, 0x78]); % Example raw byte data for 8-bit integers
int8_array = typecast(raw_bytes, 'int8');
real = int8_array(1:2:end); % even indices
imag = int8_array(2:2:end); % odd indices