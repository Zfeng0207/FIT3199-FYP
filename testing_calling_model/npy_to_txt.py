# npy_to_txt.py
import numpy as np
import argparse
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0

def npy_to_txt(
    npy_path: str,
    txt_path: str,
    fmt: str = '%.6f',
    delimiter: str = ' '
) -> None:
    """
    Convert a NumPy .npy file (numeric array or header-only) to a plain-text .txt file.

    - Numeric arrays: writes rows as formatted numbers.
    - Header-only .npy files: parses the header and writes key: value lines.
    """
    try:
        # Attempt to load as a numeric array
        arr = np.load(npy_path)
    except Exception:
        # Fallback for header-only .npy: parse the header
        with open(npy_path, 'rb') as f:
            version = read_magic(f)
            if version == (1, 0):
                header = read_array_header_1_0(f)
            else:
                header = read_array_header_2_0(f)
        # Write header dict to txt
        with open(txt_path, 'w', encoding='utf-8') as out:
            for key, val in header.items():
                out.write(f"{key}: {val}\n")
        return

    # For numeric arrays, reshape to 2D if needed
    if arr.ndim == 1:
        arr2d = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr2d = arr.reshape(arr.shape[0], -1)
    else:
        arr2d = arr

    # Save array data as plain-text
    np.savetxt(txt_path, arr2d, fmt=fmt, delimiter=delimiter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a .npy file (array or header) to a plain-text .txt file'
    )
    parser.add_argument('npy_path', help='Path to the input .npy file')
    parser.add_argument('txt_path', help='Path to save the output .txt file')
    parser.add_argument(
        '--fmt', '-f',
        default='%.6f',
        help='Format specifier for numeric elements'
    )
    parser.add_argument(
        '--delimiter', '-d',
        default=' ',
        help='Delimiter between numeric elements'
    )
    args = parser.parse_args()
    npy_to_txt(
        args.npy_path,
        args.txt_path,
        fmt=args.fmt,
        delimiter=args.delimiter
    )
