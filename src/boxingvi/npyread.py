"""Reading an untrusted .npy header before anything is allocated, then its data.

`np.load` allocates whatever shape a header claims before it checks that the data
is there: a 128-byte .npy claiming 2M frames made it allocate 816 MB and only then
fail. Callers here read the header first (`read_header`), check the dtype and shape
against what they expect, bound the size against what the source can hold, and
only then read the data (`read_data`) into one buffer of exactly the validated
size. Shared by the pose-file loader (`pose_yolo`) and the corpus loader (`data`).
Never unpickles: neither function has an `allow_pickle` path at all.
"""

from __future__ import annotations

import numpy as np

# The most a header may take: 8 bytes of magic and version, a 4-byte length, and
# numpy's own 10,000-byte header limit -- which numpy checks only AFTER reading the
# length a header claims (up to 4 GB in format 2.0), so the stream is capped here.
NPY_HEADER_MAX = 8 + 4 + 10_000
READ_CHUNK = 1 << 24   # array data is read in 16 MB pieces straight into its final buffer


class Capped:
    """A stream that yields at most `left` bytes, then reports end of file."""

    def __init__(self, raw, left: int) -> None:
        self.raw, self.left = raw, left

    def read(self, n: int = -1) -> bytes:
        data = self.raw.read(self.left if n is None or n < 0 else min(n, self.left))
        self.left -= len(data)
        return data


def read_header(stream, what: str):
    """(shape, fortran_order, dtype) from a .npy header, read through a byte cap.
    Leaves `stream` positioned at the first byte of array data (numpy reads exact
    sizes). The dtype is NOT checked here: every caller knows what it expects."""
    capped = Capped(stream, NPY_HEADER_MAX)
    try:
        version = np.lib.format.read_magic(capped)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(capped)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(capped)
        # 3.0 exists only for structured dtypes with non-latin-1 names, refused anyway
        raise ValueError(f"{what} uses .npy format {version}, not 1.0 or 2.0")
    except ValueError:
        raise
    except Exception as e:
        # numpy's header parser lets other errors out of a malformed header: TokenError
        # (unterminated, or 300 nested parentheses), IndexError (`'descr': ()`),
        # RecursionError (`1+1+...+1`) and MemoryError (a parser stack overflow on
        # ~9,000 minus signs -- nothing large is allocated). The header is capped at
        # NPY_HEADER_MAX bytes, so each is a malformed file, not the machine failing.
        raise ValueError(f"{what} has an unparseable .npy header") from e


def read_data(stream, shape, fortran: bool, dtype, what: str) -> np.ndarray:
    """Read exactly the bytes the (already validated) header promises, straight into
    one buffer of that size (`readinto`: a plain file fills it with no second copy; a
    zip member goes through at most one READ_CHUNK of its own). Short data is refused;
    trailing bytes are ignored, as `np.load` ignores them."""
    arr = np.empty(shape[::-1] if fortran else shape, dtype=dtype)
    view = memoryview(arr.reshape(-1).view(np.uint8))  # C-contiguous: a view, not a copy
    pos = 0
    while pos < len(view):
        n = stream.readinto(view[pos:pos + READ_CHUNK])
        if not n:
            raise ValueError(f"{what} is truncated ({pos} of {len(view)} bytes)")
        pos += n
    return arr.T if fortran else arr
