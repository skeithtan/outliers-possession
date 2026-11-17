"""Quantisation and packing of the binary files of the local bundle (site/data/SCHEMA.md 3.2, 3.3).

    encode_i16(values, scale)   float array to int16 steps, NaN to the sentinel -32768
    encode_u16(values, scale)   float array to uint16 steps, NaN to 65535
    encode_u8(values, scale)    float array to uint8 steps, NaN is an error
    encode_i8(values)           integer array to int8, no scale
    decode(array, scale, missing)  the inverse, the sentinel back to NaN
    pack(arrays)                one struct-of-arrays byte string and its header entries

Every function refuses a value that does not fit. It never clips in silence.
"""

import numpy as np

# Missing-value sentinels (SCHEMA.md 3.2 and 3.3).
I16_SENTINEL = -32768
U16_MISSING = 65535
# Default scale of a position: one step is 1 cm (SCHEMA.md 3.2).
CM = 0.01
# Arrays start at a multiple of this many bytes, so an Int16Array view can read them
# (SCHEMA.md 3.2: "Every array starts at an even byte offset").
ALIGN = 2

# Little-endian dtype of each type name in the headers (SCHEMA.md Section 1: little-endian).
DTYPES = {"int8": "<i1", "uint8": "<u1", "int16": "<i2", "uint16": "<u2"}


def _steps(values, scale):
    """values / scale rounded half to even, as float64, with the NaN mask."""
    values = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(values)
    steps = np.zeros(values.shape, dtype=np.float64)
    steps[~missing] = np.round(values[~missing] / scale)
    return steps, missing


def _check_range(steps, missing, low, high, name):
    """Raise OverflowError if a present step lies outside [low, high]."""
    bad = ~missing & ((steps < low) | (steps > high))
    if bad.any():
        raise OverflowError("{}: {} values outside [{}, {}] steps".format(
            name, int(bad.sum()), low, high))


def encode_i16(values, scale=CM, name="array"):
    """int16 steps of `values`. NaN and infinity become I16_SENTINEL."""
    steps, missing = _steps(values, scale)
    _check_range(steps, missing, I16_SENTINEL + 1, 32767, name)
    out = steps.astype(np.int16)
    out[missing] = I16_SENTINEL
    return out


def encode_u16(values, scale=CM, name="array"):
    """uint16 steps of `values`. NaN and infinity become U16_MISSING."""
    steps, missing = _steps(values, scale)
    _check_range(steps, missing, 0, U16_MISSING - 1, name)
    out = steps.astype(np.uint16)
    out[missing] = U16_MISSING
    return out


def encode_u8(values, scale=1.0 / 255, name="array"):
    """uint8 steps of `values`. A missing value is an error, because uint8 has no sentinel."""
    steps, missing = _steps(values, scale)
    if missing.any():
        raise ValueError("{}: {} missing values, uint8 has no sentinel".format(
            name, int(missing.sum())))
    _check_range(steps, missing, 0, 255, name)
    return steps.astype(np.uint8)


def encode_i8(values, name="array"):
    """int8 copy of an integer array. Raise if a value does not fit."""
    values = np.asarray(values)
    if values.size and (values.min() < -128 or values.max() > 127):
        raise OverflowError("{}: values outside the int8 range".format(name))
    return values.astype(np.int8)


def decode(array, scale=CM, missing=I16_SENTINEL):
    """float64 values of a quantised array. The `missing` step becomes NaN."""
    array = np.asarray(array)
    out = array.astype(np.float64) * scale
    if missing is not None:
        out[array == missing] = np.nan
    return out


def pack(arrays):
    """Struct of arrays: (bytes, {name: {offset, shape, dtype}}).

    `arrays` is a list of (name, array) pairs in file order. Each array is written whole,
    row-major and little-endian. Zero bytes pad each start to an even offset.
    """
    chunks, header, offset = [], {}, 0
    names = set()
    for name, array in arrays:
        if name in names:
            raise ValueError("duplicate array name {!r}".format(name))
        names.add(name)
        array = np.asarray(array)
        dtype = array.dtype.name
        if dtype not in DTYPES:
            raise ValueError("{}: dtype {} is not in {}".format(name, dtype, sorted(DTYPES)))
        pad = (-offset) % ALIGN
        if pad:
            chunks.append(b"\0" * pad)
            offset += pad
        data = np.ascontiguousarray(array.astype(DTYPES[dtype])).tobytes()
        header[name] = {"offset": offset, "shape": [int(n) for n in array.shape],
                        "dtype": dtype}
        chunks.append(data)
        offset += len(data)
    return b"".join(chunks), header


def unpack(blob, header):
    """{name: array} views of a pack() result, for tests and checks."""
    out = {}
    for name, entry in header.items():
        dtype = np.dtype(DTYPES[entry["dtype"]])
        count = int(np.prod(entry["shape"])) if entry["shape"] else 1
        flat = np.frombuffer(blob, dtype=dtype, count=count, offset=entry["offset"])
        out[name] = flat.reshape(entry["shape"])
    return out
