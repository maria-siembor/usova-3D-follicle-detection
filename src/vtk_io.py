"""Minimal reader/writer for the legacy VTK STRUCTURED_POINTS format used
by USOVA3D. Legacy VTK BINARY data is big-endian by spec, regardless of
platform, values written here account for that."""

import numpy as np


def read_vtk(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = path_or_bytes
    else:
        with open(path_or_bytes, 'rb') as f:
            data = f.read()

    header = {}
    pos = 0
    for _ in range(10):
        nl = data.index(b'\n', pos)
        line = data[pos:nl].decode('latin-1').strip()
        pos = nl + 1
        if line.startswith('DIMENSIONS'):
            header['dims'] = tuple(int(x) for x in line.split()[1:])
        elif line.startswith('SPACING'):
            header['spacing'] = tuple(float(x) for x in line.split()[1:])
        elif line.startswith('ORIGIN'):
            header['origin'] = tuple(float(x) for x in line.split()[1:])
        elif line.startswith('SCALARS'):
            header['dtype_name'] = line.split()[2]
        elif line.startswith('LOOKUP_TABLE'):
            break

    dtype_map = {'unsigned_char': '>u1', 'char': '>i1', 'short': '>i2',
                 'unsigned_short': '>u2', 'int': '>i4', 'float': '>f4'}
    dtype = np.dtype(dtype_map.get(header['dtype_name'], '>u1'))

    nx, ny, nz = header['dims']
    n_voxels = nx * ny * nz
    n_bytes = n_voxels * dtype.itemsize
    arr = np.frombuffer(data[pos:pos + n_bytes], dtype=dtype)

    # VTK STRUCTURED_POINTS stores data with x varying fastest, then y, then z
    volume = arr.reshape((nz, ny, nx))  # numpy: last axis varies fastest -> matches x-fastest

    return volume, header['spacing'], header['origin']


def write_vtk_labels(volume, spacing, path, dataset_name="scalars"):
    """Write an integer-labeled volume in the format the official
    USOVA3D evaluation tool expects (unsigned_char scalars, BINARY)."""
    nz, ny, nx = volume.shape
    header = (
        f"# vtk DataFile Version 1.0\n"
        f"{dataset_name}\n"
        f"BINARY\n"
        f"DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx} {ny} {nz}\n"
        f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n"
        f"ORIGIN 0 0 0\n"
        f"POINT_DATA {nx*ny*nz}\n"
        f"SCALARS scalars unsigned_char\n"
        f"LOOKUP_TABLE default\n"
    ).encode('latin-1')

    with open(path, 'wb') as f:
        f.write(header)
        f.write(volume.astype('>u1').tobytes())