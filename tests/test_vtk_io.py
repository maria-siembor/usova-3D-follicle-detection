import numpy as np

from vtk_io import read_vtk, write_vtk_labels


def test_vtk_labels_round_trip_preserves_metadata(tmp_path):
    volume = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    output = tmp_path / "labels.vtk"

    write_vtk_labels(volume, (0.5, 0.75, 1.25), output, dataset_name="test")
    restored, spacing, origin = read_vtk(output)

    np.testing.assert_array_equal(restored, volume)
    assert spacing == (0.5, 0.75, 1.25)
    assert origin == (0.0, 0.0, 0.0)


def test_read_vtk_accepts_bytes(tmp_path):
    volume = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    output = tmp_path / "labels.vtk"
    write_vtk_labels(volume, (1.0, 1.0, 1.0), output)

    restored, _, _ = read_vtk(output.read_bytes())
    np.testing.assert_array_equal(restored, volume)
