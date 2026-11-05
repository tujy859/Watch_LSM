"""Lazy PulseDB v2 reader for official Info rows and 10-second segments.

The mapping follows pulselabteam/PulseDB Generate_Subsets.m at db0824f:
Subj_Name[:7] is the file name, the final 0/1 identifies the source, and
Subj_SegIDX is a MATLAB ONE-BASED POSITION in Subj_Wins. SegmentID/WinID
are not interchangeable with this position: quality filtering leaves gaps.
No full HDF5 reference directory or waveform collection is materialized.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import h5py
import numpy as np


INFO_GROUPS = {
    "Train_Info.mat": "Train_Subset",
    "CalBased_Test_Info.mat": "CalBased_Test_Subset",
    "CalFree_Test_Info.mat": "CalFree_Test_Subset",
    "AAMI_Test_Info.mat": "AAMI_Test_Subset",
    "AAMI_Cal_Info.mat": "AAMI_Cal_Subset",
}
SOURCE_FOLDERS = {"MIMIC": "PulseDB_MIMIC", "VitalDB": "PulseDB_Vital"}
UPSTREAM_COMMIT = "db0824f18d9a462458e46fe94c31283a93a5c0d5"


def _value(handle, dataset, position):
    """Read one MATLAB struct field, supporting row/column struct arrays."""
    if dataset.ndim != 2 or 1 not in dataset.shape:
        raise ValueError(f"Expected MATLAB struct vector: {dataset.name}")
    size = dataset.size
    if not 0 <= position < size:
        raise IndexError(f"Position {position} outside {dataset.name} length {size}")
    value = dataset[0, position] if dataset.shape[0] == 1 else dataset[position, 0]
    return np.asarray(handle[value][()] if isinstance(value, h5py.Reference) else value)


def _text(array):
    return "".join(chr(int(c)) for c in np.asarray(array).ravel(order="F"))


def _scalar(array):
    values = np.asarray(array).ravel()
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError("Expected one finite scalar")
    return float(values[0])


def subject_identity(subject_name):
    """Return source and physical-patient key; never use recording identity."""
    if not re.fullmatch(r"p\d{6}_[01]", subject_name):
        raise ValueError(f"Unexpected official subject name: {subject_name!r}")
    source = "MIMIC" if subject_name[-1] == "0" else "VitalDB"
    return source, f"{source}:{subject_name[:7]}"


@dataclass(frozen=True)
class PulseDBRow:
    info_file: str
    info_row_zero_based: int
    subject_name: str
    source: str
    patient_id: str
    segment_position_one_based: int
    sbp: float
    dbp: float

    def to_dict(self):
        return asdict(self)


class PulseDBIndex:
    """Read small, selected portions of an official Info file."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.name not in INFO_GROUPS:
            raise ValueError(f"Unsupported official Info file: {self.path.name}")
        self.handle = None

    def __enter__(self):
        self.handle = h5py.File(self.path, "r")
        try:
            self.group = self.handle[INFO_GROUPS[self.path.name]]
            self.length = self.group["Subj_Name"].size
        except Exception:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, *args):
        self.handle.close()
        self.handle = None

    def subject(self, position):
        return _text(_value(self.handle, self.group["Subj_Name"], position))

    def row(self, position):
        name = self.subject(position)
        source, patient = subject_identity(name)
        declared_source = _text(_value(self.handle, self.group["Source"], position))
        if source != declared_source:
            raise ValueError(f"Source mismatch for {name}: {declared_source}")
        raw_position = _scalar(_value(self.handle, self.group["Subj_SegIDX"], position))
        if raw_position < 1 or raw_position != int(raw_position):
            raise ValueError("Subj_SegIDX must be a positive one-based integer")
        return PulseDBRow(
            self.path.name, int(position), name, source, patient,
            int(raw_position),
            _scalar(_value(self.handle, self.group["Seg_SBP"], position)),
            _scalar(_value(self.handle, self.group["Seg_DBP"], position)),
        )


def segment_file(root, subject_name):
    source, _ = subject_identity(subject_name)
    return Path(root) / "Segment_Files" / SOURCE_FOLDERS[source] / f"{subject_name[:7]}.mat"


def read_segment(root, row, *, include_reference_signals=False):
    """Read one official PPG_F window and audit labels/identity against Info.

    ECG and ABP are opt-in audit/reference outputs, never regression inputs.
    PPG_F is the official filtered, normalized signal, not an invented clean
    PPG ground truth. The sampling frequency is 125 Hz and duration 10 sec.
    """
    path = segment_file(root, row.subject_name)
    source, patient = subject_identity(row.subject_name)
    if source != row.source or patient != row.patient_id:
        raise ValueError("Inconsistent row patient identity")
    if row.segment_position_one_based < 1:
        raise ValueError("Segment positions must be one-based")
    position = row.segment_position_one_based - 1
    with h5py.File(path, "r") as handle:
        group = handle["Subj_Wins"]
        subject = _text(_value(handle, group["SubjectID"], position))
        if subject != row.subject_name[:7]:
            raise ValueError(f"Waveform patient mismatch: {subject} != {row.subject_name[:7]}")
        sbp = _scalar(_value(handle, group["SegSBP"], position))
        dbp = _scalar(_value(handle, group["SegDBP"], position))
        if not np.allclose([sbp, dbp], [row.sbp, row.dbp], rtol=0, atol=1e-6):
            raise ValueError("Info labels do not match selected waveform labels")
        fields = ["PPG_F", "T"]
        if include_reference_signals:
            fields += ["ECG_F", "ABP_Raw"]
        arrays = {}
        for field in fields:
            array = np.asarray(_value(handle, group[field], position), dtype=np.float64).ravel()
            if array.size != 1250 or not np.isfinite(array).all():
                raise ValueError(f"Invalid {field}: expected 1250 finite samples")
            arrays[field] = array
        if not np.allclose(np.diff(arrays["T"]), 1 / 125, rtol=0, atol=1e-8):
            raise ValueError("Time vector does not advance at 125 Hz")
        segment_id = _scalar(_value(handle, group["SegmentID"], position))
    return {**arrays, "sbp": sbp, "dbp": dbp, "fs": 125,
            "segment_id": segment_id, "patient_id": patient,
            "file": path, "index": row.to_dict()}


def assert_patient_disjoint(split_rows):
    """Audit the selected subsets by source + physical patient file ID."""
    patients = {name: {row.patient_id for row in rows} for name, rows in split_rows.items()}
    names = list(patients)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            overlap = patients[name] & patients[other]
            if overlap:
                raise ValueError(f"Patient leakage between {name} and {other}: {sorted(overlap)}")
    return {name: sorted(value) for name, value in patients.items()}
