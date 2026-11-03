"""Read an uploaded CSV/XLSX safely into an all-text DataFrame.

The file type is decided by content, not by the extension: an XLSX must be a ZIP holding a
workbook part, and a CSV must be NUL-free text. Macro-enabled workbooks, legacy XLS files,
ZIP bombs and oversized files are rejected rather than truncated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Literal

import pandas as pd

FileType = Literal["csv", "xlsx"]
ALLOWED_EXTENSIONS = {".csv": "csv", ".xlsx": "xlsx"}
ZIP_MAGIC = b"PK\x03\x04"
OLE_MAGIC = b"\xd0\xcf\x11\xe0"  # legacy .xls / other OLE containers
MAX_COLUMNS = 100
MAX_HEADER_LENGTH = 120
MAX_ZIP_ENTRIES = 2_000
MAX_UNCOMPRESSED_BYTES = 150 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


class FileRejected(ValueError):
    """The upload is unsafe, malformed or outside the configured limits. The message is safe to show."""

    def __init__(self, message: str, unsupported_type: bool = False) -> None:
        super().__init__(message)
        self.unsupported_type = unsupported_type


@dataclass
class ParsedFile:
    frame: pd.DataFrame
    file_type: FileType
    sha256: str
    size_bytes: int
    encoding: str | None = None
    delimiter: str | None = None
    sheet_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _extension_type(filename: str) -> FileType:
    ext = PurePath(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileRejected("Only .csv and .xlsx files are supported.", unsupported_type=True)
    return ALLOWED_EXTENSIONS[ext]  # type: ignore[return-value]


def _content_type(content: bytes) -> FileType:
    if content.startswith(ZIP_MAGIC):
        return "xlsx"
    if content.startswith(OLE_MAGIC):
        raise FileRejected("Legacy .xls (or another binary Office format) is not supported. Save the sheet as .xlsx "
                           "or .csv.", unsupported_type=True)
    if b"\x00" in content[:65536]:
        raise FileRejected("The file looks binary, not like a text CSV.")
    return "csv"


def _check_xlsx_container(content: bytes) -> list[str]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise FileRejected("The .xlsx file is corrupted (not a valid ZIP container).") from exc
    with zf:
        infos = zf.infolist()
        names = {i.filename for i in infos}
        if len(infos) > MAX_ZIP_ENTRIES:
            raise FileRejected("The .xlsx file contains too many parts.")
        if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
            raise FileRejected("The file is a ZIP archive but not an Excel workbook.")
        if any(n.lower().endswith("vbaproject.bin") for n in names):
            raise FileRejected("Macro-enabled workbooks are not accepted. Save the sheet as a plain .xlsx without macros.")
        total = sum(i.file_size for i in infos)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise FileRejected("The workbook expands to more data than the upload limit allows.")
        for i in infos:
            if i.compress_size and i.file_size / i.compress_size > MAX_COMPRESSION_RATIO:
                raise FileRejected("The workbook has an abnormal compression ratio and was rejected.")
    return sorted(names)


def _decode(content: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise FileRejected("The CSV is not valid UTF-8 or Windows-1252 text.")


def _read_csv(content: bytes, max_rows: int) -> tuple[pd.DataFrame, str, str]:
    text, encoding = _decode(content)
    sample = text[:20_000]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    try:
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, header=None, nrows=max_rows + 2,
                            skipinitialspace=True, on_bad_lines="error", engine="c", keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise FileRejected("The CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise FileRejected(f"The CSV is malformed: {str(exc).splitlines()[0][:200]}") from exc
    return _promote_header(frame), encoding, delimiter


def _promote_header(frame: pd.DataFrame) -> pd.DataFrame:
    """Use row 0 as the header ourselves: pandas would silently rename duplicate headers to 'a.1'."""
    if frame.empty:
        raise FileRejected("The file is empty.")
    header = ["" if pd.isna(v) else str(v) for v in frame.iloc[0]]
    body = frame.iloc[1:].replace({"": pd.NA})
    body.columns = header
    body.index = body.index - 1  # index 0 is the first data row (row 2 in a spreadsheet)
    return body


def _read_xlsx(content: bytes, max_rows: int) -> tuple[pd.DataFrame, list[str]]:
    _check_xlsx_container(content)
    try:
        book = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
        sheets = [str(s) for s in book.sheet_names]
        frame = book.parse(sheet_name=0, dtype=str, header=None, nrows=max_rows + 2)
    except Exception as exc:  # openpyxl raises many types for damaged workbooks
        raise FileRejected("The workbook could not be read. Re-save it in Excel as .xlsx and retry.") from exc
    return _promote_header(frame), sheets


def _check_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in frame.columns]
    if len(cols) < 2:
        raise FileRejected("The file needs at least two columns (date, region and units are required).")
    if len(cols) > MAX_COLUMNS:
        raise FileRejected(f"The file has {len(cols)} columns; the limit is {MAX_COLUMNS}.")
    for i, c in enumerate(cols, start=1):
        if not c:
            raise FileRejected(f"Column {i} has no header. Every column needs a name in the first row.")
        if len(c) > MAX_HEADER_LENGTH:
            raise FileRejected(f"Column {i} has a header longer than {MAX_HEADER_LENGTH} characters.")
    if len(set(cols)) != len(cols):
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        raise FileRejected(f"Duplicate column headers: {', '.join(dupes[:5])}.")
    frame.columns = cols
    return frame


def read_upload(content: bytes, filename: str, max_bytes: int, max_rows: int) -> ParsedFile:
    """Validate and parse an upload. Every cell is returned as stripped text (blank -> NaN)."""
    if not content:
        raise FileRejected("The file is empty.")
    if len(content) > max_bytes:
        raise FileRejected(f"The file is {len(content) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB.")
    claimed = _extension_type(filename)
    actual = _content_type(content)
    if claimed != actual:
        raise FileRejected(f"The file content is {actual.upper()} but the name ends in .{claimed}. Rename or re-export it.")

    parsed = ParsedFile(frame=pd.DataFrame(), file_type=actual, sha256=hashlib.sha256(content).hexdigest(),
                        size_bytes=len(content))
    if actual == "csv":
        frame, parsed.encoding, parsed.delimiter = _read_csv(content, max_rows)
        if parsed.encoding == "cp1252":
            parsed.warnings.append("The CSV was not UTF-8; it was read as Windows-1252 text.")
    else:
        frame, parsed.sheet_names = _read_xlsx(content, max_rows)
        if len(parsed.sheet_names) > 1:
            parsed.warnings.append(f"The workbook has {len(parsed.sheet_names)} sheets; only the first "
                                   f"('{parsed.sheet_names[0]}') was read.")

    frame = _check_columns(frame)
    # Keep the original index: index + 2 is the row number the user sees in Excel or a text editor.
    frame = frame.apply(lambda s: s.str.strip()).replace({"": pd.NA}).dropna(how="all")
    if frame.empty:
        raise FileRejected("The file has a header but no data rows.")
    if len(frame) > max_rows:
        raise FileRejected(f"The file has more than {max_rows:,} data rows, the limit for this deployment.")
    parsed.frame = frame
    return parsed
