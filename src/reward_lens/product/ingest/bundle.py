"""ZIP central-directory checks, deliberately before opening any archive member."""

from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from reward_lens.contracts import UsageError

__all__ = ["BundleMember", "BundleReport", "validate_bundle"]

MAX_MEMBERS = 1_000
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100


@dataclass(frozen=True)
class BundleMember:
    """A member as declared by ZIP metadata, not as interpreted archive content."""

    name: str
    size: int
    compressed_size: int
    crc32: str


@dataclass(frozen=True)
class BundleReport:
    path: Path
    members: tuple[BundleMember, ...]
    total_size: int


def _unsafe(member: str, rule: str) -> UsageError:
    return UsageError(
        code="RL0631",
        message=f"bundle member {member!r} is unsafe: {rule}",
        remediation="remove or rebuild that member, then run reward-lens import again",
        context={"member": member, "rule": rule},
    )


def _path_is_safe(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def validate_bundle(path: Path) -> BundleReport:
    """Check all central-directory metadata before any ``ZipFile.read``/``open`` call.

    The CRC is the archive's declared per-member digest.  Content is only read after this report
    returns; Python's ZIP reader validates that declared CRC as it streams a selected member.
    """
    source = Path(path)
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as failure:
        raise _unsafe(str(source), "not a readable ZIP archive") from failure
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise _unsafe("<archive>", f"member count exceeds {MAX_MEMBERS}")
        seen: set[str] = set()
        total = 0
        members: list[BundleMember] = []
        for info in infos:
            name = info.filename
            if not _path_is_safe(name):
                raise _unsafe(
                    name, "path is absolute, traverses parent, or uses a non-portable separator"
                )
            if name in seen:
                raise _unsafe(name, "duplicate member name")
            seen.add(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise _unsafe(name, "symlink members are not accepted")
            if info.file_size > MAX_MEMBER_BYTES:
                raise _unsafe(name, f"uncompressed size exceeds {MAX_MEMBER_BYTES} bytes")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise _unsafe(name, f"total uncompressed size exceeds {MAX_TOTAL_BYTES} bytes")
            if info.file_size and info.compress_size == 0:
                raise _unsafe(name, "compressed size is zero for non-empty content")
            if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                raise _unsafe(name, f"decompression ratio exceeds {MAX_COMPRESSION_RATIO}:1")
            members.append(
                BundleMember(
                    name=name,
                    size=info.file_size,
                    compressed_size=info.compress_size,
                    crc32=f"crc32:{info.CRC:08x}",
                )
            )
    return BundleReport(path=source, members=tuple(members), total_size=total)
