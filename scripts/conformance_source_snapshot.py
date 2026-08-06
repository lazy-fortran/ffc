#!/usr/bin/env python3
"""Snapshot one Fortran source and its recursive INCLUDE closure."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
INCLUDE_LINE = re.compile(
    r"^\s*include\s*(['\"])(.+?)\1\s*(?:!.*)?$", re.IGNORECASE
)


class SnapshotError(Exception):
    """The requested closure cannot be represented hermetically."""


class UnhermeticInclude(SnapshotError):
    """An INCLUDE target cannot be reproduced inside the snapshot."""

    def __init__(self, source: Path, name: str) -> None:
        self.source = source
        self.name = name
        super().__init__(f"unhermetic Fortran INCLUDE at {source}: {name}")


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def decoded_lines(data: bytes, included: bool) -> list[str]:
    if included:
        if data.startswith(b"\x00\x00\xfe\xff"):
            return data.decode("utf-32").splitlines()
        if data.startswith(b"\xff\xfe\x00\x00"):
            return data.decode("utf-32").splitlines()
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            return data.decode("utf-16").splitlines()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.decode("latin-1").splitlines()


def include_names(data: bytes, included: bool) -> list[str]:
    result: list[str] = []
    for line in decoded_lines(data, included):
        match = INCLUDE_LINE.fullmatch(line)
        if match is None:
            continue
        name = match.group(2)
        if name:
            result.append(name)
    return result


class Snapshot:
    def __init__(
        self,
        suite_root: Path,
        destination: Path,
        manifest: Path,
        include_dirs: list[Path],
    ) -> None:
        self.suite_root = lexical_absolute(suite_root)
        self.destination = lexical_absolute(destination)
        self.manifest = lexical_absolute(manifest)
        self.include_dirs = [lexical_absolute(path) for path in include_dirs]
        self.visiting: list[str] = []
        self.completed: set[tuple[str, str]] = set()
        self.entries: set[str] = set()
        if self.manifest.exists():
            self.entries.update(self.manifest.read_text(encoding="utf-8").splitlines())

    def canonical(self, path: Path) -> tuple[str, Path]:
        path = lexical_absolute(path)
        relative = relative_to(path, self.suite_root)
        if relative is not None:
            return f"suite:{relative.as_posix()}", self.destination / "suite" / relative
        for index, root in enumerate(self.include_dirs):
            relative = relative_to(path, root)
            if relative is not None:
                label = f"include-{index:03d}:{relative.as_posix()}"
                return label, self.destination / f"include-{index:03d}" / relative
        raise SnapshotError(f"source escapes declared closure roots: {path}")

    def snapshot_bytes(self, original: Path, target: Path) -> bytes:
        if target.exists():
            return target.read_bytes()
        try:
            data = original.read_bytes()
        except OSError as error:
            raise SnapshotError(f"cannot read source dependency {original}: {error}") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return data

    def resolve_include(
        self, name: str, source: Path, root_dir: Path
    ) -> Path | None:
        include_path = Path(name)
        if include_path.is_absolute():
            raise UnhermeticInclude(source, name)
        candidates = [source.parent / include_path, root_dir / include_path]
        candidates.extend(directory / include_path for directory in self.include_dirs)
        for candidate in candidates:
            candidate = lexical_absolute(candidate)
            if candidate.is_file():
                try:
                    self.canonical(candidate)
                except SnapshotError as error:
                    raise UnhermeticInclude(source, name) from error
                return candidate
        return None

    def add(self, path: Path, root_dir: Path, included: bool = False) -> Path:
        original = lexical_absolute(path)
        canonical, target = self.canonical(original)
        key = (canonical, lexical_absolute(root_dir).as_posix())
        data = self.snapshot_bytes(original, target)
        digest = hashlib.sha256(data).hexdigest()
        self.entries.add(f"{canonical}\t{digest}")
        if key in self.completed:
            return target
        if canonical in self.visiting:
            return target
        self.visiting.append(canonical)
        try:
            for name in include_names(data, included):
                resolved = self.resolve_include(name, original, root_dir)
                if resolved is None:
                    missing = f"missing:{canonical}:{name}"
                    self.entries.add(f"{missing}\t{EMPTY_SHA256}")
                    continue
                self.add(resolved, root_dir, included=True)
        finally:
            self.visiting.pop()
        self.completed.add(key)
        return target

    def finish(self) -> None:
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(f"{entry}\n" for entry in sorted(self.entries))
        self.manifest.write_text(text, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--suite-root", type=Path, required=True)
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--status", type=Path, required=True)
    result.add_argument("--include-dir", action="append", type=Path, default=[])
    result.add_argument("source", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        snapshot = Snapshot(
            args.suite_root, args.destination, args.manifest, args.include_dir
        )
        source = lexical_absolute(args.source)
        canonical, target = snapshot.canonical(source)
        try:
            snapshot.add(source, source.parent)
        except UnhermeticInclude as error:
            marker = f"unhermetic:{canonical}:{error.name}"
            snapshot.entries.add(f"{marker}\t{EMPTY_SHA256}")
            args.status.write_text(f"{error}\n", encoding="utf-8")
        else:
            if not args.status.exists():
                args.status.write_text("", encoding="utf-8")
        for index, include_dir in enumerate(snapshot.include_dirs):
            (snapshot.destination / f"include-{index:03d}").mkdir(
                parents=True, exist_ok=True
            )
        snapshot.finish()
        print(target)
        return 0
    except (OSError, UnicodeError, SnapshotError) as error:
        print(f"ERROR: cannot snapshot conformance closure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
