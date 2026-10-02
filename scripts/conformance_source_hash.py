#!/usr/bin/env python3
"""Hash the existing path/NUL/blob-digest/newline compiler-source contract."""

import argparse
import hashlib
import os
import subprocess
import sys


def record(digest, path, content):
    digest.update(path + b"\0")
    digest.update(hashlib.sha256(content).hexdigest().encode("ascii") + b"\n")


def working_tree(root):
    paths = [b"fpm.toml"]
    for directory in (b"src", b"app"):
        for current, _, names in os.walk(os.path.join(root, directory)):
            for name in names:
                full = os.path.join(current, name)
                if os.path.isfile(full) and not os.path.islink(full):
                    paths.append(os.path.relpath(full, root))
    digest = hashlib.sha256()
    for path in sorted(paths):
        with open(os.path.join(root, path), "rb") as source:
            record(digest, path, source.read())
    return digest.hexdigest()


def revision_tree(root, revision):
    paths = subprocess.check_output(
        ["git", "-C", root, "ls-tree", "-r", "--name-only", "-z",
         revision, "--", "src", "app"]
    ).split(b"\0")
    paths = sorted([path for path in paths if path] + [b"fpm.toml"])
    digest = hashlib.sha256()
    with subprocess.Popen(
        ["git", "-C", root, "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    ) as blobs:
        for path in paths:
            blobs.stdin.write(os.fsencode(revision) + b":" + path + b"\n")
            blobs.stdin.flush()
            header = blobs.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError(f"missing source blob: {os.fsdecode(path)}")
            size = int(header[2])
            content = blobs.stdout.read(size)
            if len(content) != size or blobs.stdout.read(1) != b"\n":
                raise ValueError("incomplete git blob stream")
            record(digest, path, content)
        blobs.stdin.close()
        if blobs.wait() != 0:
            raise ValueError("git cat-file failed")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--revision")
    args = parser.parse_args()
    try:
        if args.revision is None:
            result = working_tree(os.fsencode(args.root))
        else:
            result = revision_tree(args.root, args.revision)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"ERROR: cannot hash compiler sources: {error}")
    print(result)


if __name__ == "__main__":
    main()
