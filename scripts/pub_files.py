#!/usr/bin/env python3
"""List the files `dart pub publish` would upload for a package, one path per line.

pub decides what to publish from .gitignore and .pubignore files, which is easy to get wrong
(a git-ignored folder is silently left out). `dart pub publish --dry-run` prints the file set
as a tree; this turns that tree back into paths so a build can be tried from exactly that set.

usage: scripts/pub_files.py <package dir>
"""
import re
import subprocess
import sys

pkg = sys.argv[1]
out = subprocess.run(["dart", "pub", "publish", "--dry-run", "-C", pkg],
                     capture_output=True, text=True).stdout
entry = re.compile(r"^((?:│   |    )*)(?:├── |└── )(.+?)(?: \((?:<)?\d+ [KMG]?B\))?$")
stack, started = [], False
for line in out.splitlines():
    m = entry.match(line)
    if not m:
        if started:
            break          # the tree is one block; the first non-tree line ends it
        continue
    started = True
    depth = len(m.group(1)) // 4
    del stack[depth:]
    stack.append(m.group(2))
    if line.rstrip().endswith("B)"):   # files carry a size, folders do not
        print("/".join(stack))
