#!/usr/bin/env python3
"""Assert the installed environment satisfies python/pyproject.toml.

The ``framework_final`` stage installs the sglang source with
``pip install --no-deps``, so the whole dependency set comes from the stage
underneath it. With ``FRAMEWORK_BASE_IMAGE`` pointed at a published image of an
earlier build's ``framework`` stage, that set is frozen at the commit the base
was built from, while the source being installed is at the commit being built.
A dependency added or version-bumped in between is then simply absent from the
image: the build succeeds and the server fails with an ImportError at startup.

This script closes that gap. It re-reads the source tree's declared
requirements, resolves the extras actually being installed (including sglang's
self-referencing ``all`` extra), and checks each one against the installed
distribution metadata. Anything unsatisfied is reported and fails the build.
"""

import argparse
import sys
import tomllib
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:  # pragma: no cover - fall back to pip's vendored copy
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name

from importlib import metadata as importlib_metadata

PREFIX = "[base-deps]"


def log(message: str) -> None:
    print(f"{PREFIX} {message}", flush=True)


def read_declared(pyproject: Path):
    """Return (project name, dependencies, optional-dependencies) from pyproject."""
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project", {})
    return (
        project.get("name", ""),
        list(project.get("dependencies", [])),
        {
            name: list(values)
            for name, values in project.get("optional-dependencies", {}).items()
        },
    )


def collect(
    self_name: str,
    dependencies: list,
    optional: dict,
    extras: list,
) -> list:
    """Expand the requirement list for the extras being installed.

    ``sglang[all]`` is defined as ``sglang[diffusion]``/``[http2]``/``[tracing]``,
    so a self-referencing requirement is followed into this same pyproject
    rather than looked up as an installed distribution.
    """
    canonical_self = canonicalize_name(self_name) if self_name else ""
    collected = []
    seen_extras = set()

    def add(specs, extra_context):
        for spec in specs:
            req = Requirement(spec)
            if req.marker is not None and not req.marker.evaluate(
                {"extra": extra_context}
            ):
                continue
            if canonical_self and canonicalize_name(req.name) == canonical_self:
                for nested in req.extras:
                    walk(nested)
                continue
            collected.append((req, extra_context))

    def walk(extra):
        key = canonicalize_name(extra)
        if key in seen_extras:
            return
        seen_extras.add(key)
        for name, specs in optional.items():
            if canonicalize_name(name) == key:
                add(specs, extra)
                return
        log(f"WARNING: pyproject declares no extra named {extra!r}; skipping it")

    add(dependencies, "")
    for extra in extras:
        walk(extra)
    return collected


def provided_extras(dist) -> set:
    return {
        canonicalize_name(name)
        for name in (dist.metadata.get_all("Provides-Extra") or [])
    }


def extra_requirements(dist, extra: str) -> list:
    """Requirements the installed distribution declares for one of its extras."""
    out = []
    for spec in dist.metadata.get_all("Requires-Dist") or []:
        req = Requirement(spec)
        if req.marker is None:
            continue
        if req.marker.evaluate({"extra": extra}):
            out.append(req)
    return out


def check_one(req: Requirement, origin: str, problems: list, depth: int) -> None:
    name = req.name
    try:
        dist = importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        problems.append(f"MISSING     {req} ({origin}) - not installed")
        return

    installed = dist.version
    # Prereleases are explicit in the source (flash-attn-4>=4.0.0b18), so a
    # prerelease in the environment must not read as "outside the specifier".
    if req.specifier and not req.specifier.contains(installed, prereleases=True):
        problems.append(
            f"INCOMPATIBLE {req} ({origin}) - installed {name}=={installed}"
        )
        return

    if depth <= 0:
        return

    available = provided_extras(dist)
    for extra in sorted(req.extras):
        if canonicalize_name(extra) not in available:
            problems.append(
                f"EXTRA       {req} ({origin}) - installed {name}=={installed} "
                f"declares no extra {extra!r}"
            )
            continue
        for nested in extra_requirements(dist, extra):
            check_one(nested, f"{name}[{extra}]", problems, depth - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("python/pyproject.toml"),
        help="the source tree's pyproject.toml (default: %(default)s)",
    )
    parser.add_argument(
        "--extras",
        default="",
        help="comma-separated extras being installed, i.e. the build's BUILD_TYPE",
    )
    args = parser.parse_args()

    if not args.pyproject.is_file():
        log(f"ERROR: no such file: {args.pyproject}")
        return 1

    self_name, dependencies, optional = read_declared(args.pyproject)
    extras = [part.strip() for part in args.extras.split(",") if part.strip()]
    requirements = collect(self_name, dependencies, optional, extras)

    log(
        f"{args.pyproject} declares {len(requirements)} requirement(s) "
        f"for extras {extras or ['<none>']}"
    )

    problems = []
    for req, extra_context in requirements:
        origin = f"{self_name}[{extra_context}]" if extra_context else self_name
        check_one(req, origin, problems, depth=1)

    if problems:
        log(f"{len(problems)} unsatisfied requirement(s):")
        for problem in problems:
            log(f"  {problem}")
        log(
            "The image installs the sglang source with --no-deps, so these come "
            "from the base image. Rebuild the base from this commit "
            "(--target framework) and repoint FRAMEWORK_BASE_IMAGE at it."
        )
        log("CHECK " "FAILED")
        return 1

    # Split the last word so the sentinel exists only in this program's output,
    # never in a build log's echo of the command that starts it.
    log("CHECK " "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
