---
doc_id: reader-releases
surface: maintainer-guide
owner: reader-maintainers
last_verified: 2026-10-06
summary: Immutable Reader release tags, PyPI identity, qualification and downstream pins.
---

# Releases and downstream pins

Reader uses one version in `pyproject.toml`. The installed `reader --version`
and `reader_workbench.__version__` read distribution metadata. The PyPI
distribution is `reader-workbench`; Python imports use `reader_workbench` and
the command remains `reader`.

An immutable annotated tag `v<project.version>` identifies each reviewed source
release. Use patch versions for compatible corrections, minor versions for
compatible additions, and major versions for incompatible supported interfaces.
Use a prerelease suffix such as `rc1` when qualification is incomplete. Never
move a published tag or replace a published version's files. Config and record
schemas retain their own explicit versions.

## Prepare and qualify

1. Start from a clean committed `main`; check both GitHub releases and PyPI for
   an unused version. Update only `project.version`, refresh `uv.lock`, and
   describe affected interfaces and migrations in the release notes.
2. Pass the [change gate](../repo-change-gate.md), the portable suite and the
   installed-wheel checks in `Checks`. Check that the exact main commit has a
   successful main-push `Checks` run; a successful earlier commit is insufficient.
3. Build wheel and source distribution with `uv build --no-sources`. Inspect
   their contents for experiment data, local paths and unintended files. Verify
   metadata with `uv run --locked twine check dist/*` and test a fresh wheel
   installation outside the checkout with ordinary dependency resolution.
4. Record the source commit, lock and distribution hashes. The source distribution
   includes [CITATION.cff](../../CITATION.cff); the release evidence adds its exact
   version. Numerical agreement for a downstream paper is a separate acceptance
   check, not a consequence of the package version.

## Publish

Configure a PyPI pending Trusted Publisher for project `reader-workbench`, owner
`e-south`, repository `reader`, workflow `release.yaml`, environment `pypi`.
The GitHub environment admits release tags `v*`. Authenticate on PyPI directly;
do not store an API token or two-factor code in the repository.

After qualification, create the annotated version tag at the reviewed main
commit and publish its GitHub release. The existing `Release` workflow rejects a
tag/version mismatch, a commit outside main, or absent/failed latest main-push
checks. It builds and smoke-tests distributions in a job without publishing
credentials; only the subsequent `pypi` job can request the publishing identity.
PyPI creates the project on the pending publisher's first successful use.

Retain the distributions and `release-evidence` workflow artifacts on the
GitHub release. The latter contains the source identity, successful CI run,
artifact hashes and versioned citation. Workflow artifact retention alone is
temporary. The build also retains a hash-locked build-tool resolution, separate
from `uv.lock`, so its build environment can be repeated. Verify the published
PyPI hashes against that receipt and perform a
fresh version-based installation before announcing availability. A failed
upload is not a release; retry only the same qualified artifacts, and never
overwrite an already published file.

## Paper companions

Keep scientific inputs and figure recipes in their companion, and reusable
measurement operations in Reader. Companions pin a qualified immutable commit
or released artifact and retain hashes plus their environment lock. A new tool
release does not automatically advance a paper's pin: compare phenotype values,
qualification flags and downstream requirements before adopting it. A DOI is
an additional archival identifier, not a substitute for the executable version.

See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and the [namespace migration](./package_namespace_migration.md).

### Package-page images

The README is also the PyPI description. Keep its links absolute and use a
PNG banner at an immutable source commit or the matching `v<version>` tag.
Never use a relative asset path or a mutable branch for a release image. Keep
published tags and their assets; changing a new banner must not change older
release pages. When incrementing the version, update a version-bound README
image URL in the same change. The package tests enforce this relationship.

Before publishing, fetch the banner URL after the tag exists and compare its
bytes with the source PNG. Check the rendered PyPI page after upload. The PNG is
a package-page export; its editable SVG remains the artwork source.
