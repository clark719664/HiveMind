# Offline license-text fallbacks

The Windows build first collects license/notice files from installed distribution
metadata. If none were found, it may use an exact name-and-version entry from
`manifest.json`. It verifies the vendored file's SHA-256 before copying it. There
are no build-time downloads and no fallback to another package version. The build
manifest records each copied file's source and hash, reports remaining text gaps,
and preserves the original installed package's license metadata.

## tokenizers 0.22.2

The recorded Windows build at `docs/evidence/standalone/d36e654acefc/` has no
collected license text for this version. `tokenizers-0.22.2/LICENSE` is the exact
11,357-byte upstream root LICENSE, retrieved from the official Hugging Face
repository on 2026-10-20:

- [Release tag](https://github.com/huggingface/tokenizers/tree/v0.22.2)
- [Immutable source](https://github.com/huggingface/tokenizers/blob/f383101a26663708484cac0727792aad74f78234/LICENSE)
- [Python binding version](https://github.com/huggingface/tokenizers/blob/f383101a26663708484cac0727792aad74f78234/bindings/python/Cargo.toml)
- Upstream commit: `f383101a26663708484cac0727792aad74f78234`
- Upstream Git blob: `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`
- SHA-256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`

The downloaded bytes were checked against the blob ID in the official repository
tree. The local `.gitattributes` disables line-ending conversion for this file so
Windows checkouts preserve the verified bytes. A wheel that already supplies
license text takes precedence and does not use this fallback.

## Scope and updates

This repairs a known text-collection omission; it is not a legal clearance or a
complete inventory of native/Rust transitive dependencies. `files_collected`
means only that files were copied. The collector scans the build environment,
which can include packages excluded from the executable. Unmatched versions or
packages without text remain visible as `no_license_files_found` records and
`license_text_gaps` in new build manifests. CPython text is also reported missing
when its build-interpreter LICENSE.txt cannot be found.

When updating the pinned dependency, review its official release and relevant
license/notice files, vendor the exact text with new provenance, update the index,
and rerun `python -m pytest tests/test_standalone_license_collection.py -q`.
Never relabel an old fallback for a new version without verifying its source.

No license for HiveMind itself is selected or supplied here. Its missing project
license remains a separate unresolved issue. Historical ZIPs and manifests are
unchanged; an actual Windows rebuild is needed to include these files.
