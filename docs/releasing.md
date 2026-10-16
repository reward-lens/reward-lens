# Releasing reward-lens

A version is ready to publish only after source discovery and review, privacy
and dependency checks, the declared release matrix, package installation, and
the required pull-request and main checks pass for its exact source.

1. Prepare a focused release branch from current `origin/main`. Synchronize
   package/runtime/frontend metadata and lockfiles. Keep the changelog marked
   unreleased until publication succeeds. Review every outgoing commit and
   close required source and fixture dependencies.
2. Merge the reviewed pull request after current-head CI succeeds. Verify the
   resulting main SHA and the successful main-push `ci.yml` run. Record its tree,
   run ID, immutable `release-dist-3.1.0` artifact, filenames, and SHA-256 hashes.
3. Check GitHub tags/releases and PyPI again. An existing version with different
   source or artifact bytes blocks publication. Never overwrite or move it.
4. With the main-only protected `pypi` environment configured, dispatch
   `publish.yml` with `publish=false`, the exact release SHA, its successful main
   CI run ID, and the successful CI run ID for the current main publisher
   controller. The workflow verifies both complete main-push CI matrices,
   checks the frozen artifact's metadata with the pinned publisher image, then
   verifies OIDC trust by minting and discarding a short-lived token without
   uploading. A failure blocks tagging or upload.
5. Create the annotated `v3.1.0` tag at the exact release SHA on public main and
   push only that tag. The tag itself does not publish. Verify its peeled commit.
6. Dispatch `publish.yml` on main with the same exact release SHA and release
   CI run ID, the successful current-main publisher-controller CI run ID, and
   `publish=true`. The publisher verifies the immutable tag, version, and
   release-commit ancestry beneath the current controller commit. It downloads
   only the original release CI run's validated wheel and sdist, checks their
   manifest and metadata with the same pinned publisher image, and uploads
   those bytes through PyPI Trusted Publishing. It must not rebuild or skip
   files that already exist.
7. Observe publication completion. Compare public PyPI wheel and sdist hashes,
   then install `reward-lens==3.1.0` from public PyPI in a fresh environment.
   Verify package origin, metadata/runtime/CLI versions, dependency consistency,
   doctor, an offline audit/report, and signed export/import. The example's
   unresolved audit correctly exits 2.
8. Create one final GitHub Release for the verified existing tag. Attach the
   validated wheel, sdist, checksum manifest, and accurate notes. Update the
   changelog with the actual publication date. Update the existing official
   documentation host only after production package publication succeeds.
9. Verify live documentation, download links, release assets, source tag/tree,
   deployment commits, and browser behavior. Record exact receipts.

The production workflow needs repository contents/actions read permissions and
job-scoped OIDC permission for the upload. PyPI must recognize the official
repository and `publish.yml` Trusted Publisher identity. Respect repository and
production-environment approvals. Missing permissions are blockers.

The release artifact and the workflow that publishes it may have different main
commits. The frozen release SHA, original release CI run, tag, and artifact
hashes remain fixed. A later publisher workflow repair goes through its own
reviewed pull request and complete main CI run. Dispatch the repaired workflow
twice from that current main commit: first with `publish=false` for protected
OIDC preflight, then with `publish=true` after checking PyPI is still absent.
Supply both original release CI and current publisher-controller CI run IDs on
each dispatch. Never move the existing tag or substitute a later CI artifact
to make a controller repair appear to be a new release build.

For an uncertain upload, query PyPI and compare every filename/hash before
retrying. Resume only genuinely missing files through supported tooling. Do not
replace assets, delete/recreate a version, retag, or invent another version to
hide a collision. If publication is partial, report exactly what became public
and stop unsafe downstream promotion.
