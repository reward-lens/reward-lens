# Recoverable development source

These full source files preserve unfinished or historical alternatives. The maintained implementation is under src/. Snapshot code is outside package discovery, default test collection, and supported imports. It is not an integrated feature.

| Snapshot | Full files | Intended recovery |
| --- | ---: | --- |
| incomplete-packets/P-AUDIT-1 | 2 | Review and merge the selected files at the relative paths in its SNAPSHOT.json; current APIs may differ. |
| incomplete-packets/P-TRACE-1 | 14 | Review and merge the selected files at the relative paths in its SNAPSHOT.json; current APIs may differ. |
| incomplete-packets/P2-CAPTURE | 7 | Review and merge the selected files at the relative paths in its SNAPSHOT.json; current APIs may differ. |
| legacy/campaign | 21 | Review and merge the selected files at the relative paths in its SNAPSHOT.json; current APIs may differ. |
| legacy/experiment-run | 3 | Review and merge the selected files at the relative paths in its SNAPSHOT.json; current APIs may differ. |
| legacy/original | 2 | Review and merge the selected files at the relative paths in its SNAPSHOT.json; current APIs may differ. |

Run python tools/reconstruct_snapshot.py to verify every selected file against the public manifest. To recover one group into a new directory, use python tools/reconstruct_snapshot.py --group GROUP --out recovered-source. The destination must not exist. This copies source without importing it or accessing a private checkout.

The manifest identifies the original logical source, original hash, exported hash, intended relative destination, and known status. This recovers the selected file contents; it does not recreate a complete historical environment or unavailable evidence. Tests in an unsupported snapshot must be reviewed before execution.

No patch depends on an inaccessible private Git object. Private operational material, credentials, and excluded datasets are not backed up by these snapshots.
