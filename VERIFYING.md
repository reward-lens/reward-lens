# Verifying an export

An export keeps the canonical assay record unchanged. A bundle adds a manifest,
an offline DSSE signature and a rendered report derived from that record.

Verify the locally generated bundle before sharing it:

```console
reward-lens import --verify ./assay-bundle
```

The offline verification checks the manifest digest and the Ed25519 signature.
It identifies the signer. It does not establish that an outcome check was right.

reward-lens 3.1.0 signs its exports with Ed25519. Selecting the Sigstore
backend refuses with `RL0704`, even when the optional `sigstore` extra is
installed: external-identity signing is unavailable in offline mode.

For a Sigstore bundle created by a separate signer, verify its signed file
against an expected issuer and identity obtained independently from a trusted
source, rather than copied from the bundle being checked:

```console
sigstore verify identity --bundle attestation.sigstore.json --cert-identity USER --cert-oidc-issuer ISSUER SIGNED_FILE
```

The signed file is a required positional input to that Sigstore command. The
Ed25519 path above is the supported reproducible offline export path.
