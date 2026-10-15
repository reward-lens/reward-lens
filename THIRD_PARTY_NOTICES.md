# Third-party notices

## Model Context Protocol Python SDK 2.2.0

`reward_lens._vendor.mcp2.site_packages` contains the unchanged `mcp/` package
and `mcp-2.2.0.dist-info/` payload from
[`mcp-2.2.0-py3-none-any.whl`](https://pypi.org/project/mcp/2.2.0/), published
by the Model Context Protocol project. The artifact SHA-256 is
`bde982589473a060ae145e3406e9a5333fe538c97229ba841f5a7f92be004f81`.
The package is used only by the isolated `reward-lens mcp` child runtime.
Its wheel identity, payload digest, and license identity are recorded in
`src/reward_lens/_vendor/mcp2/MCP2_PROVENANCE.json`. The unchanged vendored
`mcp-2.2.0.dist-info/RECORD` lists the individual payload file hashes.

```text
MIT License

Copyright (c) 2024 Anthropic, PBC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Verifiers source fixture

`src/reward_lens/_vendor/verifiers_writer/save_utils.py` is an unchanged source copy
from [verifiers at edafab857aabf830237068193150694cfae50c3b](https://github.com/willccbb/verifiers/blob/edafab857aabf830237068193150694cfae50c3b/verifiers/utils/save_utils.py).
The file's SHA-256 is `07cc179170e73d0b9587f58f7782d4865afdd235db89289ac21cdd5474d98820`.
It is executed in a child process to produce a machine-observed writer receipt.
The source provenance and MIT license are retained beside the file in
`src/reward_lens/_vendor/verifiers_writer/`.
