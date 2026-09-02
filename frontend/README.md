# frontend

One workspace package, three Vite entries (D-68): `report/` (the offline single-file report, built
with vite-plugin-singlefile, one classic script, no workers, no modules; P-RENDER, wave 1),
`workbench/` (the served local workbench; P-WB-1 and P-WB-2, wave 5) and `shared/` (the one React 19
component tree the three renderings share; P-WB-1 owns it in wave 5, and P-RENDER seeds what the
report needs in wave 1 under `report/`). The site imports an island from the same tree.

The root `package.json` and Vite configuration are granted to P-RENDER in wave 1 and to P-WB-1 in
wave 5; nobody else edits them. Node is a development dependency only: the built bundle ships in the
wheel and no end user needs Node (D-58). Never symlink `node_modules`.
