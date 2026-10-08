# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Teaching code (Middlesex University, CST2420) to demonstrate deep neural network functionality. The project is written in **TypeScript**.

## Commands

- `npm run build` — compile `src/` to `dist/` with `tsc`
- `npm run watch` — recompile on change
- `npm start` — run `dist/index.js` (build first)

No test runner or linter is set up yet (`npm test` is the npm placeholder and fails).

## TypeScript setup

- TypeScript is installed globally on this machine (`tsc` 5.8) and is deliberately **not** a project dependency; don't add it to `package.json`.
- ES modules (`"type": "module"`, `module: NodeNext`): relative imports in `.ts` files must use the `.js` extension, e.g. `import { x } from "./layer.js"`.
- `strict` plus `noUncheckedIndexedAccess`: indexing an array returns `T | undefined`, which matters for matrix/vector code.
- `@types/node` is pinned to v22 to match the installed Node.
- PNG decoding uses `pngjs` (pure JS, no native build); `ImageLoader` only supports PNG.

## Data

`data/` holds the scikit-learn 8x8 digits dataset (UCI optical digits subset, often used as a "mini MNIST"), not true 28x28 MNIST:

- `data/digits.csv.gz` — original source, downloaded from scikit-learn's GitHub (`sklearn/datasets/data/digits.csv.gz`, the same file `sklearn.datasets.load_digits()` reads). 1,797 rows, no header; each row is 64 pixel values (0–16, row-major 8x8) followed by the label (0–9).
- `data/images/<label>/<index>.png` — the same images unpacked as 8x8 grayscale PNGs, pixels scaled 0–16 → 0–255 (white digit on black, like MNIST). The label is the folder name; `<index>` is the zero-padded row number in the CSV.
- `data/labels.csv` — `index,filename,label`, with `filename` relative to `data/`.

The PNGs and `labels.csv` are derived from `digits.csv.gz`; regenerate them from it rather than editing by hand.

## Environment

Windows, PowerShell. Node v22 and npm 10 are installed. Python is not installed (only the Microsoft Store stubs), so the data files above were produced with PowerShell/.NET (`System.IO.Compression`, `System.Drawing`). The repo lives in OneDrive, so avoid generating very large numbers of files.

`.gitignore` is the Node.js template (covers `node_modules/`, `dist/`, etc.); it does not ignore `data/`.
