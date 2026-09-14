# Transformer Teaching System Implementation Plan

**Goal:** Turn the existing demo into a complete 30-lesson static Transformer teaching system with the core white-box lens preserved, a local Python companion, Vercel deployment, and GitHub publication.

**Architecture:** Browser-native modules in `web/data/` hold 30 complete lessons, quizzes, and fixed teaching tensors. The current Python numeric modules are retained under `python/src/`. An explicit course map and PowerShell verifier prevent omissions.

## Task 1: Consolidate Source Material

- [ ] Copy both Markdown files verbatim to `docs/course-source/`.
- [ ] Move the 30 source screenshots to `web/assets/slides/`.
- [ ] Create `docs/course-map.md` with exactly `L01` through `L30`; every row identifies one source heading, one screenshot, one lens ID, and one quiz ID.
- [ ] Add `.gitignore` for `__pycache__/`, `*.py[cod]`, `.vercel/`, and `.superpowers/`.
- [ ] Verify `Select-String` finds 30 lecture headings and `Get-ChildItem web/assets/slides` finds 30 images.

## Task 2: Create Complete Content Modules

- [ ] Create `web/data/lessons.js`; export 30 unique lessons with `id`, `title`, full `source.text`, `source.image`, `explanation` (`intuition`, `formula`, `dimensions`, `misconception`), `lensId`, and `quizId`.
- [ ] Transcribe each `## 🖼️ 第 N 讲` source section in full; no summary may replace the original text.
- [ ] Map supporting architecture content into supplemental explanations.
- [ ] Create `web/data/quizzes.js`; export one reasoned question for every `Q01`-`Q30`.
- [ ] Assert there are exactly 30 unique lessons and that all mandatory fields are populated.
- [ ] Verify using Node module import with `lessons.length === 30`.

## Task 3: Preserve and Improve the Lens

- [ ] Create `web/data/lens-data.js` from current fixed model values: embedding/PE, Q/K/V, score and attention matrices, heads, mask, encoder memory, cross-attention, logits, and sequence steps.
- [ ] Keep `previous`, `next`, `reset`, and `autoplay` as the full-flow state machine.
- [ ] Add focused lens mappings for scaling, masking, multi-head attention, cross-attention, output Softmax, and autoregressive generation.
- [ ] Render `教学玩具模型：数值为固定演示值，不是训练权重。` in every lens path.
- [ ] Assert all required lens IDs exist.

## Task 4: Implement the Static Study Workspace

- [ ] Rework `web/index.html`, `web/styles.css`, and `web/app.js` into a three-region first screen: lesson navigation, current lesson, and contextual lens.
- [ ] Render all four required lesson layers: original source + screenshot, layered explanation, lens entry, and reasoned lesson check.
- [ ] Escape authored text before narrow KaTeX rendering and preserve text fallback.
- [ ] Wire lesson selection, progressive detail, quiz answers, direct lens entry, full-flow reset, and full-flow autoplay.
- [ ] Keep semantic controls, visible focus, `aria-current`, mobile stacking under 900px, and matrix-only horizontal scrolling.
- [ ] Verify under `python -m http.server 8765 --directory web`: open L07, its lens, a quiz, full-flow reset, and autoplay.

## Task 5: Add the Python Companion

- [ ] Move existing numerical modules to `python/src/` without reimplementation.
- [ ] Add `python/course_index.py` for all 30 lesson titles, source references, and lens mappings.
- [ ] Add `python/main.py` options: `--list-lessons`, `--lesson`, `--lens {full,encoder,decoder,theory}`, and explicit `--menu`.
- [ ] Verify `python python/main.py --list-lessons` lists 30 lessons and `python python/main.py --lesson L07` completes without a prompt.

## Task 6: Verify and Document

- [ ] Create `scripts/verify-content.ps1` to fail for missing lesson IDs, slide files, quiz IDs, course-map entries, or required lenses.
- [ ] Create a detailed `README.md`: learning goal, toy-model honesty note, technologies, layout, local web/CLI startup, source maintenance, verification, Vercel, GitHub, problems and solutions.
- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts/verify-content.ps1`; expected result: 30 complete lessons.

## Task 7: Publish

- [ ] Add `web/vercel.json` with `/` rewrite to `/index.html` and immutable asset caching.
- [ ] Commit each self-contained stage after its verification.
- [ ] Add and fetch `https://github.com/JiangZi0721/Experiment-for-learning.git`; reconcile any remote history without force-push; push `main`.
- [ ] Run `vercel link --yes --project transformer-learning-system` and `vercel deploy --prod --yes` from `web/`, record the production URL in `README.md`, then commit and push.
