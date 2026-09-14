# Transformer Teaching System Design

**Status:** Approved by the user on 2026-09-05

## Goal

Turn the existing WhiteBox Transformer demo into a detailed, interactive teaching system. The web application must present a newly authored 10-module course based on the supplied material, retain the original "lens" (透视) experience as the core interaction, and deploy as a static Vercel site. The original 30-lecture transcript is reference material, not the primary teaching text. The Python program remains a local, runnable white-box companion.

## Scope and Non-Goals

In scope:

- Ten rewritten, technically checked teaching modules based on both source documents.
- Supporting architecture explanations from `Transformer_Architecture_Master.md`.
- The existing 30 lecture screenshots.
- Interactive tensor, attention, masking, encoder-decoder, and autoregressive-generation lenses.
- A local Python CLI for course lookup and white-box demonstrations.
- Repository cleanup, a detailed README, GitHub push, and Vercel deployment.

Out of scope:

- A backend, accounts, cloud-synced progress, or a database.
- Claiming that the fixed toy weights are a trained translation model.
- React, a bundler, or additional runtime dependencies.
- Deploying the Python CLI to Vercel. Vercel hosts the static web application only.

## Product Model

The first screen is a study workspace, not a marketing page. It has a lecture directory, the current lesson, and a persistent lens/context area. Users can read sequentially, jump to a topic, expand an explanation, open a relevant lens, or enter the end-to-end lens mode.

Each of the 10 modules has five mandatory sections:

1. **Rewritten teaching text**: coherent, precise prose rather than a polished transcript.
2. **Inline concept cards**: definitions appear where a learner first needs them, with an example, purpose, and scope.
3. **Layered explanation**: intuition first, then formula, tensor dimensions/calculation, and common misconceptions.
4. **Lens experiment**: a focused entry to an existing deterministic white-box calculation or visualization. The full end-to-end mode remains available separately.
5. **Source reference**: the relevant original screenshots and transcript excerpts remain expandable evidence, not the default reading layer.

The original material remains preserved in the repository. The rewritten course is responsible for correctness, clarity, and its own pedagogical structure; it must not mechanically reproduce spoken wording or transcription errors.

## Lens Model

The educational model is a deterministic four-dimensional toy Transformer. Every view which exposes a result must expose the relevant formula, dimensions, and input/output tensors on demand. All UI labels must state that the values are teaching values, not learned production-model weights.

Reusable lenses include:

- Input embedding and positional encoding.
- Q/K/V projections, scaled scores, Softmax weights, and attention heatmaps.
- Multi-head splitting, concatenation, and output projection.
- Residual connections, LayerNorm, and FFN.
- Causal masks and zeroed future attention.
- Encoder memory and decoder cross-attention.
- Vocabulary logits, Softmax, greedy selection, and autoregressive progression.

Each lens supports back, next, expand calculation, and reset. The end-to-end lens also supports autoplay.

## Architecture

The project remains a zero-build static site plus a standard-library-first Python program.

```text
Transformer/
  README.md
  docs/
    course-source/
      Transformer_Full_Instructional_Course.md
      Transformer_Architecture_Master.md
    course-map.md
    superpowers/specs/
  web/
    index.html
    styles.css
    app.js
    data/
      lessons.js
      lens-data.js
      quizzes.js
    assets/slides/
    vercel.json
  python/
    main.py
    src/
    course_index.py
    requirements.txt
  scripts/
    verify-content.ps1
```

The web data modules are browser-ready JavaScript, while the CLI has a small Python course index. They intentionally do not share runtime files: JavaScript and Python would otherwise require a parser or generator that adds more moving parts than this project needs. `docs/course-map.md` and `scripts/verify-content.ps1` provide the single consistency contract: all 30 lesson IDs require their original text, explanation, screenshot, lens mapping, and quiz mapping.

The existing root-level duplicate web and Python directories are consolidated into the target layout. The actual white-box computational modules are retained rather than reimplemented.

## Deployment and Source Control

Git is initialized in `F:\LearningNotes\Transformer`. The remote is `https://github.com/JiangZi0721/Experiment-for-learning`. Before push, remote history must be fetched; unrelated remote files are preserved and reconciled rather than overwritten.

Vercel is linked with `web` as its root directory and deployed in production mode. `web/vercel.json` explicitly rewrites `/` to `index.html` and configures static asset caching. The deployed URL and any deployment-protection issue are recorded in `README.md`.

## Acceptance Checks

1. The web application contains 30 numbered lessons, with the four required sections for each.
2. All referenced screenshots and lens IDs exist.
3. The existing end-to-end lens, encoder lens, decoder lens, theory lens, tensors, heatmaps, mask, and autoregressive controls work.
4. A local static HTTP preview loads without JavaScript errors.
5. The Python CLI runs a non-interactive course lookup and an existing white-box flow.
6. `scripts/verify-content.ps1` fails if a lesson mapping, screenshot, or lens reference is missing.
7. The Vercel production deployment succeeds and the GitHub repository contains the final project.
8. `README.md` explains the technology choices, repository structure, local usage, deployment, issues found, and their solutions.

## Risks and Decisions

- The current root is not a Git repository, so push requires a clean initialization and remote reconciliation.
- The current web client is a compact hard-coded demo. Keeping it as one large source file would make 30 complete lessons unmaintainable; content is split into data modules, without introducing a framework.
- KaTeX currently loads from a CDN. The rebuilt site must remain legible if it fails to load; formulas keep a text fallback. Local vendoring is not necessary unless offline web use becomes a requirement.
- The existing Python computation is kept as the numerical authority for its CLI demonstrations. The browser uses intentionally fixed instructional values, which are clearly labelled.
