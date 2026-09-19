# Goals
This is delphi.cpp, a fork of llama.cpp with experimental features primarily focused on profiling / quantizing MoE models and improving MoE speed.

Prefer optimizations applicable to MoE models, especially hybrid CPU-routed-expert/GPU-dense inference; especially prefer optimizations that generalize across MoE architectures.
Dense-only optimization work is low priority unless it also benefits MoEs or the task explicitly asks for it.
This is a RAM-heavy inference machine; using host RAM for routed experts is intentional and mandatory, not an undesirable fallback to eliminate.

# Read first

Before substantial work:
1. Inspect the relevant existing code and follow its patterns.
2. Check existing project documentation before proposing a design that may already have been tried.

## Development rules

* Preserve upstream behavior outside the intended scope of a change.
* Keep the delta from upstream as small and isolated as practical.
* Prefer thin hooks into upstream files and delphi.cpp-owned implementation code over widespread invasive edits.
* Do not rename upstream APIs, files, or internal components merely for delphi.cpp branding.
* Avoid public API changes unless necessary.
* Keep model-specific, backend-specific, and server-specific behavior separate when practical.
* Experimental behavior should have an easy A/B switch when practical.
* Reuse existing llama.cpp infrastructure before adding another abstraction or subsystem.
* Before implementing a substantial change, identify the files and dependencies it should require. If the scope expands materially, reassess the design instead of continuing a sprawling implementation.

For ports:

* understand the source implementation before adapting it;
* port only the pieces required for current upstream;
* do not blindly cherry-pick a reference fork when a smaller current-upstream adaptation is possible;
* record source repository/PR/commit provenance, any prior art that you referenced, and important deviations in the worklog. (If there is feature documentation, add it to the documentation too.)	
* if copyright headers are included, ALWAYS copy the headers.

## Style

* Match surrounding llama.cpp style.
* Use ASCII in source/comments: `->`, `x`, `...`, `-`.
* Do not hard-wrap comments or prose to an arbitrary width.
* Use ASD-STE100 Simplified Technical English.
* Comments should explain non-obvious intent, invariants, constraints, or important divergence from upstream.
* Do not narrate obvious code or repeatedly teach basic C++, ggml, GPU, or ML concepts in source comments.
* If an explanation becomes tutorial-sized, put it in documentation instead.
* Prefer clear and maintainable code over clever code.

## Commits

* Keep commits logical and reviewable.
* Commit frequently enough that meaningful implementation states can be identified and tested.
* Keep functional changes, unrelated cleanup, and unrelated refactoring separate.
* Prefer useful commit bodies for non-trivial changes: explain the important design choice, invariant, limitation, or reason for divergence from upstream.
* Do not turn commit messages into implementation transcripts.
* Do not create standalone "plan update" or "status update" commits when the information belongs with the feature change.
* Use `Assisted-by:` rather than `Co-authored-by:` when AI meaningfully assisted a commit.
* For a multi-paragraph commit message, use one argument per paragraph: `git commit -m "subject" -m "body" -m "Assisted-by: Codex"`. Do not put `\n` escapes in a `-m` argument - Git records them literally.

## Public documentation

Public-facing documentation under `docs/delphi/` is for work intended to become part of the maintained project documentation.

Do not create or substantially update public feature documentation merely because experimental work was performed on a branch. Create or update it when explicitly requested or when preparing accepted work for integration.

Public feature documentation should be concise and contain what a future user or maintainer needs, such as:

* purpose and scope;
* important implementation or design decisions;
* user-visible flags, environment variables, or commands;
* important correctness, quality, and performance validation;
* known limitations and useful follow-up work;
* provenance for ports.

For multi-phase work on an existing documented feature, update the existing document rather than creating one document per phase.

Do not encode branch names in public documentation filenames.

Private development records, exploratory results, rejected approaches, deployment details, and work-in-progress notes do not belong in public feature documentation.
