# delphi.cpp

Downstream llama.cpp fork focused on hardware-aware MoE inference, profiling, quantization, and autotuning.

## Critical repository boundaries

This `delphi.cpp` repository is the only writable source tree.

* `upstream` refers to official `ggml-org/llama.cpp`. Fetching from upstream is allowed. Never push to it or create issues, PRs, comments, branches, or other remote changes on its behalf.
* Repositories under `../references/` are strictly read-only reference material. Do not edit files, create commits or branches, push, or create issues/PRs/comments in those repositories.
* `origin` is the only source-code remote that may be written to, and only when the user explicitly requests a remote action.
* `../main-docs-snapshot/` is a generated read-only filesystem snapshot of public documentation from `main`. Never edit it. If it may be stale, verify information with `git show main:<path>`.
* `docs/delphi/private/` is a separate, local-only Git repository. It persists across outer-repository branch changes. Never add its contents to the public delphi.cpp repository or copy private information into public files unless explicitly requested.

Do not create additional delphi.cpp worktrees unless explicitly requested.

## Read before substantial work

Read `STYLE-GUIDE.md` before modifying code.

Also consult relevant existing project documentation before making design decisions:

* current branch documentation: `docs/delphi/`
* integrated `main` documentation: `../main-docs-snapshot/docs/delphi/`
* private roadmap, deployment information, decisions, and rejected approaches: `docs/delphi/private/`
* reference implementations: `../references/`

Do not read all project-memory files by default. Start with its index/current-state material and load topic-specific files as needed.

Previously rejected approaches are important. Do not repeat an experiment without first checking whether its failure and reason are already documented.

## Branch policy

* `main`: tested, known-good delphi.cpp
* `next`: integration branch for the next tested release
* `feat/*`: original delphi.cpp features
* `port/*`: genuine ports from another project or PR
* `fix/*`: delphi.cpp-specific fixes
* `exp/*`: disposable experiments not yet accepted for integration
* `upstream/master`: pristine upstream llama.cpp

Do not develop directly on `main` unless explicitly requested.

New feature work should normally branch from `next`.

An isolated port may branch from `upstream/master` when it has no delphi.cpp dependencies and keeping the upstream delta isolated makes maintenance easier.

## Development rules

* Preserve upstream behavior outside the intended scope of a change.
* Keep the delta from upstream as small and isolated as practical.
* Prefer thin hooks into upstream code and delphi.cpp-owned files for substantial new subsystems when practical.
* Do not rename upstream APIs, files, internal components, or identifiers only for delphi.cpp branding.
* Prefer small, understandable changes over broad rewrites.
* Reuse existing llama.cpp infrastructure before introducing a new abstraction or subsystem.
* Avoid public API changes unless necessary.
* Keep model-specific, backend-specific, and server-specific changes separate when practical.
* Experimental behavior should have an easy A/B switch when practical.
* Before a substantial implementation, inspect the relevant files and identify the expected scope. If the implementation starts spreading substantially beyond that scope, reassess the design instead of continuing a sprawling change.

For ports:

* Understand the source implementation before adapting it.
* Port only the pieces required for current upstream and delphi.cpp.
* Do not blindly cherry-pick an entire reference-fork commit when a smaller adaptation is possible.
* Preserve and document the original source repository and relevant PR/commit provenance.
* Record important deviations from the source implementation.

## Style reminder

`STYLE-GUIDE.md` contains the full rules. These points are easy to lose during long sessions and must remain in mind:

* Match surrounding llama.cpp style.
* Use ASCII only: `->`, `-`, `x`, `...`, etc.
* Use ASD-STE100 Simplified Technical English.
* Keep comments rare and concise, usually 1-2 lines.
* Explain non-obvious reasons, invariants, or constraints; do not narrate obvious code.
* Write code first and comments afterward.
* Never describe the user's task or conversation in a code comment.
* Do not hard-wrap comments or prose to a fixed width.
* Do not split sentences across source lines merely to satisfy a line-length target.
* Prefer existing infrastructure and simple implementations over unnecessary new abstractions.

Prefer 20 clear lines of documentation over 100 exhaustive lines. Brevity must not omit information required to understand, use, validate, or maintain the feature.

## Testing

Use the smallest test matrix that establishes correctness and measures the intended effect.

* Run relevant existing tests or backend diagnostics where available.
* Performance changes require an A/B comparison against an appropriate baseline.
* Test more than one representative model, shape, or workload when a change is intended to generalize.
* Use perplexity, KLD, or another quality test when a change can affect numerical behavior, quantization quality, or model output.
* Do not run expensive quality tests for changes that are demonstrably output-identical unless there is another reason.
* Record important regressions and failures, not only successful results.

When running models or benchmarks, consult `docs/delphi/private/DEPLOYMENT_INFO.md` for current local hardware, model, and invocation information. Do not duplicate that private information in tracked public files.

## Commits

Local commits are allowed as part of implementation work unless the user says otherwise.

* Keep commits logical and reviewable.
* A small feature may be one commit. A larger feature should be a short series of independently understandable commits.
* Do not create standalone "plan update", "status update", or trivial documentation commits when those changes belong with implementation work.
* Do not mix unrelated cleanup or refactoring with a functional change.
* Follow `STYLE-GUIDE.md` for commit-message style.
* Use `Assisted-by: <assistant name>` when AI meaningfully contributed. Do not use `Co-authored-by:` for AI assistance.
* Never push or perform another remote write unless the user explicitly requests it.

## Public feature documentation

Substantial feature, port, and accepted experimental branches should create or update one concise document under:

`docs/delphi/features/<feature>.md`

Keep the documentation with the implementation branch so it is merged with the feature.

Include, as relevant:

* purpose and scope
* important implementation or design decisions
* user-visible flags, environment variables, commands, or configuration
* important correctness and performance validation
* known limitations
* useful follow-up work
* source/provenance for ports

For a multi-phase feature, update the existing feature document instead of creating one file per phase.

Do not encode branch names in documentation filenames.

When the feature reaches `main`, its public documentation becomes the durable public record and will appear in the main documentation snapshot.

## Private project memory

Use `docs/delphi/private/` for information that must persist across outer-repository branches but should not be public, including:

* roadmap and current project status
* deployment-specific information
* design decisions
* exploratory benchmark notes
* rejected approaches and why they failed
* cross-branch conclusions that future sessions need to know

Do not duplicate the same document in public and private locations.

When work reveals a non-obvious failed approach or an important cross-branch conclusion, record it in the private project-memory repository so another branch or future session does not rediscover it.

If private documentation is modified, treat `docs/delphi/private/` as its own Git repository and commit its changes separately from delphi.cpp.
