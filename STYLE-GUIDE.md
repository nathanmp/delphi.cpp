# delphi.cpp Style Guide

delphi.cpp follows llama.cpp style unless there is a specific reason to differ. Match the surrounding code before introducing a new pattern.

## Code and comments

* Use ASCII. Avoid em dashes, Unicode arrows, multiplication signs, ellipses, and similar characters. Use `-`, `->`, `x`, `...`, etc.
* Do not hard-wrap comments or prose to a fixed column width.
* Do not split a sentence across lines only to satisfy a line-length target.
* Keep comments concise, usually 1-2 lines.
* Use ASD-STE100 Simplified Technical English. Prefer short, direct, unambiguous wording.
* Comment non-obvious reasons, invariants, constraints, or behavior. Do not narrate code that is already clear.
* Comments must describe the code itself, not the task or conversation that produced it.
* Write the code first. Add comments afterward only where they are needed.
* Prefer existing llama.cpp infrastructure over new abstractions or subsystems.
* Avoid invasive rewrites when a smaller change can accomplish the same goal.
* Reuse existing test infrastructure where practical. Do not create new test files for trivial changes.

### Comment examples

```cpp
// GOOD: explains a non-obvious invariant

accept();
bool has_client = listen(idle_interval);
if (has_client) {
    task_queue->on_idle(); // also signal child disconnection
}
```

```cpp
// BAD: restates the implementation in prose

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD: generic and useful to a future reader

// reset here, as we will release the slot below
n_tokens = 0;
release();
```

```cpp
// BAD: refers to task context rather than the code

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

Do not add commentary merely because code was copied or adapted from another location when the surrounding code is already self-explanatory.

## Commit messages

Keep commit messages concise and specific. Prefer the style used by llama.cpp.

Good:

```text
llama : fix KV being cleared during context shift

Assisted-by: Claude Sonnet
```

Bad:

```text
This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.
anything
Co-authored-by: Claude Sonnet
```

Use `Assisted-by: <assistant name>` when AI meaningfully contributed to a commit. Do not use AI tools as `Co-authored-by:` authors.

