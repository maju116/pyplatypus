<!--
Shared in shape with platypus/.github/PULL_REQUEST_TEMPLATE.md: the same three headings, each
side listing its own checks. Nothing here is a gate you must pass before opening a pull
request - a draft with half of it done is welcome, and CI runs the same checks anyway.

The one that is not obvious is the last. A test that cannot fail reads as proof and is none,
so the ask is to break the thing it guards and watch it go red. It takes a minute and it has
caught more than it has cost.
-->

## What this changes

## Closes

<!-- `Closes #123`, or say which issue it is part of. -->

## How it was checked

- [ ] `uv run pytest -q`
- [ ] `uv run --only-group lint ruff check pyplatypus tests examples tools`
- [ ] `uv run --only-group lint ruff format --check pyplatypus tests examples tools`
- [ ] `uv lock --check`
- [ ] every new test was mutated to prove it can fail
- [ ] `CHANGELOG.md`, with the measurement if the change rests on one
- [ ] documentation updated, if an exported name, a specification field or a number in prose changed

<!--
A release also bumps the version in four places - pyproject.toml, pyplatypus/__init__.py,
CITATION.cff and uv.lock - in the same pull request as the change, never in one of its own.
Three of the four are held together by tests/test_version.py and the fourth by `uv lock
--check`, so you will hear about it if one is missed.

If this adds a specification field, its `description=` is the documentation: the engine's
configuration reference is generated from the schema. The R package does not generate
anything from it, so a field documented only here is documented for half the users.
-->
