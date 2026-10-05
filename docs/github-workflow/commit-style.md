# Commit Style

**VisionX CXR-CAD team convention**

## Message Format

```text
type: short description
```

Use lowercase types and an action verb: `add`, `fix`, `update`, or `remove`. Keep the first line short (aim for 72 characters or fewer), with no ending period.

| Type | Use for |
| --- | --- |
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `docs` | Documentation changes |
| `test` | Adding or updating tests |
| `refactor` | Restructuring code without changing behavior |
| `chore` | Repository setup, dependencies, or maintenance |

## Examples

```text
feat: add patient-wise data split
fix: correct image normalization
docs: add GitHub workflow guides
test: check for patient overlap between splits
refactor: extract preprocessing into a shared function
chore: configure pull request checks
```

Issue titles may use prefixes such as `[INFRA]`; commit messages use the types above. The Issue number belongs in the branch name, and the PR description includes `Closes #N`.

## What to Commit

- Keep each commit focused on one logical change.
- Review the changed files before committing; exclude unrelated changes, datasets, patient information, credentials, and large model files.
- Avoid vague messages such as `updates`, `stuff`, or `final version`.
- If the reason is not obvious, add a blank line after the title and briefly explain why the change was needed.

## Quick Reference

**Review changes → Group one logical change → Write `type: action + description` → Commit → Push your task branch**
