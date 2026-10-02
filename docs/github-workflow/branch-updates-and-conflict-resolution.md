# Updating Your Branch and Handling Conflicts

**VisionX CXR-CAD team guide**

Update your task branch when it needs recent changes from `main` or GitHub reports a conflict. Commit your current work first; check that you have no uncommitted changes with `git status`.

## Update Your Branch

Example for Issue #6:

```bash
git switch 6-verify-github-workflow
git fetch origin
git merge origin/main
```

If the merge succeeds, run the relevant checks and push:

```bash
git push origin 6-verify-github-workflow
```

Use **merge** to bring `main` into your branch. Do not rebase or force-push for this team workflow.

## If There Is a Conflict

1. Run `git status` to see the conflicted files.
2. Open each file. Resolve sections marked like this:

```text
<<<<<<< HEAD
Your task branch's version
=======
The version from main
>>>>>>> origin/main
```

3. Edit the section into the correct final result, keeping or combining the changes as needed. Remove the marker lines. Do not blindly accept one side; ask the other author if the intended result is unclear.
4. Save the files, then stage each resolved file and finish the merge:

```bash
git add path/to/resolved-file
git merge --continue
```

5. Check the final changes, run the relevant checks, and push your branch. If a PR is open, it updates automatically; ask your reviewer to recheck the changes.

To cancel an unfinished merge and try again later:

```bash
git merge --abort
```

## Quick Reference

**Commit current work → Fetch → Merge `origin/main` → Resolve conflicts → Stage + finish merge → Check → Push → Re-review**
