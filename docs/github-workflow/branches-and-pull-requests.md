# Branches and Pull Requests

**VisionX CXR-CAD team workflow**

1. **Choose an Issue.** Assign yourself and move its Project item to **In Progress**.
2. **Create a branch from the Issue.** In the Issue’s **Development** section, select **Create a branch**, branch from `main`, and include the Issue number: `12-patient-wise-split`. Fetch and switch to that branch locally before editing.
3. **Work, commit, and push.** Keep changes focused on the Issue. Use clear commit messages and push your branch to GitHub.
4. **Open a PR into `main`.** Briefly describe the change and how you checked it. Include `Closes #12` (replace `12` with your Issue number). Move the Project item to **In Review** and request a teammate’s review.
5. **Resolve review comments.** Make any changes, commit, and push to the same branch. Get **1 teammate approval** before merging; request another review if needed after changes.
6. **Merge and clean up.** Choose normal **Merge pull request**, then confirm. Do **not** squash or rebase. Check that the Issue closes, move the Project item to **Done**, and delete the merged branch on GitHub.
7. **Update local `main`.** Switch to `main` and pull the latest changes before starting another task.

## Small Example

Issue **#12: Implement patient-wise data split** → branch `12-patient-wise-split` → PR into `main` with `Closes #12` → teammate approval → **Merge pull request**.

After merging:

```bash
git switch main
git pull origin main
git branch -d 12-patient-wise-split
```

## Quick Reference

**Assign Issue → In Progress → Branch → Commit/push → PR + `Closes #N` → In Review → Resolve comments + 1 approval → Merge pull request → Issue closed + Done → Delete branch → Pull `main`**
