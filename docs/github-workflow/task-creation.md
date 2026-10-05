# Creating Tasks in GitHub Projects

**VisionX CXR-CAD team guide**

Use the [Task Template](task-template.md) when writing a new Issue.

1. **Check for an existing task.** Search the Project and repository Issues before creating a duplicate.
2. **Create a GitHub Issue.** In the VisionX CXR-CAD repository, select **Issues → New issue**. Use the available template, if any. Keep each task focused on one clear outcome.
3. **Write a clear title and description.** Use `[CATEGORY] Action + specific outcome`, such as `[INFRA] Verify GitHub workflow`. Use the team's existing category prefixes. Include what needs doing and a short completion checklist. After creating the Issue, add its exact branch name to the description: `<issue-number>-<short-task-description>` (lowercase, hyphens, no category prefix). For Issue #6: `6-verify-github-workflow`.
4. **Add it to the Project.** Select the team Project under **Projects** on the Issue. If needed, use **Add item** in the Project and paste the Issue URL.
5. **Set its details.** Assign the teammate responsible if agreed; otherwise leave it unassigned. Add relevant existing labels and fill in priority or other Project fields if the team uses them.
6. **Set its status.** Use the Project’s initial status (for example, **Todo** or **Backlog**). Move it to **In Progress** only when work begins.

If you created a draft item in the Project, use its item menu to **convert it to an Issue** in the VisionX CXR-CAD repository before starting development.

## Small Example

**Title:** [INFRA] Verify GitHub workflow

**Issue:** #6

**Branch:** `6-verify-github-workflow`

**Description:** Confirm the team can complete the Issue → branch → PR → review → merge workflow.

**Done when:**

- [ ] A test PR links its Issue with `Closes #6`.
- [ ] A teammate approves the PR before normal **Merge pull request**.
- [ ] The Issue closes and the Project item reaches **Done**.

## Quick Reference

**Check duplicates → Create Issue → Add exact branch name → Add to Project → Set owner/fields → Initial status → In Progress when started**

Next: [Branches and Pull Requests](branches-and-pull-requests.md).

GitHub help: [Creating an Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue) · [Adding Project items](https://docs.github.com/en/issues/planning-and-tracking-with-projects/managing-items-in-your-project/adding-items-to-your-project).
