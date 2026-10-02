# Reviewing Pull Requests

**VisionX CXR-CAD team guide**

1. **Read the PR and linked Issue.** Understand the goal and its **Done When** checklist.
2. **Open Files changed.** Check that the changes meet the goal, are understandable, and contain no unrelated changes, patient data, datasets, or credentials.
3. **Check verification.** Read the author's testing notes and check automated results, if available. Run the relevant checks locally when needed. If you cannot verify something, say so.
4. **Leave specific comments.** Explain the problem and the expected result. Mark optional suggestions as optional.
5. **Submit your review.** Choose **Approve** if ready, **Request changes** for blocking problems, or **Comment** for questions and feedback without approval.
6. **Recheck fixes.** Review the author's updates and approve when blocking comments are addressed. The author resolves addressed conversations and merges after **1 teammate approval** and any required checks pass.

The author uses normal **Merge pull request**, then checks that the Issue is closed and the Project item is **Done**.

## Small Example

“This split uses image IDs. Please split by patient ID so the same patient cannot appear in both training and test sets. Verify that patient overlap is zero.”

## Quick Reference

**Read Issue → Inspect changes → Check verification → Give clear feedback → Recheck fixes → Approve**
