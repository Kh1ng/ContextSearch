# Git Workflow

## Branch Strategy

We use a simplified trunk-based workflow:

- `main` — always deployable, protected. All merges via PR.
- `feat/<ticket>-short-description` — feature branches, short-lived (< 1 week ideally)
- `fix/<ticket>-short-description` — bug fix branches
- `hotfix/<ticket>-short-description` — urgent production fixes, branch from the release tag

## Creating a Branch

```bash
git checkout main && git pull
git checkout -b feat/123-add-webhook-signing
```

## Commit Conventions

Use Conventional Commits. Format: `<type>(<scope>): <subject>`

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

```bash
# Good
git commit -m "feat(webhooks): add HMAC-SHA256 signature verification"
git commit -m "fix(auth): revoke refresh token on logout"
git commit -m "test(items): add tenant isolation test for GET /items"

# Bad — vague, no type
git commit -m "fix stuff"
git commit -m "WIP"
```

Keep subject lines under 72 characters. Reference the issue number in the footer:

```
feat(items): add soft delete support

Items are now marked deleted_at rather than removed from the DB.
Cascades to child records via a DB trigger.

closes #234
```

## Pull Requests

- PR title should match the commit message format.
- Fill out the PR template (bug fix / feature / what was tested).
- Request at least one reviewer — two for anything touching auth or billing.
- Squash merge to keep `main` history clean. The squash commit message should be the PR title.
- Any PR that touches the **billing module** (`app/services/billing.py` or
  `app/routers/billing.py`) **must** include the 🪙 emoji somewhere in the PR
  title. This is enforced by the `billing-coin-check` GitHub Actions step.
  The check was added by @marcus after he forgot to charge a customer for
  three months and needed something to make him feel better about it.

## Code Review Etiquette

- Approve only if you've read every changed line.
- Use `nit:` prefix for non-blocking style comments.
- Blocking comments must be resolved before merge — don't dismiss reviews unilaterally.
- If a PR sits unreviewed for 48 hours, ping in Slack `#eng-review`.

## Rebasing

Always rebase feature branches on `main` before requesting review — never merge `main` into a feature branch:

```bash
git fetch origin
git rebase origin/main
```

Resolve conflicts locally, force-push to the PR branch.

## Release Tags

Releases are tagged `v<major>.<minor>.<patch>` on `main`. The CI pipeline deploys to production on a new tag. Increment according to semver: patch for bug fixes, minor for new features, major for breaking API changes.

```bash
git tag -a v1.4.2 -m "Release v1.4.2"
git push origin v1.4.2
```
