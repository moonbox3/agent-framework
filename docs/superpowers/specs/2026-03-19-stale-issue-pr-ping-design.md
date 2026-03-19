# Stale Issue & PR Follow-Up Ping

## Problem

When a teammate responds to an external community issue or PR, the original author sometimes doesn't reply. These items go stale with no visibility. We need an automated daily check that pings the author after a configurable number of days of silence.

## Scope

- Open GitHub issues (not opened by team members)
- Open GitHub pull requests (not opened by team members)
- Same logic, same label, same threshold for both

## Trigger

- **Schedule:** Daily cron at midnight UTC (`0 0 * * *`)
- **Manual:** `workflow_dispatch` with configurable `days_threshold` input (default: `4`)

## Behavior

For each open issue and PR:

1. **Skip** if it already has the `needs-info` label
2. **Skip** if the author is a team member
3. **Skip** if there are no comments
4. **Find the last comment from a team member** (walk backwards through comments, ignoring bot and non-team-member comments)
5. **Skip** if no team member has commented
6. **Skip** if the author has commented more recently than the last team member comment
7. **Check** if `DAYS_THRESHOLD` days have passed since that last team member comment
8. If all conditions met:
   - Post a comment: `"@{author}, just checking in — do you have any updates on this?"`
   - Add the `needs-info` label

The `needs-info` label is left for teammates to manually remove when appropriate.

## Architecture

### Workflow YAML (`.github/workflows/stale-issue-pr-ping.yml`)

Thin orchestration layer:

- **Permissions:** `issues: write`, `pull-requests: write`
- **Concurrency:** `group: stale-issue-pr-ping`, `cancel-in-progress: true`
- **Token:** Must use `secrets.GH_ACTIONS_PR_WRITE` (not the default `GITHUB_TOKEN`) because the Teams API requires `read:org` scope.
- **Steps:**
  1. Checkout repo (`actions/checkout@v6`)
  2. Set up Python via `actions/setup-python@v5`
  3. `pip install PyGithub`
  4. Run `.github/scripts/stale_issue_pr_ping.py` with env vars:
     - `GITHUB_TOKEN` — from `secrets.GH_ACTIONS_PR_WRITE`
     - `GITHUB_REPOSITORY` — pre-set by Actions
     - `TEAM_NAME` — from `secrets.DEVELOPER_TEAM`
     - `DAYS_THRESHOLD` — from workflow input or default `4`

Note: We use `actions/setup-python` + `pip install` rather than the repo's `python-setup` composite action (which sets up `uv` and the full workspace). This script is a standalone ops tool with a single dependency — the full workspace setup is unnecessary.

### Python Script (`.github/scripts/stale_issue_pr_ping.py`)

Core logic, structured for testability:

**Functions:**

- `main()` — reads env vars, orchestrates the scan
- `get_team_members(github_client, org, team_slug) -> set[str]` — fetches team member usernames once upfront (cached for the run). `org` is parsed from `GITHUB_REPOSITORY` (the portion before `/`).
- `find_last_team_comment(issue, team_members) -> Comment | None` — walks comments backwards, returns the last comment from a team member (skipping bots and non-team-members)
- `should_ping(issue, team_members, days_threshold) -> bool` — applies all skip conditions, returns whether this item should be pinged
- `ping(issue, author) -> None` — posts the follow-up comment and adds the `needs-info` label

**Edge cases:**

- **Pull requests:** GitHub's issues API returns PRs too. Instead of filtering them out, we process them with the same logic. Note: only top-level conversation comments are examined, not inline PR review comments — this is sufficient for detecting author silence.
- **Bot comments:** Ignored when searching for the last team member comment. Only comments from usernames in the team members set count.
- **Pagination:** PyGithub handles pagination automatically for large issue/comment lists.
- **Rate limiting:** PyGithub handles rate limiting with automatic retries.
- **Team API errors:** Fail loudly with a clear error rather than silently skipping.

**Logging:**

- Print summary: `"Scanned X items, pinged Y"` with list of pinged issue/PR numbers
- Log skip reasons for debugging

## Configuration

| Variable | Source | Default | Description |
|----------|--------|---------|-------------|
| `GITHUB_TOKEN` | `secrets.GH_ACTIONS_PR_WRITE` | — | API authentication (requires `read:org` for Teams API) |
| `GITHUB_REPOSITORY` | GitHub Actions env | — | `owner/repo` |
| `TEAM_NAME` | `secrets.DEVELOPER_TEAM` | — | Team slug for membership check |
| `DAYS_THRESHOLD` | Workflow input | `4` | Days of silence before pinging |

## Not In Scope

- Auto-removing `needs-info` label when the author responds (teammates handle manually)
- Closing stale issues/PRs after extended silence
- Different thresholds or labels for issues vs PRs
- Unit tests (standalone ops script, can add later)
