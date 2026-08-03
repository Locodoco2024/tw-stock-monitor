#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STATE_DIR="${1:-runtime/institutional_twse}"
REMOTE="${2:-origin}"
BRANCH="${3:-state}"
for file in rolling_market_data.csv.gz universe.csv score_history.csv.gz latest_scores.csv.gz notification_plan.csv update_manifest.json; do
  [[ -f "$STATE_DIR/$file" ]] || { echo "Missing TWSE seed file: $file" >&2; exit 1; }
done

tmp="$(mktemp -d)"
cleanup() {
  git worktree remove --force "$tmp" >/dev/null 2>&1 || true
  rm -rf "$tmp"
}
trap cleanup EXIT

if git fetch "$REMOTE" "$BRANCH"; then
  git worktree add --force -B "$BRANCH" "$tmp" "$REMOTE/$BRANCH"
else
  git worktree add --detach "$tmp" HEAD
  git -C "$tmp" switch --orphan "$BRANCH"
  git -C "$tmp" rm -rf . || true
fi
rm -rf "$tmp/institutional_twse"
cp -R "$STATE_DIR" "$tmp/institutional_twse"
if [[ -f runtime/state.json ]]; then
  cp runtime/state.json "$tmp/state.json"
elif [[ ! -f "$tmp/state.json" ]]; then
  printf '{"records":{},"institutional_notification_keys":{}}\n' > "$tmp/state.json"
fi
git -C "$tmp" add state.json institutional_twse
git -C "$tmp" config user.name phase6d-seed
git -C "$tmp" config user.email phase6d-seed@users.noreply.github.com
if ! git -C "$tmp" diff --cached --quiet; then
  git -C "$tmp" commit -m "Publish Phase 6D TWSE seed [skip ci]"
fi
git -C "$tmp" push "$REMOTE" "HEAD:$BRANCH"
