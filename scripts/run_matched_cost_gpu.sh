#!/usr/bin/env bash
# Full-PG supplies a measured reference; all later methods share its run budget.
set -eo pipefail
export COMPARISON_MODE=wall
exec bash "$(dirname "$0")/run_minimal_gpu.sh" "$@"
