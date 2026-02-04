#!/bin/bash
# Run all tests: doctests and integration tests
#
# Usage:
#   ./tools/run_tests.sh              # Run all tests (quiet mode - only errors shown)
#   ./tools/run_tests.sh --verbose    # Run all tests with full output
#   ./tools/run_tests.sh --quick      # Skip integration tests (doctests only)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
QUICK_MODE=false
VERBOSE=false
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --verbose|-v)
            VERBOSE=true
            ;;
    esac
done

# Helper to run commands - suppresses stdout in quiet mode, keeps stderr for errors
run_quiet() {
    if [ "$VERBOSE" = true ]; then
        "$@"
    else
        "$@" > /dev/null
    fi
}

echo "Running doctests..."
run_quiet pytest -q --xdoctest --xdoctest-modules src/
echo "Doctests passed."

if [ "$QUICK_MODE" = true ]; then
    echo "Quick mode: skipping integration tests."
    exit 0
fi

echo "Running EPM pipeline integration test..."
cd tests/epm_pipeline
run_quiet env MPLBACKEND=Agg python epm_pipeline.py
cd "$PROJECT_ROOT"
echo "EPM pipeline passed."

echo "Running OFT pipeline integration test..."
cd tests/oft_pipeline
run_quiet env MPLBACKEND=Agg python oft_pipeline.py
cd "$PROJECT_ROOT"
echo "OFT pipeline passed."

echo "All tests passed!"
