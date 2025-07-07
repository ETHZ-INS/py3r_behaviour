#!/bin/bash
set -e  # exit on errors

echo "Running release.sh..."

# Extract version from pyproject.toml
VERSION=$(grep '^version' pyproject.toml | cut -d '"' -f2)
echo "Extracted version: '$VERSION'"

TAG="v$VERSION"
echo "Using tag: $TAG"

# Check if version is empty
if [[ -z "$VERSION" ]]; then
  echo "Error: Version is empty! Check pyproject.toml."
  exit 1
fi

# Check if tag exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Tag $TAG already exists, skipping tag creation."
else
  echo "Creating git tag $TAG"
  git tag "$TAG"
  git push origin "$TAG"
fi

echo "Generating changelog..."
auto-changelog

echo "Done."
