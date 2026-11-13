#!/usr/bin/env bash
# Downloads the Core ML models and tokenizer this version of MoonshineCoreML works with, from Hugging Face, into Models/
# at the top of the repo (gitignored), where this repo's code looks for them. To put them somewhere else, such as an
# app's repo, pass that folder as the first argument. The folder is replaced whole, so give the models one of their own.
#
# Hugging Face caches every download (in ~/.cache/huggingface) by commit, so running this again, or going back to an
# earlier release, doesn't download anything.
set -euo pipefail

huggingface_repository=dantrejo03/moonshine-streaming-tiny-coreml
# The release: its tag is for people, its commit is what's downloaded. A tag can be moved to another commit; a commit
# can't change.
release_tag=v1.1
release_commit=69b759028e564acf57ad7ac5af881dceee79e9f6

repository_root=$(cd "$(dirname "$0")/.." && pwd)
models_directory=${1:-$repository_root/Models}
models_directory=${models_directory%/}
partial_directory=$models_directory.partial

# The repo's own virtual environment if it has one, otherwise hf from the PATH.
if [[ -x $repository_root/.venv/bin/hf ]]; then
  hf=$repository_root/.venv/bin/hf
elif command -v hf > /dev/null; then
  hf=hf
else
  echo "Needs the Hugging Face CLI (hf). From the top of the repo:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install huggingface_hub" >&2
  echo "or for the whole system: brew install huggingface-cli" >&2
  exit 1
fi

# The folder gets replaced whole, so refuse anything that holds more than models (Finder's .DS_Store aside).
if [[ -e $models_directory ]]; then
  if [[ ! -d $models_directory ]]; then
    echo "$models_directory is a file, not a folder." >&2
    exit 1
  fi
  unexpected_entries=$(find "$models_directory" -mindepth 1 -maxdepth 1 \
    ! -name "*.mlpackage" ! -name tokenizer.bin ! -name .DS_Store)
  if [[ -n $unexpected_entries ]]; then
    echo "$models_directory holds more than models:" >&2
    echo "$unexpected_entries" >&2
    echo "This script replaces the whole folder, so give the models a folder of their own." >&2
    exit 1
  fi
fi

snapshot_directory=$("$hf" download "$huggingface_repository" --revision "$release_commit" \
  --include "*.mlpackage/*" --include tokenizer.bin --quiet)

# Copy into a folder beside the destination, then swap it in, so an interrupted run never leaves a folder that looks
# complete. -L: the cache's files are symlinks to its blobs. -c: on APFS the copy is a clone and takes no extra space.
rm -rf "$partial_directory"
mkdir -p "$partial_directory"
cp -cRL "$snapshot_directory"/*.mlpackage "$snapshot_directory"/tokenizer.bin "$partial_directory"
rm -rf "$models_directory"
mv "$partial_directory" "$models_directory"
echo "Models $release_tag ($release_commit) are in $models_directory"
