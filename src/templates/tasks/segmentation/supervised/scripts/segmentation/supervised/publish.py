"""Release a trained chain, and archive the run that made it, on the Hugging Face Hub.

    python scripts/segmentation/supervised/publish.py \\
        artifacts/<name>/version_0/checkpoints/last.ckpt \\
        --release-repo org/model --training-repo org/model-training

Pass --out release/model to retain an optional local export.

    checkpoint ──▶ release repo (public), optionally exported under release/
    workspace code + this run ──────────────────────────────▶ training repo (private)

The release holds the chain alone: this method defines no ingestor or
postprocessor yet, so `ChainModel.from_pipeline` cannot release a pipeline.
Data belongs in its own dataset repo; LitData stores stay local.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import HfApi

from geosave.ml.release import ChainModel

# An allowlist, so `.env` credentials, data, and other runs never reach the Hub.
# Top-level files are named: a `*` pattern also matches inside folders.
WORKSPACE_FILES = [
    "main.py",
    "README.md",
    "project.yaml",
    "configs/**",
    "scripts/**",
    "modules/**",
]
COMPILED_FILES = ["*.pyc", "**/__pycache__/**"]


def main() -> None:
    """Publish the release; optionally export locally and archive the run."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="Lightning checkpoint to release."
    )
    parser.add_argument("--out", type=Path, help="Optional local release directory.")
    parser.add_argument(
        "--release-repo", required=True, help="Public model repo users load."
    )
    parser.add_argument("--training-repo", help="Private repo of code and runs.")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(workspace))
    from modules.segmentation.supervised.module import Module

    checkpoint = args.checkpoint.resolve()

    # Release: the trained chain, rebuilt from its registered recipe.
    task = Module.load_from_checkpoint(checkpoint, map_location="cpu")
    release = ChainModel.from_chain(task.model)
    if args.out is not None:
        release.save_pretrained(args.out)
        print(args.out)

    api = HfApi()
    if args.training_repo is not None:
        # One commit per run: the code, the run's config, checkpoints and metrics.
        run = checkpoint.parent.parent.relative_to(workspace).as_posix()
        api.create_repo(args.training_repo, private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.training_repo,
            folder_path=workspace,
            allow_patterns=[
                *WORKSPACE_FILES,
                f"{run}/*.yaml",
                f"{run}/events.out.tfevents.*",
                f"{run}/checkpoints/*.ckpt",
            ],
            ignore_patterns=COMPILED_FILES,
            commit_message=f"Run {run}",
        )
        print(f"Archived {run} in {args.training_repo} at {commit.oid}")

    commit = release.push_to_hub(args.release_repo)
    print(f"Released {args.release_repo} at {commit.oid}")


if __name__ == "__main__":
    main()
