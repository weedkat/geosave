"""Release a trained chain, and archive the run that made it, on the Hugging Face Hub.

    python scripts/publish.py \\
        artifacts/<name>/version_0/checkpoints/last.ckpt \\
        --release-repo org/model --version v1.0 --training-repo org/model-training \\
        --base-model ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL --dataset org/model-data

Pass --out release/model to retain an optional local export.

    checkpoint ──▶ release repo (public), tagged with --version
    workspace code + this run ──────────────────────────────▶ training repo (private)

The release holds the model chain and a model card naming its base model,
training data and training run. The editable processing stages travel with
the experiment code. The project owns its training data.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, ModelCard, ModelCardData

# An allowlist, so `.env` credentials, data, and other runs never reach the Hub.
# Top-level files are named: a `*` pattern also matches inside folders.
WORKSPACE_FILES = [
    "main.py",
    "README.md",
    "pipeline.py",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "configs/**",
    "scripts/**",
    "processing/**",
    "training/**",
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
    parser.add_argument(
        "--version", required=True, help="Release tag users pin, such as v1.0."
    )
    parser.add_argument("--training-repo", help="Private repo of code and runs.")
    parser.add_argument(
        "--base-model", help="Hub repo of the pretrained weights the encoder started from."
    )
    parser.add_argument("--dataset", help="Hub repo of the training data.")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(workspace))
    from training.module import Module

    checkpoint = args.checkpoint.resolve()

    task = Module.load_from_checkpoint(checkpoint, map_location="cpu")

    api = HfApi()
    summary = f"Released from checkpoint {checkpoint.name}."
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
        summary = f"Trained in {args.training_repo}, run {run}, at commit {commit.oid}."

    # The card names where the weights came from; save_pretrained keeps an existing card.
    datasets = None
    if args.dataset is not None:
        datasets = [args.dataset]
    card_data = ModelCardData(library_name="geosave", base_model=args.base_model, datasets=datasets)
    card = ModelCard.from_template(card_data, model_summary=summary)

    with TemporaryDirectory() as temporary:
        folder = args.out or Path(temporary)
        folder.mkdir(parents=True, exist_ok=True)
        card.save(folder / "README.md")
        task.model.save_pretrained(folder)

        api.create_repo(args.release_repo, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.release_repo,
            folder_path=folder,
            commit_message=f"Release {args.version}",
        )

    api.create_tag(args.release_repo, tag=args.version, revision=commit.oid)
    print(f"Released {args.release_repo} {args.version} at {commit.oid}")


if __name__ == "__main__":
    main()
