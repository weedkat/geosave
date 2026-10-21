"""Publish a trained run: python scripts/regression/mean_teacher/publish.py org/model artifacts/.../best.ckpt."""

from argparse import ArgumentParser
from pathlib import Path

import sys

from huggingface_hub import HfApi

from geosave.model.release import GeoSaveModel
from geosave.model.spec import ModelSpec

# An allowlist, so `.env` credentials, data, and logs never reach the Hub.
WORKSPACE_FILES = ["*.py", "README.md", "configs/**", "scripts/**", "modules/**"]


def main() -> None:
    """Push the inference release, then this workspace with the run's checkpoints."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Hugging Face model repository.")
    parser.add_argument(
        "checkpoint", type=Path, help="Lightning checkpoint to release."
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(workspace))
    from modules.regression.mean_teacher.module import Module

    checkpoint = args.checkpoint.resolve()
    run = checkpoint.parent.relative_to(workspace)
    task = Module.load_from_checkpoint(checkpoint)
    spec = ModelSpec.load(workspace / "configs/regression-mean-teacher/model_spec.yaml")
    GeoSaveModel.from_chain(task.model, spec=spec).push_to_hub(args.repo_id)

    commit = HfApi().upload_folder(
        repo_id=args.repo_id,
        folder_path=workspace,
        path_in_repo="workspace",
        allow_patterns=[*WORKSPACE_FILES, f"{run.as_posix()}/*.ckpt"],
    )
    print(f"Released {args.repo_id} at {commit.oid}")


if __name__ == "__main__":
    main()
