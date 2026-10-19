import os
from pathlib import Path
from huggingface_hub import HfApi


def resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    repo_root = Path(__file__).resolve().parents[2]
    env_candidates = [
        repo_root / ".env",
        repo_root.parent / "Dental_000" / ".env",
        repo_root.parent / "Dental_Panoramic_Reader" / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s.startswith("HF_TOKEN="):
                        return line_s.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def upload_weights_to_hf():
    repo_root = Path(__file__).resolve().parents[2]
    best_pt = repo_root / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = repo_root / "weights" / "yolov8m_best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Weight file not found at {best_pt}")

    token = resolve_hf_token()
    api = HfApi(token=token)

    target_repo = "chemahc94/Dental-AI-Models"
    target_paths_in_repo = [
        "Dental_008_weights_best.pt",
        "Dental_008/yolov8m_best.pt",
        "Dental_008/weights/best.pt",
    ]

    for path_in_repo in target_paths_in_repo:
        print(f"[upload_to_hf] Uploading {best_pt} -> {target_repo}:{path_in_repo} ...")
        api.upload_file(
            path_or_fileobj=str(best_pt),
            path_in_repo=path_in_repo,
            repo_id=target_repo,
            repo_type="model",
            commit_message=f"Update Dental_008 leak-free YOLOv8m-seg weights ({path_in_repo})",
        )
        print(f"[upload_to_hf] Uploaded {path_in_repo} successfully.")


if __name__ == "__main__":
    upload_weights_to_hf()
