import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_or_create_split(
    images_info: list,
    train_img_dir: Path,
    manifest_candidates: list,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    manifest_path = None
    for cand in manifest_candidates:
        if cand and os.path.exists(cand):
            manifest_path = Path(cand)
            break

    rows_by_fname = {}
    if manifest_path is not None:
        print(f"[prepare_yolo_data] Loading deterministic manifest: {manifest_path}")
        train_files = set()
        val_files = set()
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"].strip()
                split = row["split"].strip()
                rows_by_fname[fname] = dict(row)
                if split == "train":
                    train_files.add(fname)
                elif split == "val":
                    val_files.add(fname)
    else:
        print(f"[prepare_yolo_data] Manifest not found; falling back to deterministic 80/20 split (seed={seed})")
        all_files = sorted(img["file_name"] for img in images_info)
        rng = random.Random(seed)
        rng.shuffle(all_files)
        n_train = int(round(len(all_files) * 0.8))
        train_files = set(all_files[:n_train])
        val_files = set(all_files[n_train:])

    # Compute SHA-256 for every image to guarantee 0 hash overlap as well as 0 filename overlap
    file_to_hash = {}
    hash_to_files = {}
    for img in images_info:
        fname = img["file_name"]
        fpath = train_img_dir / fname
        if fpath.exists():
            h = compute_sha256(fpath)
            file_to_hash[fname] = h
            hash_to_files.setdefault(h, []).append(fname)

    train_hashes = {file_to_hash[f] for f in train_files if f in file_to_hash}
    val_hashes = {file_to_hash[f] for f in val_files if f in file_to_hash}
    overlapping_hashes = train_hashes & val_hashes

    if overlapping_hashes:
        print(
            f"[prepare_yolo_data] Detected {len(overlapping_hashes)} duplicate-content image pairs across train/val; "
            f"re-grouping by SHA-256 to enforce 0 byte-level overlap while preserving 507/127 split counts..."
        )
        # Move val files whose hash is already in train into train, and swap with unique-hash train files
        moved_to_train = []
        for vf in sorted(val_files):
            if file_to_hash.get(vf) in overlapping_hashes:
                val_files.remove(vf)
                train_files.add(vf)
                moved_to_train.append(vf)

        swap_needed = len(moved_to_train)
        swapped_to_val = []
        for tf in sorted(train_files):
            if swap_needed <= 0:
                break
            h = file_to_hash.get(tf)
            # Only swap a file whose hash appears exactly once across the entire dataset
            if h and len(hash_to_files.get(h, [])) == 1:
                train_files.remove(tf)
                val_files.add(tf)
                swapped_to_val.append(tf)
                swap_needed -= 1

        print(f"  -> Moved {moved_to_train} to train; swapped {swapped_to_val} to val.")

        # Persist updated manifest to all candidate locations
        if rows_by_fname:
            for fname in train_files:
                if fname in rows_by_fname:
                    rows_by_fname[fname]["split"] = "train"
            for fname in val_files:
                if fname in rows_by_fname:
                    rows_by_fname[fname]["split"] = "val"

            fieldnames = ["image_id", "filename", "split", "tooth_count", "width", "height"]
            sorted_rows = sorted(rows_by_fname.values(), key=lambda r: int(r["image_id"]))
            for cand in manifest_candidates:
                if cand:
                    try:
                        os.makedirs(os.path.dirname(cand), exist_ok=True)
                        with open(cand, "w", encoding="utf-8", newline="") as out_f:
                            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(sorted_rows)
                    except OSError:
                        pass

    # Mirror manifest to reports_archive if missing
    if manifest_path is not None:
        for cand in manifest_candidates:
            if cand and not os.path.exists(cand):
                try:
                    os.makedirs(os.path.dirname(cand), exist_ok=True)
                    shutil.copy2(manifest_path, cand)
                except OSError:
                    pass

    overlap = train_files & val_files
    assert len(overlap) == 0, f"CRITICAL: Train/Val filename overlap detected ({len(overlap)} files): {overlap}"
    print(f"[prepare_yolo_data] Split verified: train={len(train_files)}, val={len(val_files)}, filename_overlap=0")
    return train_files, val_files


def prepare_leak_free_yolo_dataset(
    cache_dir: Path,
    output_dir: Path,
    manifest_candidates: list,
    seed: int = 42,
    verify_hashes: bool = True,
) -> dict:
    train_json = cache_dir / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json"
    train_img_dir = cache_dir / "training_data" / "quadrant_enumeration" / "xrays"

    if not train_json.exists():
        raise FileNotFoundError(f"DENTEX annotation file not found: {train_json}")
    if not train_img_dir.exists():
        raise FileNotFoundError(f"DENTEX image directory not found: {train_img_dir}")

    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_info = data["images"]
    train_files, val_files = load_or_create_split(images_info, train_img_dir, manifest_candidates, seed=seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_to_anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        img_to_anns.setdefault(img_id, []).append(ann)

    counts = {"train": 0, "val": 0}
    for img_info in images_info:
        file_name = img_info["file_name"]
        if file_name in train_files:
            split_name = "train"
        elif file_name in val_files:
            split_name = "val"
        else:
            continue

        width = img_info["width"]
        height = img_info["height"]
        src_img = train_img_dir / file_name
        if not src_img.exists():
            continue

        dst_img = output_dir / "images" / split_name / file_name
        label_file = Path(file_name).stem + ".txt"
        label_path = output_dir / "labels" / split_name / label_file

        anns = img_to_anns.get(img_info["id"], [])
        lines = []
        for ann in anns:
            polygons = ann.get("segmentation", [])
            for poly in polygons:
                if len(poly) < 6:
                    continue
                norm_coords = []
                for i in range(0, len(poly), 2):
                    x = max(0.0, min(1.0, poly[i] / width))
                    y = max(0.0, min(1.0, poly[i + 1] / height))
                    norm_coords.append(f"{x:.6f} {y:.6f}")
                if len(norm_coords) >= 3:
                    lines.append("0 " + " ".join(norm_coords))

        if lines:
            shutil.copy2(src_img, dst_img)
            with open(label_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines) + "\n")
            counts[split_name] += 1

    yaml_path = output_dir / "data.yaml"
    yaml_content = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: Tooth\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    train_out_files = set(os.listdir(output_dir / "images" / "train"))
    val_out_files = set(os.listdir(output_dir / "images" / "val"))
    assert len(train_out_files & val_out_files) == 0, "Filename overlap in generated dataset!"

    hash_overlap_count = 0
    if verify_hashes:
        train_hashes = {compute_sha256(output_dir / "images" / "train" / f) for f in train_out_files}
        val_hashes = {compute_sha256(output_dir / "images" / "val" / f) for f in val_out_files}
        hash_overlap = train_hashes & val_hashes
        hash_overlap_count = len(hash_overlap)
        assert hash_overlap_count == 0, f"CRITICAL: SHA-256 hash overlap detected ({hash_overlap_count} files)!"

    print(
        f"[prepare_yolo_data] Generated leak-free dataset at {output_dir}: "
        f"train={counts['train']}, val={counts['val']}, sha256_overlap={hash_overlap_count}"
    )
    return {
        "output_dir": str(output_dir),
        "train_count": counts["train"],
        "val_count": counts["val"],
        "sha256_overlap": hash_overlap_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare leak-free YOLO segmentation dataset for Dental_008")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expanduser("~/.cache/dentex_dataset"),
        help="Path to DENTEX cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.cache/dentex_dataset/yolo_format"),
        help="Primary output directory for leak-free YOLO dataset",
    )
    parser.add_argument(
        "--mirror-repo-dataset",
        action="store_true",
        default=False,
        help="Also copy full yolo_dataset/ into Dental_008 repository directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    if repo_root.name == "dentex_seg":
        repo_root = repo_root.parents[1]

    dental_000_root = repo_root.parent / "Dental_000"
    manifest_candidates = [
        str(dental_000_root / "reports_archive" / "dental_008_dentex_manifest.csv"),
        str(dental_000_root / "reports" / "dental_008_dentex_manifest.csv"),
    ]

    cache_dir = Path(args.cache_dir)
    primary_out = Path(args.output_dir)
    result = prepare_leak_free_yolo_dataset(
        cache_dir=cache_dir,
        output_dir=primary_out,
        manifest_candidates=manifest_candidates,
        seed=args.seed,
        verify_hashes=True,
    )

    repo_yolo_dir = repo_root / "yolo_dataset"
    repo_yolo_dir.mkdir(parents=True, exist_ok=True)
    with open(repo_yolo_dir / "data.yaml", "w", encoding="utf-8") as yf:
        yf.write("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: Tooth\n")
    with open(repo_yolo_dir / "split_verification.json", "w", encoding="utf-8") as vf:
        json.dump(result, vf, indent=2)

    if args.mirror_repo_dataset and repo_yolo_dir.resolve() != primary_out.resolve():
        print(f"[prepare_yolo_data] Syncing leak-free dataset to {repo_yolo_dir}...")
        if repo_yolo_dir.exists():
            shutil.rmtree(repo_yolo_dir)
        shutil.copytree(primary_out, repo_yolo_dir)


if __name__ == "__main__":
    main()
