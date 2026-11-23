import hashlib
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from recipes import LEGACY_PROTOCOL


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck")


def prepare_data(options):
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive

    root = Path(options["root"])
    root.mkdir(parents=True, exist_ok=True)
    files = CIFAR10.train_list + CIFAR10.test_list + [
        [CIFAR10.meta["filename"], CIFAR10.meta["md5"]]]

    def ready():
        return all(check_integrity(root / CIFAR10.base_folder / name, checksum)
                   for name, checksum in files)

    if ready():
        print("CIFAR-10 files verified:", root, flush=True)
        return
    if not options["download"]:
        raise RuntimeError("CIFAR-10 files are missing or corrupted in {}. "
                           "Extract cifar-10-python.tar.gz there or enable download.".format(root))
    archive = root / CIFAR10.filename
    if check_integrity(archive, CIFAR10.tgz_md5):
        extract_archive(str(archive), str(root))
        if ready():
            print("Local CIFAR-10 archive extracted and verified:", root, flush=True)
            return
    urls = options.get("download_urls", [
        "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz", CIFAR10.url])
    curl = shutil.which("curl")
    errors = []
    for url in urls:
        try:
            print("Downloading CIFAR-10 with torchvision:", url, flush=True)
            download_and_extract_archive(url, str(root), filename=CIFAR10.filename,
                                         md5=CIFAR10.tgz_md5)
            if not ready():
                raise RuntimeError("Extracted CIFAR-10 files failed checksum verification")
            return
        except Exception as exc:
            errors.append("torchvision {}: {}".format(url, exc))
            print("torchvision download failed:", exc, flush=True)
        if curl is None:
            continue
        partial = archive.with_name(archive.name + ".part")
        try:
            print("Retrying CIFAR-10 download with curl:", url, flush=True)
            subprocess.run([
                curl, "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
                "--retry", "2", "--retry-delay", "2", "--connect-timeout", "20",
                "--max-time", "300", "--output", str(partial), url,
            ], check=True, timeout=960)
            if not check_integrity(partial, CIFAR10.tgz_md5):
                raise RuntimeError("Downloaded archive failed MD5 verification")
            partial.replace(archive)
            extract_archive(str(archive), str(root))
            if not ready():
                raise RuntimeError("Extracted CIFAR-10 files failed checksum verification")
            return
        except Exception as exc:
            errors.append("curl {}: {}".format(url, exc))
            print("curl download failed:", exc, flush=True)
        finally:
            if partial.exists():
                partial.unlink()
    raise RuntimeError("Automatic CIFAR-10 download failed. Download cifar-10-python.tar.gz "
                       "manually into {} and rerun; the archive will be verified and extracted.\n{}"
                       .format(root, "\n".join(errors)))


def stratified_split(targets, root, seed=42, val_size=5000, index_path=None):
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != (50000,) or not np.array_equal(
            np.bincount(targets, minlength=10), np.full(10, 5000)):
        raise ValueError("Expected the balanced official 50,000-image training set")
    if seed != 42 or val_size != 5000:
        raise ValueError("Use the shared split_seed=42 and val_size=5000")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = Path(index_path) if index_path is not None else root / "split_seed42.npz"
    if index_path is not None and not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(targets.tobytes()).hexdigest()
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(targets == label))
        val.extend(indices[:500])
        train.extend(indices[500:])
    expected_train = rng.permutation(np.asarray(train, dtype=np.int64))
    expected_val = rng.permutation(np.asarray(val, dtype=np.int64))
    if not path.exists():
        try:
            with path.open("xb") as stream:
                np.savez_compressed(stream, train=expected_train, val=expected_val,
                                    seed=np.int64(seed), labels_sha256=digest)
        except FileExistsError:
            pass
    with np.load(path, allow_pickle=False) as saved:
        train, val = saved["train"].copy(), saved["val"].copy()
        if (int(saved["seed"]) != seed or str(saved["labels_sha256"]) != digest
                or not np.array_equal(train, expected_train)
                or not np.array_equal(val, expected_val)):
            raise ValueError("Saved split differs from the reproducible split: " + str(path))
    if len(np.unique(np.concatenate([train, val]))) != 50000:
        raise ValueError("Training and validation indices must be disjoint and exhaustive")
    info = {
        "path": str(path.resolve()),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": seed, "train_samples": len(train), "validation_samples": len(val),
        "train_per_class": np.bincount(targets[train], minlength=10).tolist(),
        "validation_per_class": np.bincount(targets[val], minlength=10).tolist(),
        "sample_index_definition": "zero-based index in the official training set",
    }
    return train, val, info


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, config, training=False):
    options = config["training"]
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=options["batch_size"], shuffle=training,
                      num_workers=options["num_workers"], drop_last=False,
                      pin_memory=True, worker_init_fn=seed_worker,
                      generator=generator)


def training_transform(options):
    from torchvision import transforms
    normal = transforms.Normalize(options["mean"], options["std"])
    train_transforms = [
        transforms.RandomCrop(32, padding=options["crop_padding"]),
        transforms.RandomHorizontalFlip(options["horizontal_flip_probability"]),
    ]
    if "randaugment" in options:
        augment = dict(options["randaugment"])
        if augment.pop("interpolation") != "bilinear":
            raise ValueError("RandAugment requires bilinear interpolation")
        train_transforms.append(transforms.RandAugment(
            **augment, interpolation=transforms.InterpolationMode.BILINEAR))
    if "color_jitter" in options:
        train_transforms.append(transforms.ColorJitter(**options["color_jitter"]))
    train_transforms.extend([transforms.ToTensor(), normal])
    if "random_erasing" in options:
        train_transforms.append(transforms.RandomErasing(**options["random_erasing"]))
    return transforms.Compose(train_transforms)


def mix_batch(inputs, labels, options, force=False):
    if options is None or (not force and options["probability"] < 1
                           and np.random.random() >= options["probability"]):
        return inputs, labels, None, 1.0
    alpha = options["alpha"]
    weight = float(np.random.beta(alpha, alpha))
    paired = torch.randperm(inputs.size(0), device=inputs.device)
    if options["name"] == "mixup":
        inputs = inputs * weight + inputs[paired] * (1 - weight)
    elif options["name"] == "cutmix":
        height, width = inputs.shape[-2:]
        cut_h, cut_w = int(height * np.sqrt(1 - weight)), int(width * np.sqrt(1 - weight))
        cy, cx = int(np.random.randint(height)), int(np.random.randint(width))
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h - cut_h // 2, height)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w - cut_w // 2, width)
        mixed = inputs.clone()
        mixed[:, :, y1:y2, x1:x2] = inputs[paired, :, y1:y2, x1:x2]
        inputs = mixed
        weight = 1.0 - ((y2 - y1) * (x2 - x1)) / (height * width)
    else:
        raise ValueError("Unknown batch mixing method")
    return inputs, labels, labels[paired], weight


def training_data(config):
    from torchvision import datasets, transforms

    options = config["data"]
    prepare_data(options)
    normal = transforms.Normalize(options["mean"], options["std"])
    train_transform = training_transform(options)
    eval_transform = transforms.Compose([transforms.ToTensor(), normal])
    train_data = datasets.CIFAR10(options["root"], train=True,
                                 download=False, transform=train_transform)
    val_data = datasets.CIFAR10(options["root"], train=True,
                               download=False, transform=eval_transform)
    train, val, split_info = stratified_split(
        train_data.targets, options["root"], options["split_seed"], options["val_size"],
        index_path=(Path(config["ablation"]["baseline_run"]) / "split_indices.npz"
                    if config.get("protocol") == LEGACY_PROTOCOL else None))
    return (make_loader(Subset(train_data, train.tolist()), config, training=True),
            make_loader(Subset(val_data, val.tolist()), config), split_info)


def test_data(config):
    from torchvision import datasets, transforms

    options = config["data"]
    prepare_data(options)
    dataset = datasets.CIFAR10(
        options["root"], train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(options["mean"], options["std"]),
        ]))
    return dataset, make_loader(dataset, config)


def evaluation_data(config, index_path):
    from torchvision import datasets, transforms

    options = config["data"]
    prepare_data(options)
    dataset = datasets.CIFAR10(
        options["root"], train=True, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(options["mean"], options["std"]),
        ]))
    train, val, _ = stratified_split(dataset.targets, options["root"], index_path=index_path)
    return (make_loader(Subset(dataset, train.tolist()), config),
            make_loader(Subset(dataset, val.tolist()), config))
