import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck")


def stratified_split(targets, root, seed=42, val_size=5000):
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != (50000,) or not np.array_equal(
            np.bincount(targets, minlength=10), np.full(10, 5000)):
        raise ValueError("Expected the balanced official 50,000-image training set")
    if seed != 42 or val_size != 5000:
        raise ValueError("The basic experiment fixes split_seed=42 and val_size=5000")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "split_seed42.npz"
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


def training_data(config):
    from torchvision import datasets, transforms

    options = config["data"]
    normal = transforms.Normalize(options["mean"], options["std"])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=options["crop_padding"]),
        transforms.RandomHorizontalFlip(options["horizontal_flip_probability"]),
        transforms.ToTensor(), normal,
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), normal])
    train_data = datasets.CIFAR10(options["root"], train=True,
                                 download=options["download"], transform=train_transform)
    val_data = datasets.CIFAR10(options["root"], train=True,
                               download=False, transform=eval_transform)
    train, val, split_info = stratified_split(
        train_data.targets, options["root"], options["split_seed"], options["val_size"])
    return (make_loader(Subset(train_data, train.tolist()), config, training=True),
            make_loader(Subset(val_data, val.tolist()), config), split_info)


def test_data(config):
    from torchvision import datasets, transforms

    options = config["data"]
    dataset = datasets.CIFAR10(
        options["root"], train=False, download=options["download"],
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(options["mean"], options["std"]),
        ]))
    return dataset, make_loader(dataset, config)
