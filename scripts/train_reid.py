"""Phase 6: train the occlusion-robust re-ID embedding (D24 two-source design).

Recipe: ResNet18 trunk -> 256-d embedding (BN-neck), cross-entropy + batch-hard triplet,
P x K identity sampling with low-visibility upweighting, random-erasing augmentation
(synthetic occlusion). Eval: identity-disjoint val — standard rank-1/mAP plus the
OCCLUDED-QUERY protocol (D38 bounds: query vis in [0.10, 0.50), gallery vis >= 0.60) —
the metric aligned with the tracker's re-emergence gating job.

Ablation arms via --sources: mot17_dev | sim | mot17_dev sim.
Local prototype: --epochs 5 on the 4060. Full run: PERUN (single-GPU per node is fine
at this model size; device/seed/batch are injected — D8).

Usage:
  .venv/Scripts/python.exe scripts/train_reid.py --sources mot17_dev sim --epochs 25 \
      --tag simreal [--device cuda] [--batch-p 16] [--batch-k 4] [--eval-every 5]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    stream=sys.stdout)
logger = logging.getLogger("train_reid")

ROOT = Path(__file__).resolve().parents[1]
CROP_HW = (128, 64)
EMBED_DIM = 256
# D38 occluded-query visibility bounds (derived from the MOT20 GT vis distribution):
# queries vis in [Q_LO, Q_HI) — Q_LO > 0 excludes pixel-less slivers (vis<0.1 is
# 14-22% of MOT20 GT rows) that are unanswerable as queries; Q_HI = 0.5 matches the
# D14 occlusion boundary. Gallery vis >= G_LO — the [0.5, 0.6) band (7-8% of rows)
# is boundary-ambiguous; 0.6 gives a clean visible-gallery margin.
QUERY_VIS_LO = 0.10
QUERY_VIS_HI = 0.50
GALLERY_VIS_LO = 0.60


def eval_epochs(epochs: int, eval_every: int) -> set[int]:
    """D43-delta(a) cadence: the set of 1-based epochs that get evaluated -- every
    `eval_every`th epoch, PLUS always the final epoch (so a checkpoint always exists
    even when `epochs % eval_every != 0`). eval_every=1 evaluates every epoch
    (byte-identical to the pre-delta unconditional behavior)."""
    assert epochs > 0, f"epochs must be positive, got {epochs}"
    assert eval_every > 0, f"eval_every must be a positive int, got {eval_every}"
    return {e for e in range(1, epochs + 1) if e % eval_every == 0} | {epochs}


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_items(sources: list[str]) -> tuple[list[dict], list[dict]]:
    """Crop records from each source's index; identity-disjoint train/val via split hints."""
    train: list[dict] = []
    val: list[dict] = []
    for src in sources:
        base = ROOT / "data" / "reid" / src
        index = json.loads((base / "index.json").read_text(encoding="utf-8"))
        for ident, e in index["identities"].items():
            ident_dir = base / ident
            label = f"{src}/{ident}"
            for jpg in ident_dir.glob("*.jpg"):
                vis = int(jpg.stem.rsplit("_v", 1)[1]) / 100.0
                rec = {"path": str(jpg), "identity": label, "vis": vis}
                (train if e["split"] == "train" else val).append(rec)
    return train, val


def subsample_identities(
    train: list[dict], sources: list[str], frac: float, seed: int
) -> list[dict]:
    """Subsample TRAIN identities per source to round(frac * n_ids), nested across fracs.

    D29 identity-scaling study: for a fixed seed, the identity set kept at frac f1 < f2
    is a strict subset of the set kept at f2, because both are prefixes of the SAME
    seeded shuffle of the sorted identity list (i.e. we shuffle once per source, then
    slice a prefix -- shrinking frac only ever drops identities off the tail).
    """
    assert 0.0 < frac <= 1.0, f"identity-frac must be in (0, 1], got {frac}"
    if frac >= 1.0:
        return train
    by_source: dict[str, set[str]] = defaultdict(set)
    for rec in train:
        src = rec["identity"].split("/", 1)[0]
        by_source[src].add(rec["identity"])
    keep: set[str] = set()
    for src in sources:
        idents = sorted(by_source.get(src, set()))
        rng = random.Random(seed)
        rng.shuffle(idents)
        n_keep = max(1, round(frac * len(idents))) if idents else 0
        kept = idents[:n_keep]
        keep.update(kept)
        logger.info("identity-frac=%.3f source=%s: kept %d/%d identities",
                    frac, src, len(kept), len(idents))
    return [rec for rec in train if rec["identity"] in keep]


class ReidDataset:
    def __init__(self, items: list[dict], training: bool) -> None:
        import torch  # noqa: F401

        self.items = items
        self.training = training
        self.by_identity: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(items):
            self.by_identity[rec["identity"]].append(i)
        self.labels = {ident: j for j, ident in enumerate(sorted(self.by_identity))}

    def __len__(self) -> int:
        return len(self.items)

    def load(self, idx: int):  # -> torch.Tensor
        import cv2
        import torch

        rec = self.items[idx]
        img = cv2.imread(rec["path"])[:, :, ::-1].astype(np.float32) / 255.0
        if self.training:
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy()
            if random.random() < 0.4:  # random erasing = synthetic occlusion
                eh = random.randint(CROP_HW[0] // 6, CROP_HW[0] // 2)
                ew = random.randint(CROP_HW[1] // 4, CROP_HW[1])
                ey = random.randint(0, CROP_HW[0] - eh)
                ex = random.randint(0, CROP_HW[1] - ew)
                img[ey : ey + eh, ex : ex + ew] = np.random.uniform(size=(eh, ew, 3))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class PKSampler:
    """P identities x K instances per batch; low-visibility crops upweighted."""

    def __init__(self, ds: ReidDataset, p: int, k: int, seed: int) -> None:
        self.ds = ds
        self.p = p
        self.k = k
        self.rng = random.Random(seed)
        self.idents = [i for i, idxs in ds.by_identity.items() if len(idxs) >= 2]

    def batches(self, n_batches: int):
        for _ in range(n_batches):
            batch: list[int] = []
            for ident in self.rng.sample(self.idents, min(self.p, len(self.idents))):
                idxs = self.ds.by_identity[ident]
                weights = [1.0 + (self.ds.items[i]["vis"] < 0.5) for i in idxs]
                chosen = self.rng.choices(idxs, weights=weights,
                                          k=min(self.k, len(idxs)))
                batch.extend(chosen)
            yield batch


def build_model(device: str):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    class ReidNet(torch.nn.Module):
        def __init__(self, n_classes: int) -> None:
            super().__init__()
            trunk = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            trunk.fc = torch.nn.Identity()
            self.trunk = trunk
            self.embed = torch.nn.Linear(512, EMBED_DIM)
            self.bnneck = torch.nn.BatchNorm1d(EMBED_DIM)
            self.classifier = torch.nn.Linear(EMBED_DIM, n_classes, bias=False)

        def forward(self, x):  # noqa: ANN001
            feat = self.embed(self.trunk(x))
            neck = self.bnneck(feat)
            return feat, self.classifier(neck)

    return ReidNet


def batch_hard_triplet(feats, labels, margin: float = 0.3):  # noqa: ANN001
    import torch

    feats = torch.nn.functional.normalize(feats, dim=1)
    dist = torch.cdist(feats, feats)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    hardest_pos = (dist * (same & ~eye).float()).max(dim=1).values
    inf = torch.full_like(dist, float("inf"))
    hardest_neg = torch.where(same, inf, dist).min(dim=1).values
    return torch.nn.functional.relu(hardest_pos - hardest_neg + margin).mean()


def evaluate(model, ds: ReidDataset, device: str, max_queries: int = 800):  # noqa: ANN001
    """Rank-1 / mAP overall + occluded-query protocol on the identity-disjoint val set."""
    import torch

    model.eval()
    feats = torch.zeros(len(ds.items), EMBED_DIM)
    with torch.no_grad():
        for start in range(0, len(ds.items), 256):
            idxs = list(range(start, min(start + 256, len(ds.items))))
            batch = torch.stack([ds.load(i) for i in idxs]).to(device)
            f, _ = model(batch)
            feats[idxs] = torch.nn.functional.normalize(f, dim=1).cpu()
    labels = np.array([ds.labels[r["identity"]] for r in ds.items])
    vis = np.array([r["vis"] for r in ds.items])

    def retrieval(query_mask: np.ndarray, gallery_mask: np.ndarray) -> tuple[float, float]:
        q_idx = np.flatnonzero(query_mask)
        if len(q_idx) > max_queries:
            q_idx = np.random.default_rng(0).choice(q_idx, max_queries, replace=False)
        ranks, aps = [], []
        gal = np.flatnonzero(gallery_mask)
        gf = feats[gal]
        for qi in q_idx:
            g_mask = gal != qi
            sims = (feats[qi] @ gf[g_mask].T).numpy()
            g_labels = labels[gal[g_mask]]
            positives = g_labels == labels[qi]
            if not positives.any():
                continue
            order = np.argsort(-sims)
            hits = positives[order]
            ranks.append(1.0 if hits[0] else 0.0)
            cum = np.cumsum(hits)
            prec = cum[hits] / (np.flatnonzero(hits) + 1)
            aps.append(float(prec.mean()))
        return float(np.mean(ranks)) if ranks else 0.0, float(np.mean(aps)) if aps else 0.0

    all_mask = np.ones(len(ds.items), bool)
    r1_all, map_all = retrieval(all_mask, all_mask)
    r1_occ, map_occ = retrieval(
        (vis >= QUERY_VIS_LO) & (vis < QUERY_VIS_HI), vis >= GALLERY_VIS_LO
    )
    model.train()
    return {"rank1": r1_all, "mAP": map_all, "occ_rank1": r1_occ, "occ_mAP": map_occ}


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["mot17_dev"],
                    choices=["mot17_dev", "sim", "mot20", "market1501"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batches-per-epoch", type=int, default=200)
    ap.add_argument("--batch-p", type=int, default=16)
    ap.add_argument("--batch-k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="reid")
    ap.add_argument("--identity-frac", type=float, default=1.0,
                    help="D29 scaling study: subsample TRAIN identities to this fraction "
                         "(nested prefix of a seeded shuffle, per source)")
    ap.add_argument("--identity-seed", type=int, default=0,
                    help="seed for --identity-frac subsampling (independent of --seed)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="D43-delta(a): evaluate every N epochs PLUS always the final epoch "
                         "(default 1 = byte-identical to eval-every-epoch behavior); cuts "
                         "PERUN budget by ~5x at eval_every=5 without changing which epoch "
                         "wins best-checkpoint selection among evaluated epochs")
    args = ap.parse_args()
    assert 0.0 < args.identity_frac <= 1.0, \
        f"--identity-frac must be in (0, 1], got {args.identity_frac}"
    assert args.eval_every > 0, f"--eval-every must be a positive int, got {args.eval_every}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)
    train_items, val_items = load_items(args.sources)
    if args.identity_frac < 1.0:
        train_items = subsample_identities(
            train_items, args.sources, args.identity_frac, args.identity_seed
        )
    train_ds = ReidDataset(train_items, training=True)
    val_ds = ReidDataset(val_items, training=False)
    logger.info("sources=%s train: %d crops / %d ids; val: %d crops / %d ids; device=%s",
                args.sources, len(train_items), len(train_ds.by_identity),
                len(val_items), len(val_ds.by_identity), device)
    logger.info("D38 occluded-query bounds: query vis [%.2f, %.2f), gallery vis >= %.2f",
                QUERY_VIS_LO, QUERY_VIS_HI, GALLERY_VIS_LO)
    logger.info("eval-every=%d (evaluated epochs: every %dth + always the final epoch %d)",
                args.eval_every, args.eval_every, args.epochs)

    model = build_model(device)(n_classes=len(train_ds.by_identity)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    sampler = PKSampler(train_ds, args.batch_p, args.batch_k, args.seed)

    best = {"occ_rank1": -1.0}
    out_dir = ROOT / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"reid_{args.tag}.pt"
    eval_set = eval_epochs(args.epochs, args.eval_every)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for batch_idx in sampler.batches(args.batches_per_epoch):
            imgs = torch.stack([train_ds.load(i) for i in batch_idx]).to(device)
            labels = torch.tensor(
                [train_ds.labels[train_ds.items[i]["identity"]] for i in batch_idx],
                device=device,
            )
            with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                feats, logits = model(imgs)
                loss = ce(logits, labels) + batch_hard_triplet(feats, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        sched.step()
        if epoch not in eval_set:
            logger.info("epoch %d/%d loss=%.3f (%.0fs) [eval skipped, eval-every=%d]",
                        epoch, args.epochs, float(np.mean(losses)), time.time() - t0,
                        args.eval_every)
            continue
        metrics = evaluate(model, val_ds, device)
        logger.info("epoch %d/%d loss=%.3f %s (%.0fs)", epoch, args.epochs,
                    float(np.mean(losses)), metrics, time.time() - t0)
        if metrics["occ_rank1"] > best["occ_rank1"]:
            best = metrics
            torch.save(
                {"state_dict": model.state_dict(), "embed_dim": EMBED_DIM,
                 "arch": "resnet18_bnneck", "sources": args.sources,
                 "metrics": metrics, "seed": args.seed,
                 "identity_frac": args.identity_frac, "identity_seed": args.identity_seed,
                 "n_train_ids": len(train_ds.by_identity)},
                ckpt_path,
            )
    logger.info("BEST %s -> %s", best, ckpt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
