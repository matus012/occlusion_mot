"""CARLA scenario driver — runs under .venv-sim (py3.10, D22). NO omot imports.

Renders one scenario JSON (omot.sim.scenario schema) to a MOT17-style dir with the same
contract as MockBackend: img1/*.jpg, gt/gt.txt (frame,id,x,y,w,h,1,1,vis), seqinfo.ini.

Mechanics: synchronous mode at 1/fps; walkers teleported along the analytic piecewise
paths (deterministic, no AI controllers); RGB + instance-segmentation cameras at the
scenario's pinhole pose; GT boxes from projected actor bounding boxes; visibility from
instance-seg pixel counts normalized by a per-walker fill-factor calibrated on frames
where the walker's box overlaps nothing (post-pass).

Usage: .venv-sim/Scripts/python.exe scripts/sim_driver.py <scenario.json> <out_dir>
       [--host localhost] [--port 2000] [--anchor-spawn 0]
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate={fps}
seqLength={length}
imWidth={w}
imHeight={h}
imExt=.jpg
"""


def walker_pos(waypoints: list, speed: float, t: float) -> tuple[float, float]:
    """Mirror of omot.sim.feeder._walker_pos (kept in sync by the agreement test)."""
    pts = np.asarray(waypoints, dtype=np.float64)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = min(speed * t, float(seg_len.sum()))
    for i, sl in enumerate(seg_len):
        if dist <= sl or i == len(seg_len) - 1:
            r = 0.0 if sl == 0 else min(dist / sl, 1.0)
            p = pts[i] + r * (pts[i + 1] - pts[i])
            return float(p[0]), float(p[1])
        dist -= sl
    return float(pts[-1][0]), float(pts[-1][1])


class LocalFrame:
    """Scenario-local plane -> CARLA world, anchored at a map spawn point."""

    def __init__(self, anchor: carla.Transform) -> None:
        self.origin = anchor.location
        yaw = math.radians(anchor.rotation.yaw)
        self.fwd = np.array([math.cos(yaw), math.sin(yaw)])
        self.right = np.array([-math.sin(yaw), math.cos(yaw)])
        self.base_yaw = anchor.rotation.yaw

    def to_world(self, lx: float, ly: float, z: float) -> carla.Location:
        wx = self.origin.x + lx * self.fwd[0] + ly * self.right[0]
        wy = self.origin.y + lx * self.fwd[1] + ly * self.right[1]
        return carla.Location(x=wx, y=wy, z=self.origin.z + z)


def build_intrinsics(w: int, h: int, fov_deg: float) -> np.ndarray:
    f = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    k = np.identity(3)
    k[0, 0] = k[1, 1] = f
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def project_bbox(
    actor: carla.Actor, cam_transform: carla.Transform, k: np.ndarray, w: int, h: int
) -> np.ndarray | None:
    """Project the actor's 3D bounding box -> image tlwh (clipped), or None."""
    bb = actor.bounding_box
    verts = bb.get_world_vertices(actor.get_transform())
    world_to_cam = np.array(cam_transform.get_inverse_matrix())
    pts = []
    for v in verts:
        p_world = np.array([v.x, v.y, v.z, 1.0])
        p_cam = world_to_cam @ p_world  # UE: x fwd, y right, z up
        if p_cam[0] < 0.2:
            continue
        u = k[0, 2] + k[0, 0] * p_cam[1] / p_cam[0]
        vv = k[1, 2] - k[1, 1] * p_cam[2] / p_cam[0]
        pts.append((u, vv))
    if len(pts) < 4:
        return None
    us = np.clip([p[0] for p in pts], 0, w - 1)
    vs = np.clip([p[1] for p in pts], 0, h - 1)
    box = np.array([us.min(), vs.min(), us.max() - us.min(), vs.max() - vs.min()])
    if box[2] < 2 or box[3] < 4:
        return None
    return box


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--anchor-spawn", type=int, default=0)
    ap.add_argument("--debug-votes", action="store_true",
                    help="dump the walker<->instance-id vote matrix to votes_debug.json")
    args = ap.parse_args()

    spec = json.loads(args.scenario.read_text(encoding="utf-8"))
    fps: int = spec["fps"]
    n_frames = int(spec["duration_s"] * fps)
    img_w, img_h = spec["img_w"], spec["img_h"]

    seq_dir = args.out_dir / spec["scenario_id"]
    (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    original_settings = world.get_settings()
    actors: list[carla.Actor] = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        world.apply_settings(settings)

        anchor = world.get_map().get_spawn_points()[args.anchor_spawn]
        frame_conv = LocalFrame(anchor)
        bp_lib = world.get_blueprint_library()

        # --- walkers (deterministic blueprint choice per id) --------------------------
        # Child pedestrian models (0009-0014) fail to render via teleport spawning on
        # this setup (calibration reads only scenery where they stand) — excluded.
        child_models = {f"walker.pedestrian.{i:04d}" for i in range(9, 15)}
        walker_bps = sorted(
            (b for b in bp_lib.filter("walker.pedestrian.*") if b.id not in child_models),
            key=lambda b: b.id,
        )
        walkers: dict[int, carla.Actor] = {}
        bp_of_walker: dict[int, str] = {}
        # D24 diversification with a hard uniqueness guarantee: same-blueprint walkers
        # in one scenario collide in instance segmentation (shared skeletal-mesh ids —
        # caught by the duplicate-id assert). Per-scenario seeded permutation gives
        # within-scenario uniqueness and cross-scenario diversity over all adult models.
        perm = np.random.default_rng(int(spec["seed"])).permutation(len(walker_bps))
        for slot, wspec in enumerate(spec["walkers"]):
            bp = walker_bps[int(perm[slot % len(walker_bps)])]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "true")
            # Spawn at a spaced holding slot far behind the camera (dense crowds collide
            # at their true start positions), then teleport onto the path — teleports
            # do not collision-check.
            hold = carla.Transform(frame_conv.to_world(-30.0 - 3.0 * slot, 0.0, 1.1))
            actor = world.try_spawn_actor(bp, hold)
            if actor is None:
                hold.location.z += 1.0
                actor = world.spawn_actor(bp, hold)
            actor.set_simulate_physics(False)
            lx, ly = walker_pos(wspec["waypoints"], wspec["speed"], 0.0)
            actor.set_transform(carla.Transform(frame_conv.to_world(lx, ly, 1.1)))
            walkers[wspec["walker_id"]] = actor
            bp_of_walker[wspec["walker_id"]] = bp.id
            actors.append(actor)
            print(f"WALKER {wspec['walker_id']} actor_id={actor.id} bp={bp.id}")

        # --- occluders: best-fit static props ----------------------------------------
        prop_ids = ["static.prop.container", "static.prop.box03", "static.prop.box02"]
        for occ in spec["occluders"]:
            cx, cy, _hw, _hd, _hgt = occ
            bp = bp_lib.find(prop_ids[0])
            tf = carla.Transform(
                frame_conv.to_world(cx, cy, 0.0),
                carla.Rotation(yaw=frame_conv.base_yaw + 90.0),
            )
            prop = world.try_spawn_actor(bp, tf)
            if prop is not None:
                prop.set_simulate_physics(False)
                actors.append(prop)

        # --- cameras -------------------------------------------------------------------
        cam_lx, cam_ly, cam_z = spec["cam_pos"]
        cam_tf = carla.Transform(
            frame_conv.to_world(cam_lx, cam_ly, cam_z),
            carla.Rotation(yaw=frame_conv.base_yaw + spec["cam_yaw_deg"]),
        )
        queues: dict[str, queue.Queue] = {"rgb": queue.Queue(), "iseg": queue.Queue()}
        cams: list[carla.Actor] = []
        for kind, bp_name in (("rgb", "sensor.camera.rgb"),
                              ("iseg", "sensor.camera.instance_segmentation")):
            bp = bp_lib.find(bp_name)
            bp.set_attribute("image_size_x", str(img_w))
            bp.set_attribute("image_size_y", str(img_h))
            bp.set_attribute("fov", str(spec["cam_fov_deg"]))
            cam = world.spawn_actor(bp, cam_tf)
            cam.listen(queues[kind].put)
            cams.append(cam)
            actors.append(cam)
        k = build_intrinsics(img_w, img_h, spec["cam_fov_deg"])

        # --- instance-id calibration -------------------------------------------------
        # Renderer instance ids are scene-internal (NOT actor ids) and heuristics over
        # semantic tags proved fragile. Ground truth instead: hide each walker for one
        # tick and diff the instance images — changed pixels inside its projected bbox
        # are its silhouette, their modal id its renderer id. Deterministic, per-run.
        def read_inst() -> np.ndarray:
            world.tick()
            _ = queues["rgb"].get(timeout=10.0)
            im = queues["iseg"].get(timeout=10.0)
            s = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(img_h, img_w, 4)
            return s[:, :, 1].astype(np.int32) + s[:, :, 2].astype(np.int32) * 256

        # Isolation protocol: all walkers underground, then one at a time at a clear
        # spot in front of the camera — its bbox then contains only itself, so the
        # modal id is exact regardless of t=0 occlusion or off-screen starts.
        inst_id_of: dict[int, int] = {}
        underground = {
            wid: carla.Transform(
                carla.Location(
                    a.get_transform().location.x, a.get_transform().location.y, -60.0
                )
            )
            for wid, a in walkers.items()
        }
        for wid, actor in walkers.items():
            actor.set_transform(underground[wid])
        read_inst()  # flush one tick with everyone hidden
        calib_spot = frame_conv.to_world(6.0, 0.0, 1.1)
        for wid, actor in walkers.items():
            actor.set_transform(carla.Transform(calib_spot))
            for _ in range(4):  # settle: transform latency + skeletal mesh streaming
                read_inst()
            inst = read_inst()
            box = project_bbox(actor, cam_tf, k, img_w, img_h)
            assert box is not None, f"walker {wid} invisible at calibration spot"
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2 = min(img_w, int(box[0] + box[2]) + 1)
            y2 = min(img_h, int(box[1] + box[3]) + 1)
            patch = inst[y1:y2, x1:x2]
            cand = patch[patch != 0]
            assert len(cand) > 50, f"walker {wid}: too few silhouette pixels at calibration"
            ids, counts = np.unique(cand, return_counts=True)
            inst_id_of[wid] = int(ids[np.argmax(counts)])
            actor.set_transform(underground[wid])
        for wspec in spec["walkers"]:  # restore everyone to their t=0 path positions
            lx, ly = walker_pos(wspec["waypoints"], wspec["speed"], 0.0)
            walkers[wspec["walker_id"]].set_transform(
                carla.Transform(frame_conv.to_world(lx, ly, 1.1))
            )
        read_inst()  # settle before frame 1
        print(f"CALIBRATED_IDS {inst_id_of}")
        assert len(set(inst_id_of.values())) == len(inst_id_of), "duplicate instance ids"

        # --- main loop -------------------------------------------------------------------
        # Per frame: count each walker's calibrated instance id inside its padded bbox.
        raw_rows: list[list[float]] = []  # frame, wid, tlwh
        vis_px: dict[tuple[int, int], int] = {}  # (frame, wid) -> visible pixels
        for fidx in range(1, n_frames + 1):
            t = (fidx - 1) / fps
            for wspec in spec["walkers"]:
                lx, ly = walker_pos(wspec["waypoints"], wspec["speed"], t)
                nlx, nly = walker_pos(wspec["waypoints"], wspec["speed"], t + 1.0 / fps)
                yaw = math.degrees(math.atan2(nly - ly, nlx - lx)) if (nlx, nly) != (lx, ly) \
                    else 0.0
                walkers[wspec["walker_id"]].set_transform(
                    carla.Transform(
                        frame_conv.to_world(lx, ly, 1.1),
                        carla.Rotation(yaw=frame_conv.base_yaw + yaw),
                    )
                )
            world.tick()
            rgb = queues["rgb"].get(timeout=10.0)
            iseg = queues["iseg"].get(timeout=10.0)

            rgb_arr = np.frombuffer(rgb.raw_data, dtype=np.uint8).reshape(img_h, img_w, 4)
            cv2.imwrite(str(seq_dir / "img1" / f"{fidx:06d}.jpg"), rgb_arr[:, :, :3])
            seg = np.frombuffer(iseg.raw_data, dtype=np.uint8).reshape(img_h, img_w, 4)
            inst = seg[:, :, 1].astype(np.int32) + seg[:, :, 2].astype(np.int32) * 256

            for wid, actor in walkers.items():
                box = project_bbox(actor, cam_tf, k, img_w, img_h)
                if box is None:
                    continue
                raw_rows.append([fidx, wid, *box.tolist()])
                iid = inst_id_of.get(wid)
                if iid is None:
                    continue
                pad_x, pad_y = box[2] * 0.15, box[3] * 0.08
                x1 = max(0, int(box[0] - pad_x))
                y1 = max(0, int(box[1] - pad_y))
                x2 = min(img_w, int(box[0] + box[2] + pad_x) + 1)
                y2 = min(img_h, int(box[1] + box[3] + pad_y) + 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                vis_px[(fidx, wid)] = int((inst[y1:y2, x1:x2] == iid).sum())

        # --- visibility post-pass: per-walker fill factor -------------------------------
        rows_by_walker: dict[int, list[list[float]]] = {}
        for r in raw_rows:
            rows_by_walker.setdefault(int(r[1]), []).append(r)
        gt_lines: list[str] = []
        for wid, rows in rows_by_walker.items():
            fills = [
                vis_px.get((int(r[0]), wid), 0) / max(r[4] * r[5], 1.0) for r in rows
            ]
            plausible = [f for f in fills if 0.15 <= f <= 0.98]
            fill_ref = float(np.percentile(plausible, 90)) if plausible else 0.6
            fill_ref = max(fill_ref, 1e-3)
            for r in rows:
                expected = max(r[4] * r[5] * fill_ref, 1.0)
                vis = float(np.clip(vis_px.get((int(r[0]), wid), 0) / expected, 0.0, 1.0))
                gt_lines.append(
                    f"{int(r[0])},{wid},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
                    f"1.0000,1,{vis:.4f}"
                )
        (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
        (seq_dir / "seqinfo.ini").write_text(
            SEQINFO.format(name=spec["scenario_id"], fps=fps, length=n_frames,
                           w=img_w, h=img_h),
            encoding="utf-8",
        )
        (seq_dir / "scenario.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        (seq_dir / "walkers_meta.json").write_text(
            json.dumps({"blueprint_of_walker": {str(k): v for k, v in bp_of_walker.items()}},
                       indent=2),
            encoding="utf-8",
        )
        print(f"SIM_RENDER_OK {spec['scenario_id']} frames={n_frames} gt_rows={len(gt_lines)}")
        return 0
    finally:
        for a in actors:
            try:
                if a.type_id.startswith("sensor."):
                    a.stop()
                a.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original_settings)


if __name__ == "__main__":
    sys.exit(main())
