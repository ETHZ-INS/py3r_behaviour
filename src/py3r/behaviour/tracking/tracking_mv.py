from py3r.behaviour.tracking.tracking import Tracking, LoadOptions
import pandas as pd
import numpy as np
import warnings
import copy


class TrackingMV:
    """
    multi-view tracking object for stereo or multi-camera setups
    can be used as a drop-in replacement for Tracking in TrackingCollection
    stores dict of view name -> Tracking, calibration, and handle
    """

    def __init__(self, views: dict[str, Tracking], calibration: dict, handle: str):
        self.views = views  # e.g., {'left': Tracking, 'right': Tracking}
        self.calibration = calibration
        self.handle = handle

    @classmethod
    def from_dlc(
        cls,
        filepaths: dict[str, str],
        handle: str,
        options: LoadOptions,
        calibration: dict,
    ):
        tracks = {
            view: Tracking.from_dlc(fp, handle=f"{handle}_{view}", options=options)
            for view, fp in filepaths.items()
        }
        return cls(tracks, calibration, handle)

    @classmethod
    def from_dlcma(
        cls,
        filepaths: dict[str, str],
        handle: str,
        options: LoadOptions,
        calibration: dict,
    ):
        tracks = {
            view: Tracking.from_dlcma(fp, handle=f"{handle}_{view}", options=options)
            for view, fp in filepaths.items()
        }
        return cls(tracks, calibration, handle)

    @classmethod
    def from_yolo3r(
        cls,
        filepaths: dict[str, str],
        handle: str,
        options: LoadOptions,
        calibration: dict,
    ):
        tracks = {
            view: Tracking.from_yolo3r(fp, handle=f"{handle}_{view}", options=options)
            for view, fp in filepaths.items()
        }
        return cls(tracks, calibration, handle)

    def stereo_triangulate(self, invert_z: bool = True) -> "Tracking":
        import cv2
        import numpy as np

        calib = self.calibration

        # validation
        views = list(self.views.keys())
        if len(views) != 2 or len(views) != len(set(views)):
            raise ValueError("Exactly two unique views are required for triangulation")
        if not all(v in calib["view_order"] for v in views):
            raise ValueError("Views must correspond to calibration")

        v1, v2 = calib["view_order"][:2]
        K1, dist1 = (
            np.array(calib["views"][v1]["K"]),
            np.array(calib["views"][v1]["dist"]),
        )
        K2, dist2 = (
            np.array(calib["views"][v2]["K"]),
            np.array(calib["views"][v2]["dist"]),
        )
        R, T = (
            np.array(calib["relative_pose"]["R"]),
            np.array(calib["relative_pose"]["T"]).reshape(3, 1),
        )

        # data extraction
        df1, df2 = self.views[v1].data, self.views[v2].data
        if not df1.columns.equals(df2.columns):
            raise ValueError("Views have different columns")
        point_names = self.views[v1].get_point_names()
        frames = df1.index.intersection(df2.index)

        # projection matrices
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, T))

        # helper for triangulation
        def triangulate_point(xs1, ys1, xs2, ys2, l1, l2):
            valid = (
                np.isfinite(xs1)
                & np.isfinite(ys1)
                & np.isfinite(xs2)
                & np.isfinite(ys2)
            )
            n = len(xs1)
            x3d, y3d, z3d, likelihood = [np.full(n, np.nan) for _ in range(4)]
            if np.any(valid):
                pts1 = np.stack([xs1[valid], ys1[valid]], axis=1).astype(np.float32)
                pts2 = np.stack([xs2[valid], ys2[valid]], axis=1).astype(np.float32)
                pts1_ud = cv2.undistortPoints(
                    pts1.reshape(-1, 1, 2), K1, dist1, P=K1
                ).reshape(-1, 2)
                pts2_ud = cv2.undistortPoints(
                    pts2.reshape(-1, 1, 2), K2, dist2, P=K2
                ).reshape(-1, 2)
                pts4d = cv2.triangulatePoints(P1, P2, pts1_ud.T, pts2_ud.T)
                pts3d = (pts4d[:3] / pts4d[3]).T
                x3d[valid], y3d[valid], z3d[valid] = (
                    pts3d[:, 0],
                    pts3d[:, 1],
                    pts3d[:, 2],
                )
                likelihood[valid] = np.minimum(l1[valid], l2[valid])
            return x3d, y3d, z3d, likelihood

        # triangulate all points
        triangulated = {}
        l1_all = df1.loc[frames, [p + ".likelihood" for p in point_names]].values
        l2_all = df2.loc[frames, [p + ".likelihood" for p in point_names]].values
        for i, point in enumerate(point_names):
            xs1, ys1 = (
                df1.loc[frames, point + ".x"].values,
                df1.loc[frames, point + ".y"].values,
            )
            xs2, ys2 = (
                df2.loc[frames, point + ".x"].values,
                df2.loc[frames, point + ".y"].values,
            )
            l1, l2 = l1_all[:, i], l2_all[:, i]
            x3d, y3d, z3d, likelihood = triangulate_point(xs1, ys1, xs2, ys2, l1, l2)
            triangulated[point + ".x"] = x3d
            triangulated[point + ".y"] = y3d
            if invert_z:
                triangulated[point + ".z"] = -z3d
            else:
                triangulated[point + ".z"] = z3d
            triangulated[point + ".likelihood"] = likelihood

        triangulated_df = pd.DataFrame(triangulated, index=frames)
        triangulated_meta = self.views[v1].meta.copy()
        triangulated_meta["calibration"] = calib
        triangulated_meta["views"] = views
        return Tracking(triangulated_df, triangulated_meta, self.handle)

    @staticmethod
    def _get_animals_and_bodyparts(df):
        animals = set()
        bodyparts = set()
        for col in df.columns:
            parts = col.split(".")
            if len(parts) >= 2:
                animals.add(parts[0])
                bodyparts.add(parts[1])
        return sorted(animals), sorted(bodyparts)

    @staticmethod
    def _extract_coords(df, animals, keypoints, frames):
        coords = np.full(
            (len(frames), len(animals), len(keypoints), 2), np.nan, dtype=np.float32
        )
        for ai, animal in enumerate(animals):
            for ki, kp in enumerate(keypoints):
                xcol = f"{animal}.{kp}.x"
                ycol = f"{animal}.{kp}.y"
                if xcol in df.columns and ycol in df.columns:
                    coords[:, ai, ki, 0] = df.loc[frames, xcol].values
                    coords[:, ai, ki, 1] = df.loc[frames, ycol].values
        return coords

    @staticmethod
    def _compute_cost_matrices(coords1, coords2):
        coords1_exp = coords1[:, :, None, :, :]
        coords2_exp = coords2[:, None, :, :, :]
        diffs = coords1_exp - coords2_exp
        sqdist = np.sum(diffs**2, axis=-1)
        dists = np.sqrt(sqdist)
        mask1 = np.isfinite(coords1).all(axis=-1)
        mask2 = np.isfinite(coords2).all(axis=-1)
        mask1_exp = mask1[:, :, None, :]
        mask2_exp = mask2[:, None, :, :]
        valid_mask = mask1_exp & mask2_exp
        dists_masked = np.where(valid_mask, dists, 0.0)
        n_valid = np.sum(valid_mask, axis=-1)
        sum_dists = np.sum(dists_masked, axis=-1)
        large_cost = 1e6
        cost_matrices = np.where(n_valid > 0, sum_dists, large_cost)
        return cost_matrices

    @staticmethod
    def _reorder_view2_data(
        df2, animals1, animals2, keypoints, aligned_indices, frames
    ):
        import numpy as np
        import pandas as pd

        n_frames = len(frames)
        new_data_dict = {}
        for ai, animal1 in enumerate(animals1):
            for ki, kp in enumerate(keypoints):
                for coord in ["x", "y", "likelihood"]:
                    colname = f"{animal1}.{kp}.{coord}"
                    new_data_dict[colname] = np.full(n_frames, np.nan, dtype=np.float32)
        for fi, frame in enumerate(frames):
            for ai, animal1 in enumerate(animals1):
                aj = aligned_indices[fi, ai]
                for ki, kp in enumerate(keypoints):
                    for coord in ["x", "y", "likelihood"]:
                        src_col = f"{animals2[aj]}.{kp}.{coord}"
                        dst_col = f"{animal1}.{kp}.{coord}"
                        if src_col in df2.columns:
                            new_data_dict[dst_col][fi] = df2.loc[frame, src_col]
        new_df2 = pd.DataFrame(new_data_dict, index=frames)
        extra_cols = [c for c in df2.columns if not any(bp in c for bp in keypoints)]
        if extra_cols:
            new_df2 = pd.concat([new_df2, df2.loc[frames, extra_cols]], axis=1)
        new_df2 = new_df2[
            [c for c in df2.columns if c in new_df2.columns]
            + [c for c in new_df2.columns if c not in df2.columns]
        ]
        return new_df2

    def align_ids_by_keypoints(
        self,
        keypoints: list[str],
        views: list[str] | None = None,
    ) -> "TrackingMV":
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        all_views = list(self.views.keys())
        if views is not None:
            if len(views) != 2:
                raise ValueError(
                    "views argument must be a list or tuple of two view names."
                )
            v1, v2 = views
            if v1 not in self.views or v2 not in self.views:
                raise ValueError(
                    f"Specified views {views} not found in TrackingMV object."
                )
        else:
            if len(all_views) < 2:
                raise ValueError("TrackingMV must have at least two views to align.")
            v1, v2 = all_views[:2]

        df1, df2 = self.views[v1].data, self.views[v2].data
        animals1, _ = self._get_animals_and_bodyparts(df1)
        animals2, _ = self._get_animals_and_bodyparts(df2)
        if len(animals1) != len(animals2):
            raise ValueError("Number of animals differs between views.")
        n_animals = len(animals1)
        frames = df1.index.intersection(df2.index)
        n_frames = len(frames)

        coords1 = self._extract_coords(df1, animals1, keypoints, frames)
        coords2 = self._extract_coords(df2, animals2, keypoints, frames)
        cost_matrices = self._compute_cost_matrices(coords1, coords2)

        aligned_indices = np.zeros((n_frames, n_animals), dtype=int)
        for fi in range(n_frames):
            row_ind, col_ind = linear_sum_assignment(cost_matrices[fi])
            aligned_indices[fi] = col_ind

        new_df2 = self._reorder_view2_data(
            df2, animals1, animals2, keypoints, aligned_indices, frames
        )
        new_views = self.views.copy()
        new_views[v1] = Tracking(
            df1.loc[frames], self.views[v1].meta.copy(), self.views[v1].handle
        )
        new_views[v2] = Tracking(
            new_df2, self.views[v2].meta.copy(), self.views[v2].handle
        )
        return TrackingMV(new_views, self.calibration, self.handle)

    def plot(
        self,
        trajectories=None,
        static=None,
        lines=None,
        dims=("x", "y"),
        ax=None,
        title=None,
        show=True,
    ):
        n = len(self.views)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        axes = axes[0]  # flatten to 1D
        for i, (view, track) in enumerate(self.views.items()):
            track.plot(
                trajectories=trajectories,
                static=static,
                lines=lines,
                dims=dims,
                ax=axes[i],
                title=view,
                show=False,
            )
        if title is None:
            title = self.handle
        fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for view, track in self.views.items():
                method = getattr(track, name)
                results[view] = method(*args, **kwargs)
            return results

        return batch_method

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.views)} views: {', '.join(self.views.keys())}>"
