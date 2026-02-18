from __future__ import annotations

from typing import Literal

from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.tracking.tracking_collection_batch_mixin import TrackingCollectionBatchMixin
from py3r.behaviour.util.base_collection import BaseCollection

class TrackingCollection(BaseCollection, TrackingCollectionBatchMixin):
    @property
    def each(self) -> Tracking: ...
    @property
    def tracking_dict(self) -> dict[str, Tracking]: ...
    @classmethod
    def from_mapping(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        tracking_loader,
        tracking_cls=Tracking,
        **loader_kwargs,
    ) -> TrackingCollection: ...
    @classmethod
    def from_dlc(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls=Tracking,
    ) -> TrackingCollection: ...
    @classmethod
    def from_yolo3r(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls=Tracking,
    ) -> TrackingCollection: ...
    @classmethod
    def from_dlcma(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls=Tracking,
    ) -> TrackingCollection: ...
    @classmethod
    def from_folder(
        cls,
        folder_path: str,
        *,
        tracking_loader,
        tracking_cls=Tracking,
        file_ext: str = ".csv",
        recursive: bool = False,
        **loader_kwargs,
    ) -> TrackingCollection: ...
    @classmethod
    def from_yolo3r_folder(
        cls,
        folder_path: str,
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls: type = Tracking,
    ) -> TrackingCollection: ...
    @classmethod
    def from_dlc_folder(
        cls,
        folder_path: str,
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls: type = Tracking,
    ) -> TrackingCollection: ...
    @classmethod
    def from_dlcma_folder(
        cls,
        folder_path: str,
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls: type = Tracking,
    ) -> TrackingCollection: ...
    @classmethod
    def concat(
        cls,
        collections: list[TrackingCollection],
        *,
        reindex: Literal["rezero", "follow_previous", "keep_original"] = "follow_previous",
    ) -> TrackingCollection: ...
