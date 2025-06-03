import dataclasses
import typing as tp

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import ultralytics.engine.model
import ultralytics.engine.results
import ultralytics.trackers.utils.gmc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclasses.dataclass
class Detector:
    weights_path: str
    ui_name: str
    model_class: type[ultralytics.engine.model.Model]


class GMC(tp.Protocol):
    """
    The minimal interface required to replace
    the stock `ultralytics.trackers.utils.gmc.GMC` class
    to implement a new GMC method within BoT-SORT.
    """

    def apply(self, raw_frame: np.ndarray, detections: list) -> np.ndarray: ...
    def reset_params(self) -> None: ...


class RaftGMC:
    def __init__(
        self,
        model_size: str = "small",
        image_size: int = 128,
        num_flow_updates: int = 1,
    ) -> None:
        if model_size == "small":
            wgts = torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT
            self.model = torchvision.models.optical_flow.raft_small(wgts)
        elif model_size == "large":
            wgts = torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT
            self.model = torchvision.models.optical_flow.raft_large(wgts)
        else:
            raise ValueError

        self.model = self.model.to(DEVICE).eval()
        self.transforms = wgts.transforms()
        self.image_size = image_size
        self.num_flow_updates = num_flow_updates
        self.last_frame = None

    def apply(
        self, raw_frame: np.ndarray, detections: list = None
    ) -> np.ndarray:
        """
        `raw_frame`: frame with shape [H, W, C].
        `detections`: unused for RAFT.
        Returns a 2x3 homography matrix.
        """
        raw_frame = torch.tensor(raw_frame)
        raw_frame = raw_frame.permute(2, 0, 1)[None, :]  # [1, C, H, W]
        raw_frame = raw_frame[:, :, :, 10:-10]

        scale_factor = raw_frame.shape[2] / self.image_size
        raw_frame = F.resize(
            raw_frame, size=self.image_size, antialias=False
        )  # [1, C, h, w]

        if self.last_frame is None:
            self.last_frame = raw_frame
            return np.eye(2, 3)

        frame1, frame2 = self.transforms(self.last_frame, raw_frame)
        flows = self.model(  # list[num_flow_updates] of [1, 2, h/8, w/8]
            frame1.to(DEVICE),
            frame2.to(DEVICE),
            num_flow_updates=self.num_flow_updates,
        )

        medians = flows[-1].reshape(2, -1).median(dim=1).values  # [2]
        medians = medians.cpu().numpy()
        hom = np.eye(2, 3)
        hom[:, 2] = medians * scale_factor

        self.last_frame = raw_frame
        return hom

    def reset_params(self) -> None:
        self.last_frame = None


def scale_raft_flow(flow: torch.Tensor, **kwargs) -> torch.Tensor:
    return flow * 8


@dataclasses.dataclass
class Tracker:
    cfg_path: str
    ui_name: str
    gmc_class: type[GMC]
    gmc_args: dict[str, tp.Any]


def track(
    source: str, detector: Detector, tracker: Tracker
) -> list[ultralytics.engine.results.Results]:
    model = detector.model_class(detector.weights_path)

    def gmc_patch(method: str) -> GMC:
        """Deliberately ignores `method` in favor of our GMC class."""
        return tracker.gmc_class(**tracker.gmc_args)

    ultralytics.trackers.bot_sort.GMC = gmc_patch
    torchvision.models.optical_flow.raft.upsample_flow = scale_raft_flow

    return model.track(source=source, tracker=tracker.cfg_path)


DETECTORS = {
    "march-best": Detector(
        weights_path="models/march-best.pt",
        ui_name="best yolo11l",
        model_class=ultralytics.YOLO,
    ),
    "march-best-s": Detector(
        weights_path="models/march-best-s.pt",
        ui_name="best yolo11s",
        model_class=ultralytics.YOLO,
    ),
    "baseline": Detector(
        weights_path="models/baseline.pt",
        ui_name="baseline yolo11s",
        model_class=ultralytics.YOLO,
    ),
}

TRACKERS = {
    "raft": Tracker(
        cfg_path="config/botsort.yaml",
        ui_name="RAFT",
        gmc_class=RaftGMC,
        gmc_args={},
    ),
}
TRACKERS |= {
    f"spofl-{downscale}x": Tracker(
        cfg_path="config/botsort.yaml",
        ui_name=f"Sparse Optical Flow ({downscale}x downscale)",
        gmc_class=ultralytics.trackers.utils.gmc.GMC,
        gmc_args=dict(method="sparseOptFlow", downscale=downscale),
    )
    for downscale in [2, 8, 10, 16, 20]
}
