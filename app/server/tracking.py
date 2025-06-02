import dataclasses

import ultralytics.engine.model


@dataclasses.dataclass
class Detector:
    path: str
    ui_name: str
    cls: type[ultralytics.engine.model.Model]

    def track(self, source: str, tracker: str):
        return self.cls(self.path).track(source=source, tracker=tracker)


DETECTORS = {
    "march-best": Detector(
        path="models/march-best.pt",
        ui_name="best yolo11l",
        cls=ultralytics.YOLO,
    ),
    "march-best-s": Detector(
        path="models/march-best-s.pt",
        ui_name="best yolo11s",
        cls=ultralytics.YOLO,
    ),
    "baseline": Detector(
        path="models/baseline.pt",
        ui_name="baseline yolo11s",
        cls=ultralytics.YOLO,
    ),
}
