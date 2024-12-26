import pathlib

import bbox_visualizer as bbox
import moviepy.video.io.ImageSequenceClip
import pydantic
import ultralytics.engine.results


class PlayerParams(pydantic.BaseModel):
    label: str | None = None
    draw: bool = True


async def draw_bboxes(
    in_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    boxes_list: list[ultralytics.engine.results.Boxes],
    params_list: list[PlayerParams]
) -> None:
    clip = moviepy.VideoFileClip(in_path, audio=False)

    out_frames = []
    for frame, boxes in zip(clip.iter_frames(), boxes_list):
        if boxes.is_track:
            for xyxy, id in zip(boxes.xyxy.round().int().tolist(), boxes.id.int()):
                params = params_list[id - 1]
                if not params.draw:
                    continue
                label = params.label if params.label is not None else f"id{id}"

                frame = bbox.draw_rectangle(frame, xyxy)
                frame = bbox.add_label(frame, label, xyxy)
        out_frames.append(frame)

    moviepy.video.io.ImageSequenceClip.ImageSequenceClip(out_frames, fps=clip.fps).write_videofile(out_path)


async def crop_to_player(
    in_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    boxes_list: list[ultralytics.engine.results.Boxes],
    id: int
) -> None:
    raise NotImplemented
