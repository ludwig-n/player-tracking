import dataclasses
import pathlib

import bbox_visualizer as bbox
import moviepy.video.io.ImageSequenceClip
import pydantic
import ultralytics.engine.results


class PlayerParams(pydantic.BaseModel):
    label: str = ""
    draw: bool = True


@dataclasses.dataclass
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int


def fit_interval(left: int, right: int, min_lim: int, max_lim: int) -> tuple[int, int]:
    if left < min_lim:
        add = min_lim - left
    elif right > max_lim:
        add = max_lim - right
    else:
        add = 0
    return left + add, right + add


async def draw_bboxes(
    in_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    boxes_list: list[ultralytics.engine.results.Boxes],
    params_dict: dict[int, PlayerParams]
) -> None:
    clip = moviepy.VideoFileClip(in_path, audio=False)

    out_frames = []
    for frame, boxes in zip(clip.iter_frames(), boxes_list):
        if boxes.is_track:
            for xyxy, id in zip(boxes.xyxy.round().int().tolist(), boxes.id.int().tolist()):
                params = params_dict[id]
                if not params.draw:
                    continue
                label = params.label if params.label else f"id{id}"

                frame = bbox.draw_rectangle(frame, xyxy)
                frame = bbox.add_label(frame, label, xyxy)
        out_frames.append(frame)

    moviepy.video.io.ImageSequenceClip.ImageSequenceClip(out_frames, fps=clip.fps).write_videofile(out_path)


async def crop_to_player(
    in_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    boxes_list: list[ultralytics.engine.results.Boxes],
    player_id: int
) -> None:
    rects = []
    for boxes in boxes_list:
        if boxes.is_track:
            xyxy = boxes.xyxy[boxes.id == player_id]
            if len(xyxy) > 0:
                rects.append(Rect(*xyxy.round().int().flatten().tolist()))

    max_x = max(rect.x2 - rect.x1 for rect in rects)
    max_y = max(rect.y2 - rect.y1 for rect in rects)

    clip = moviepy.VideoFileClip(in_path, audio=False)
    out_frames = []
    for frame, rect in zip(clip.iter_frames(), rects):
        extra_x = max_x - (rect.x2 - rect.x1)
        extra_y = max_y - (rect.y2 - rect.y1)

        rect.x1 -= extra_x // 2
        rect.x2 += extra_x // 2 + extra_x % 2
        rect.x1, rect.x2 = fit_interval(rect.x1, rect.x2, 0, frame.shape[1])

        rect.y1 -= extra_y // 2
        rect.y2 += extra_y // 2 + extra_y % 2
        rect.y1, rect.y2 = fit_interval(rect.y1, rect.y2, 0, frame.shape[0])

        out_frames.append(frame[rect.y1:rect.y2, rect.x1:rect.x2])

    moviepy.video.io.ImageSequenceClip.ImageSequenceClip(out_frames, fps=clip.fps).write_videofile(out_path)
