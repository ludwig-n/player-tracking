import logging.handlers
import pathlib
import zipfile

import fastapi
import uvicorn

import tracking
import video


RESULTS_DIR = pathlib.Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = pathlib.Path("logs/server.log")
LOG_PATH.parent.mkdir(exist_ok=True)

ORIGINAL_PATH_NO_SUFFIX = RESULTS_DIR / "original"
ANNOTATED_PATH = RESULTS_DIR / "annotated.mp4"
ZIP_PATH = RESULTS_DIR / "archive.zip"

app = fastapi.FastAPI()


@app.post("/infer", response_class=fastapi.responses.FileResponse)
async def infer(
    video_file: fastapi.UploadFile,
    detector_name: str = "march-best",
    tracker_name: str = "raft",
):
    """
    Infers the detector `detector_name` and tracker `tracker_name`
    on a video file.

    Returns a zip file with:
      - a new video with added boxes and labels highlighting the players,
      - an image of each detected player from their first detection.

    The response headers contain:
      - ids of all detected players (player_ids),
      - time ranges when each player was present in the video (player_times).
    """
    logging.info("Received POST /infer")

    original_path = ORIGINAL_PATH_NO_SUFFIX.with_suffix(
        pathlib.Path(video_file.filename).suffix
    )
    with open(original_path, "wb") as fout:
        fout.write(await video_file.read())

    if detector_name not in tracking.DETECTORS:
        logging.warning(
            f"detector {detector_name} not found in available detectors "
            f"{list(tracking.DETECTORS.keys())}"
        )
        raise fastapi.HTTPException(
            fastapi.status.HTTP_404_NOT_FOUND,
            f"detector {detector_name} not found",
        )

    if tracker_name not in tracking.TRACKERS:
        logging.warning(
            f"tracker {tracker_name} not found in available trackers "
            f"{list(tracking.TRACKERS.keys())}"
        )
        raise fastapi.HTTPException(
            fastapi.status.HTTP_404_NOT_FOUND,
            f"tracker {tracker_name} not found",
        )

    results = tracking.track(
        source=original_path,
        detector=tracking.DETECTORS[detector_name],
        tracker=tracking.TRACKERS[tracker_name],
    )
    boxes_list = [res.boxes for res in results]

    start_time, end_time = await video.get_player_times(
        original_path, boxes_list
    )

    player_ids = set(start_time.keys())
    player_ids_sorted = sorted(player_ids)
    player_ids_header = ",".join(str(pid) for pid in player_ids_sorted)
    player_times_header = ",".join(
        f"{start_time[pid]}-{end_time[pid]}" for pid in player_ids_sorted
    )

    await video.draw_bboxes(
        original_path,
        ANNOTATED_PATH,
        boxes_list,
        {pid: video.PlayerParams() for pid in player_ids},
    )
    await video.save_player_images(original_path, IMAGES_DIR, boxes_list)

    archive = zipfile.ZipFile(ZIP_PATH, "w")
    archive.write(ANNOTATED_PATH, ANNOTATED_PATH.relative_to(RESULTS_DIR))
    for file in IMAGES_DIR.iterdir():
        archive.write(file, file.relative_to(RESULTS_DIR))

    app.state.original_path = original_path
    app.state.boxes_list = boxes_list
    app.state.player_ids = player_ids

    logging.info(f"/infer done, returning {ZIP_PATH}")

    return fastapi.responses.FileResponse(
        path=ZIP_PATH,
        headers={
            "player_ids": player_ids_header,
            "player_times": player_times_header,
        },
    )


@app.post("/make_video", response_class=fastapi.responses.FileResponse)
async def make_video(player_params: dict[int, video.PlayerParams]):
    """
    Generates a video with bounding boxes and labels like /infer,
    but using custom visualization parameters.
    """
    logging.info(f"Received POST /make_video, player_params: {player_params}")

    if not hasattr(app.state, "player_ids"):
        logging.warning("/make_video called without an /infer")
        raise fastapi.HTTPException(
            fastapi.status.HTTP_400_BAD_REQUEST, "/infer must be called first"
        )

    if set(player_params.keys()) != app.state.player_ids:
        logging.warning(
            "player_params has wrong keys: "
            f"expected {sorted(app.state.player_ids)}, "
            f"got {sorted(player_params.keys())}"
        )
        raise fastapi.HTTPException(
            fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
            "player_params must have exactly one element for each player id",
        )

    await video.draw_bboxes(
        app.state.original_path,
        ANNOTATED_PATH,
        app.state.boxes_list,
        player_params,
    )
    logging.info(f"/make_video done, returning {ANNOTATED_PATH}")
    return ANNOTATED_PATH


@app.post("/make_focused_video", response_class=fastapi.responses.FileResponse)
async def make_focused_video(player_id: int):
    """Generates a video focused on a specific player's movements."""
    logging.info(f"Received POST /make_focused_video, player_id: {player_id}")

    if not hasattr(app.state, "player_ids"):
        logging.warning("/make_focused_video called without an /infer")
        raise fastapi.HTTPException(
            fastapi.status.HTTP_400_BAD_REQUEST, "/infer must be called first"
        )

    if player_id not in app.state.player_ids:
        logging.warning(
            f"player_id {player_id} not found in player ids "
            f"{sorted(app.state.player_ids)}"
        )
        raise fastapi.HTTPException(
            fastapi.status.HTTP_404_NOT_FOUND,
            f"player id {player_id} not found",
        )

    await video.crop_to_player(
        app.state.original_path,
        ANNOTATED_PATH,
        app.state.boxes_list,
        player_id,
    )
    logging.info(f"/make_focused_video done, returning {ANNOTATED_PATH}")
    return ANNOTATED_PATH


if __name__ == "__main__":
    log_handler = logging.handlers.TimedRotatingFileHandler(
        filename=LOG_PATH, when="D", backupCount=7
    )
    logging.basicConfig(handlers=[log_handler], level=logging.INFO)
    uvicorn.run("main:app", host="0.0.0.0", port=8500)
