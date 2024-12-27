import pathlib
import zipfile

import fastapi
import ultralytics.engine.results
import uvicorn

import video


MODEL_PATH = pathlib.Path("models/baseline.pt")
BOTSORT_CONFIG_PATH = pathlib.Path("config/botsort.yaml")

RESULTS_DIR = pathlib.Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_PATH_NO_SUFFIX = RESULTS_DIR / "original"
ANNOTATED_PATH = RESULTS_DIR / "annotated.mp4"
ZIP_PATH = RESULTS_DIR / "archive.zip"

app = fastapi.FastAPI()
model = ultralytics.YOLO(MODEL_PATH)


@app.post("/infer", response_class=fastapi.responses.FileResponse)
async def infer(video_file: fastapi.UploadFile):
    original_path = ORIGINAL_PATH_NO_SUFFIX.with_suffix(
        pathlib.Path(video_file.filename).suffix
    )
    with open(original_path, "wb") as fout:
        fout.write(await video_file.read())

    results = model.track(original_path, tracker=BOTSORT_CONFIG_PATH)
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

    return fastapi.responses.FileResponse(
        path=ZIP_PATH,
        headers={
            "player_ids": player_ids_header,
            "player_times": player_times_header,
        },
    )


@app.post("/make_video", response_class=fastapi.responses.FileResponse)
async def make_video(player_params: dict[int, video.PlayerParams]):
    if not hasattr(app.state, "player_ids"):
        raise fastapi.HTTPException(
            fastapi.status.HTTP_400_BAD_REQUEST, "/infer must be called first"
        )
    if set(player_params.keys()) != app.state.player_ids:
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
    return ANNOTATED_PATH


@app.post("/make_focused_video", response_class=fastapi.responses.FileResponse)
async def make_focused_video(player_id: int):
    if not hasattr(app.state, "player_ids"):
        raise fastapi.HTTPException(
            fastapi.status.HTTP_400_BAD_REQUEST, "/infer must be called first"
        )
    if player_id not in app.state.player_ids:
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
    return ANNOTATED_PATH


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8500, reload=True)
