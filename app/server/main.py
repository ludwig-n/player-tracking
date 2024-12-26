import pathlib

import fastapi
import ultralytics.engine.results
import uvicorn

import video


MODEL_PATH = pathlib.Path("models/baseline.pt")
BOTSORT_CONFIG_PATH = pathlib.Path("config/botsort.yaml")

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
ORIGINAL_PATH_NO_SUFFIX = RESULTS_DIR / "original"
MODIFIED_PATH = RESULTS_DIR / "modified.mp4"

original_path: str
boxes_list: list[ultralytics.engine.results.Boxes]
player_ids: set[int]

app = fastapi.FastAPI()
model = ultralytics.YOLO(MODEL_PATH)


@app.post("/infer", response_class=fastapi.responses.FileResponse)
async def infer(video_file: fastapi.UploadFile):
    global original_path, boxes_list, player_ids

    original_path = ORIGINAL_PATH_NO_SUFFIX.with_suffix(pathlib.Path(video_file.filename).suffix)
    with open(original_path, "wb") as fout:
        fout.write(await video_file.read())

    results = model.track(original_path, tracker=BOTSORT_CONFIG_PATH)
    boxes_list = [res.boxes for res in results]

    player_ids = set()
    for boxes in boxes_list:
        if boxes.is_track:
            player_ids.update(boxes.id.int().tolist())
    player_ids_header = ",".join(str(pid) for pid in sorted(player_ids))

    await video.draw_bboxes(original_path, MODIFIED_PATH, boxes_list, {pid: video.PlayerParams() for pid in player_ids})
    return fastapi.responses.FileResponse(path=MODIFIED_PATH, headers={"player_ids": player_ids_header})


@app.post("/make_video", response_class=fastapi.responses.FileResponse)
async def make_video(player_params: dict[int, video.PlayerParams]):
    if set(player_params.keys()) != player_ids:
        raise fastapi.HTTPException(
            fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"player_params must have exactly one element for each player id"
        )
    await video.draw_bboxes(original_path, MODIFIED_PATH, boxes_list, player_params)
    return MODIFIED_PATH


@app.post("/make_focused_video", response_class=fastapi.responses.FileResponse)
async def make_focused_video(player_id: int):
    if player_id not in player_ids:
        raise fastapi.HTTPException(fastapi.status.HTTP_404_NOT_FOUND, f"player id {player_id} not found")
    await video.crop_to_player(original_path, MODIFIED_PATH, boxes_list, player_id)
    return MODIFIED_PATH


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8500, reload=True)
