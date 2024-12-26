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
max_id: int

app = fastapi.FastAPI()
model = ultralytics.YOLO(MODEL_PATH)


@app.post("/infer", response_class=fastapi.responses.FileResponse)
async def infer(video_file: fastapi.UploadFile):
    global original_path, boxes_list, max_id

    original_path = ORIGINAL_PATH_NO_SUFFIX.with_suffix(pathlib.Path(video_file.filename).suffix)
    with open(original_path, "wb") as fout:
        fout.write(await video_file.read())

    results = model.track(original_path, tracker=BOTSORT_CONFIG_PATH)
    boxes_list = [res.boxes for res in results]
    max_id = max(max(boxes.id.int().tolist()) for boxes in boxes_list if boxes.is_track)

    await video.draw_bboxes(original_path, MODIFIED_PATH, boxes_list, [video.PlayerParams() for _ in range(max_id)])
    return fastapi.responses.FileResponse(path=MODIFIED_PATH, headers={"max_id": str(max_id)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8500, reload=True)
