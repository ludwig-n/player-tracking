import dataclasses
import io
import logging.handlers
import os
import pathlib
import zipfile

import requests
import streamlit as st


@dataclasses.dataclass
class PlayerParams:
    label: str
    draw: bool


SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8500")
REQUEST_TIMEOUT = 60
LOG_PATH = pathlib.Path("logs/client.log")
LOG_PATH.parent.mkdir(exist_ok=True)
VIDEO_FORMATS = [
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpeg4",
    "mpg",
    "ts",
    "webm",
    "wmv",
]


def to_video_time(n_secs):
    """Converts `n_secs` seconds to a video timestamp (e.g. 63 -> 1:03)."""
    return f"{n_secs // 60}:{n_secs % 60:02}"


def try_request(method, url, **kwargs):
    """
    Tries to send a request to the server.
    Returns None if an error occurs.
    """
    logging.info(f"Calling {method} {url}, kwargs: {kwargs}")
    try:
        response = requests.request(
            method=method, url=url, timeout=REQUEST_TIMEOUT, **kwargs
        )
        response.raise_for_status()
    except requests.Timeout as e:
        st.error("Server timed out")
        logging.error(f"Server timed out: {e}")
        return None
    except requests.ConnectionError as e:
        st.error("Could not connect to server")
        logging.error(f"Connection error: {e}")
        return None
    except requests.HTTPError as e:
        st.error("Something went wrong after server request")
        logging.error(f"Server returned an error code: {e}")
        return None
    except requests.RequestException as e:
        st.error("Something went wrong after server request")
        logging.error(f"Unknown request error: {e}")
        return None
    logging.info("Got response successfully")
    return response


def get_models():
    """Gets and stores the list of detectors and trackers from the server."""
    st.session_state.models = try_request(
        method="GET", url=f"{SERVER_URL}/get_models"
    ).json()


def infer(video_file, detector, tracker):
    """Infers the model on a new video file and resets the interface."""

    # Clear session state except for models lists (these don't change)
    models = st.session_state.models
    st.session_state.clear()
    st.session_state.models = models

    response = try_request(
        method="POST",
        url=f"{SERVER_URL}/infer",
        files={"video_file": video_file},
        params={"detector": detector, "tracker": tracker},
    )
    if response is None:
        return

    st.session_state.player_ids = [
        int(x) for x in response.headers["player_ids"].split(",")
    ]

    for pid, time_range in zip(
        st.session_state.player_ids,
        response.headers["player_times"].split(","),
    ):
        start_secs, end_secs = time_range.split("-")
        start_secs_vid = to_video_time(int(start_secs))
        end_secs_vid = to_video_time(int(end_secs))
        st.session_state[f"times{pid}"] = f"{start_secs_vid} - {end_secs_vid}"

    with zipfile.ZipFile(io.BytesIO(response.content), "r") as archive:
        st.session_state.video = archive.read("annotated.mp4")
        for pid in st.session_state.player_ids:
            st.session_state[f"image{pid}"] = archive.read(f"images/{pid}.jpg")

    set_all_checkboxes(True)
    set_all_inputs("")


def regenerate_video(params_dict):
    """Regenerates the annotated video based on the current params."""
    del st.session_state.video
    response = try_request(
        method="POST",
        url=f"{SERVER_URL}/make_video",
        json={
            pid: {"label": params.label, "draw": params.draw}
            for pid, params in params_dict.items()
        },
    )
    if response is not None:
        st.session_state.video = response.content


def get_focused_video(player_id):
    """Generates a video focused on a specific player."""
    response = try_request(
        method="POST",
        url=f"{SERVER_URL}/make_focused_video",
        params={"player_id": player_id},
    )
    if response is not None:
        st.session_state[f"focused{player_id}"] = response.content


def set_all_inputs(value):
    """Sets all text inputs' values to `value`."""
    for pid in st.session_state.player_ids:
        st.session_state[f"label{pid}"] = value


def set_all_checkboxes(value):
    """Sets all checkboxes' states to `value` (bool)."""
    for pid in st.session_state.player_ids:
        st.session_state[f"draw{pid}"] = value


def build_app():
    """Builds the client interface."""

    st.title("Player Tracking")

    video_file = st.file_uploader("Upload video", VIDEO_FORMATS)

    if "models" not in st.session_state:
        get_models()

    col_det, col_trk = st.columns(2)

    detectors = st.session_state.models["detectors"]
    det_slugs = [det["slug"] for det in detectors]
    selected_detector = col_det.selectbox(
        "Detector:",
        options=det_slugs,
        format_func=lambda slug: detectors[det_slugs.index(slug)]["ui_name"],
    )

    trackers = st.session_state.models["trackers"]
    trk_slugs = [trk["slug"] for trk in trackers]
    selected_tracker = col_trk.selectbox(
        "Tracker:",
        options=trk_slugs,
        format_func=lambda slug: trackers[trk_slugs.index(slug)]["ui_name"],
    )

    st.columns([3, 2, 3])[1].button(
        "Track!",
        use_container_width=True,
        type="primary",
        on_click=infer,
        args=(video_file, selected_detector, selected_tracker),
    )

    if "video" in st.session_state:
        st.video(st.session_state.video)

        params_dict = {}
        st.columns([3, 2, 3])[1].button(
            "Regenerate video",
            use_container_width=True,
            type="primary",
            on_click=regenerate_video,
            args=(params_dict,),
        )

        st.header("Detected players")

        button_columns = st.columns(4)
        button_columns[1].button(
            "Deselect all",
            use_container_width=True,
            on_click=set_all_checkboxes,
            args=(False,),
        )
        button_columns[2].button(
            "Select all",
            use_container_width=True,
            on_click=set_all_checkboxes,
            args=(True,),
        )

        grid = [
            st.columns(3)
            for _ in range(len(st.session_state.player_ids) // 3 + 1)
        ]
        for i, pid in enumerate(st.session_state.player_ids, start=1):
            with grid[(i - 1) // 3][(i - 1) % 3].expander(
                f"Player id{pid}", expanded=True
            ):
                st.columns([1, 2, 1])[1].image(
                    st.session_state[f"image{pid}"],
                    caption=st.session_state[f"times{pid}"],
                    use_container_width=True,
                )

                label = st.text_input(
                    "Custom label",
                    placeholder="Custom label",
                    label_visibility="collapsed",
                    key=f"label{pid}",
                )
                draw = st.checkbox("Highlight in main video", key=f"draw{pid}")
                params_dict[pid] = PlayerParams(label=label, draw=draw)

                if f"focused{pid}" in st.session_state:
                    st.video(st.session_state[f"focused{pid}"])
                else:
                    st.button(
                        "Get focused video",
                        on_click=get_focused_video,
                        args=(pid,),
                        key=f"focusbtn{pid}",
                    )


if __name__ == "__main__":
    log_handler = logging.handlers.TimedRotatingFileHandler(
        filename=LOG_PATH, when="D", backupCount=7
    )

    # Does nothing if any handlers are already set up
    logging.basicConfig(handlers=[log_handler], level=logging.INFO)

    build_app()
