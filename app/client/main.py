import dataclasses
import io
import os
import zipfile

import requests
import streamlit as st


@dataclasses.dataclass
class PlayerParams:
    label: str
    draw: bool


SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8500")
REQUEST_TIMEOUT = 60
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
    return f"{n_secs // 60}:{n_secs % 60:02}"


def infer(video_file):
    st.session_state.clear()
    response = requests.post(
        f"{SERVER_URL}/infer",
        files={"video_file": video_file},
        timeout=REQUEST_TIMEOUT,
    )
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
    del st.session_state.video
    st.session_state.video = requests.post(
        f"{SERVER_URL}/make_video",
        json={
            pid: {"label": params.label, "draw": params.draw}
            for pid, params in params_dict.items()
        },
        timeout=REQUEST_TIMEOUT,
    ).content


def get_focused_video(player_id):
    st.session_state[f"focused{player_id}"] = requests.post(
        f"{SERVER_URL}/make_focused_video",
        params={"player_id": player_id},
        timeout=REQUEST_TIMEOUT,
    ).content


def set_all_inputs(value):
    for pid in st.session_state.player_ids:
        st.session_state[f"label{pid}"] = value


def set_all_checkboxes(value):
    for pid in st.session_state.player_ids:
        st.session_state[f"draw{pid}"] = value


def build_app():
    st.title("Player Tracking")

    video_file = st.file_uploader("Upload video", VIDEO_FORMATS)

    st.columns([3, 2, 3])[1].button(
        "Track!",
        use_container_width=True,
        type="primary",
        on_click=infer,
        args=(video_file,),
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


build_app()
