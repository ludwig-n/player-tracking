import requests
import streamlit as st


class PlayerParams:
    label: str
    draw: bool


def infer():
    st.session_state.clear()
    response = requests.post(f"{SERVER_URL}/infer", files={"video_file": video_file})
    st.session_state.video = response.content
    st.session_state.player_ids = [int(x) for x in response.headers["player_ids"].split(",")]
    set_all_checkboxes(True)


def regenerate_video():
    del st.session_state.video
    st.session_state.video = requests.post(
        f"{SERVER_URL}/make_video",
        json={pid: {"label": params.label, "draw": params.draw} for pid, params in params_dict.items()}
    ).content


def get_focused_video(player_id):
    st.session_state[f"focused{player_id}"] = requests.post(
        f"{SERVER_URL}/make_focused_video",
        params={"player_id": player_id}
    ).content


def set_all_checkboxes(value):
    for pid in st.session_state.player_ids:
        st.session_state[f"draw{pid}"] = value


SERVER_URL = "http://localhost:8500"
VIDEO_FORMATS = ["asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpeg4", "mpg", "ts", "webm", "wmv"]


st.title("Player Tracking")

video_file = st.file_uploader("Upload video", VIDEO_FORMATS)

st.columns([3, 2, 3])[1].button("Track!", use_container_width=True, type="primary", on_click=infer)

if "video" in st.session_state:
    st.video(st.session_state.video)
    st.columns([3, 2, 3])[1].button("Regenerate video", use_container_width=True, type="primary", on_click=regenerate_video)

    st.header("Detected players")

    button_columns = st.columns(4)
    button_columns[1].button("Deselect all", use_container_width=True, on_click=set_all_checkboxes, args=(False,))
    button_columns[2].button("Select all", use_container_width=True, on_click=set_all_checkboxes, args=(True,))

    params_dict = {pid: PlayerParams() for pid in st.session_state.player_ids}
    for i, (pid, params) in enumerate(params_dict.items(), start=1):
        if (i - 1) % 3 == 0:
            columns = st.columns(3)

        with columns[(i - 1) % 3], st.expander(f"Player id{pid}", expanded=True):
            params.label = st.text_input(
                "Custom label", placeholder="Custom label", label_visibility="collapsed", key=f"label{pid}"
            )
            params.draw = st.checkbox("Highlight in main video", value=True, key=f"draw{pid}")
            if f"focused{pid}" in st.session_state:
                st.video(st.session_state[f"focused{pid}"])
            else:
                st.button("Get focused video", on_click=get_focused_video, args=(pid,), key=f"focusbtn{pid}")
