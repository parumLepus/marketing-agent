import streamlit as st
from agent import build_agent
import base64
from PIL import Image
import io
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

llm = ChatOpenAI(model = "gpt-5.4-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))


st.set_page_config(
    page_title="Your helpful Marketing Friend",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Marketing AI Agent")
st.caption("Ask me to anything. I can analyse your marketing data, create content plan, build a strategy and more.")

st.markdown("""
<style>

[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 50px;
    padding: 0;
    border: none;
    background: transparent;
}

[data-testid="stFileUploader"] button {
    border-radius: 12px;
}

[data-testid="stFileUploaderFile"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

if "agent" not in st.session_state:
    with st.spinner("Starting up agent..."):
        st.session_state.agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hi! I'm your Marketing AI Agent. "
            "I can analyse your campaign data, research trends, "
            "and build you a marketing or content plan.\n\n"
            "To get started — **what are your main marketing goals right now?**"
        )
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    MAX_FILE_SIZE_MB = 2

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error("Please upload an image smaller than 2 MB.")
        st.stop()

    img = Image.open(uploaded_file)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    st.session_state.uploaded_image = image_data

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            background:#262730;
            padding:10px 14px;
            border-radius:12px;
            width:fit-content;
            margin-bottom:12px;
        ">
            📎 {uploaded_file.name}
        </div>
        """,
        unsafe_allow_html=True
    )

user_input = st.chat_input(
    "Describe your goals or ask for a marketing plan..."
)

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and pulling data..."):
            try:
                image_data = st.session_state.get("uploaded_image", None)

                input_data = {
                    "input": user_input
                }

                if image_data:
                    input_data["image"] = f"data:image/jpeg;base64,{image_data}"
                response = st.session_state.agent.invoke(
                    input_data,
                    config={"configurable": {"session_id": "default"}}
                )

                answer = response["output"]

            except Exception as e:
                answer = f"Something went wrong: {e}"

            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })