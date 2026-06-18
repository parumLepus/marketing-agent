import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import streamlit.components.v1 as components
import secrets
import base64
from PIL import Image
import io
import json
import urllib.parse
import requests as http_requests
from google.oauth2.credentials import Credentials
from agent import build_agent
from tools.image_generation_tool import get_last_generated_image
from tools.notion_tool import exchange_notion_code_for_token, find_accessible_page_id
from langchain_core.messages import HumanMessage


st.set_page_config(
    page_title="Your helpful Marketing Friend",
    page_icon="📊",
    layout="centered"
)

REFRESH_FLAG_PARAM = "_refreshed"


def _handle_manual_refresh():
    qp = st.query_params

    if qp.get(REFRESH_FLAG_PARAM) == "1":
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.query_params.clear()
        return

    if "code" in qp:
        return

    components.html(
        f"""
        <script>
        const nav = performance.getEntriesByType("navigation")[0];
        const isReload = nav && nav.type === "reload";
        const url = new URL(window.location.href);
        const alreadyFlagged = url.searchParams.get("{REFRESH_FLAG_PARAM}") === "1";
        if (isReload && !alreadyFlagged) {{
            url.searchParams.set("{REFRESH_FLAG_PARAM}", "1");
            window.location.replace(url.toString());
        }}
        </script>
        """,
        height=0,
        width=0,
    )


_handle_manual_refresh()

st.title("📊 Marketing AI Agent")
st.caption("Ask me anything. My creator Anastasiia equipped me with data analysis, image generation, and document wizardry. I'm not saying I can do everything... but I've never seen proof that I can't.")

st.markdown("""
<style>
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stFileUploaderDropzone"] { min-height: 50px; padding: 0; border: none; background: transparent; }
[data-testid="stFileUploader"] button { border-radius: 12px; }
[data-testid="stFileUploaderFile"] { display: none; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# CONSTANTS
# -------------------------
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file"
]
NOTION_SCOPES = ""

# Works locally (reads env var) AND on Streamlit Cloud (reads st.secrets)
REDIRECT_URI = st.secrets.get("REDIRECT_URI", os.getenv("REDIRECT_URI", "http://localhost:8501"))
NOTION_CLIENT_ID = st.secrets.get("NOTION_CLIENT_ID", os.getenv("NOTION_CLIENT_ID"))
NOTION_CLIENT_SECRET = st.secrets.get("NOTION_CLIENT_SECRET", os.getenv("NOTION_CLIENT_SECRET"))

# Load Google credentials: from st.secrets on Cloud, from file locally
if "google" in st.secrets:
    GOOGLE_CLIENT_ID = st.secrets["google"]["client_id"]
    GOOGLE_CLIENT_SECRET = st.secrets["google"]["client_secret"]
else:
    with open("client_secret.json") as f:
        _secret = json.load(f)["web"]
    GOOGLE_CLIENT_ID = _secret["client_id"]
    GOOGLE_CLIENT_SECRET = _secret["client_secret"]

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# -------------------------
# SESSION STATE
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = secrets.token_hex(8)

if "google_connected" not in st.session_state:
    st.session_state.google_connected = False
if "creds" not in st.session_state:
    st.session_state.creds = None
if "notion_connected" not in st.session_state:
    st.session_state.notion_connected = False
if "notion_token" not in st.session_state:
    st.session_state.notion_token = None
if "notion_page_id" not in st.session_state:
    st.session_state.notion_page_id = None
if "user_openai_key" not in st.session_state:
    st.session_state.user_openai_key = ""
if "generated_images" not in st.session_state:
    st.session_state.generated_images = {}
if "show_google_success" not in st.session_state:
    st.session_state.show_google_success = False
if "show_notion_success" not in st.session_state:
    st.session_state.show_notion_success = False
if "pending_google_action" not in st.session_state:
    st.session_state.pending_google_action = None
if "pending_notion_action" not in st.session_state:
    st.session_state.pending_notion_action = None
if "auto_running" not in st.session_state:
    st.session_state.auto_running = False
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hi! I'm your Marketing AI Agent. "
            "I can analyse your campaigns, generate logos, stalk (legally) marketing trends,"
            "and build you content plans that actually make sense\n\n"
            "To get started, just drop your marketing goals, an image, or even a vague idea - I'll work with it.\n\n"
            "P.S. You're on a token budget, so the more context you give me now, the less we both suffer later"
        )
    }]

# -------------------------
# HELPERS
# -------------------------
def seed_agent_memory(agent, messages):
    """Push restored UI messages into the agent's LangChain memory."""
    try:
        history = agent.get_session_history(st.session_state.session_id)
        history.clear()
        for msg in messages:
            if msg["role"] == "user":
                history.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                history.add_ai_message(msg["content"])
    except Exception as e:
        print(f"Failed to seed memory: {e}")


def is_google_doc_request(text: str) -> bool:
    keywords = [
        "google doc", "create doc", "make a doc", "write a doc",
        "google document", "save to doc", "create a document"
    ]
    return any(k in text.lower() for k in keywords)


def is_notion_request(text: str) -> bool:
    keywords = [
        "notion", "content calendar", "calendar in notion",
        "save to notion", "create a calendar"
    ]
    return any(k in text.lower() for k in keywords)


def build_current_agent():
    return build_agent(
        creds=st.session_state.creds,
        openai_api_key=st.session_state.user_openai_key,
        notion_token=st.session_state.notion_token,
        notion_page_id=st.session_state.notion_page_id,
    )


def _current_creds_fingerprint():
    google_token = getattr(st.session_state.creds, "token", None) if st.session_state.creds else None
    return (
        st.session_state.user_openai_key,
        google_token,
        st.session_state.notion_token,
        st.session_state.notion_page_id,
    )


def ensure_agent_is_current():
    fingerprint = _current_creds_fingerprint()
    if (
        "agent" not in st.session_state
        or st.session_state.get("_agent_fingerprint") != fingerprint
    ):
        st.session_state.agent = build_current_agent()
        st.session_state._agent_fingerprint = fingerprint
        seed_agent_memory(st.session_state.agent, st.session_state.messages)


def run_agent(prompt: str) -> dict:
    """Run agent and return a dict with answer + image data."""
    image_data = st.session_state.get("uploaded_image")

    if image_data:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ]
    else:
        content = prompt

    input_data = {"input": [HumanMessage(content=content)]}

    response = st.session_state.agent.invoke(
        input_data,
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
    answer = response["output"]

    img_data = get_last_generated_image()
    generated_image_b64 = img_data.get("image_b64")
    drive_url = img_data.get("drive_url")
    campaign_visuals = img_data.get("campaign_visuals", [])

    needs_google_connect = False
    needs_notion_connect = False

    for step in response.get("intermediate_steps", []):
        tool_action = step[0] if isinstance(step, tuple) else None
        tool_output = step[1] if isinstance(step, tuple) else None
        tool_name = getattr(tool_action, "tool", "unknown")

        parsed_output = tool_output
        if isinstance(tool_output, str):
            try:
                parsed_output = json.loads(tool_output)
            except (json.JSONDecodeError, TypeError):
                parsed_output = None

        if isinstance(parsed_output, dict):
            if parsed_output.get("status") == "error":
                st.error(f"`{tool_name}` error: {parsed_output.get('message', 'unknown')}")
            elif parsed_output.get("status") == "auth_required":
                if parsed_output.get("message") == "GOOGLE_NOT_CONNECTED":
                    needs_google_connect = True
                elif parsed_output.get("message") == "NOTION_NOT_CONNECTED":
                    needs_notion_connect = True

    return {
        "answer": answer,
        "generated_image_b64": generated_image_b64,
        "drive_url": drive_url,
        "campaign_visuals": campaign_visuals,
        "needs_google_connect": needs_google_connect,
        "needs_notion_connect": needs_notion_connect,
    }

def display_agent_result(result: dict):
    """Render agent output (text + images) and save to session state."""
    answer = result["answer"]
    generated_image_b64 = result["generated_image_b64"]
    drive_url = result["drive_url"]
    campaign_visuals = result["campaign_visuals"]

    st.markdown(answer, unsafe_allow_html=True)
    if generated_image_b64:
        st.image(base64.b64decode(generated_image_b64), caption="Generated image", use_container_width=True)
    if drive_url:
        st.markdown(f"📁 [View in Google Drive]({drive_url})")
    for v in campaign_visuals:
        if v.get("image_b64"):
            st.image(base64.b64decode(v["image_b64"]), caption=v.get("visual_type", ""), use_container_width=True)
        if v.get("drive_url"):
            st.markdown(f"📁 [View in Google Drive]({v['drive_url']})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    if generated_image_b64 or campaign_visuals:
        msg_index = len(st.session_state.messages) - 1
        st.session_state.generated_images[msg_index] = {
            "image_b64": generated_image_b64,
            "drive_url": drive_url,
            "campaign_visuals": campaign_visuals,
        }

# -------------------------
# OAUTH CALLBACK — must be before st.stop()
# -------------------------

query_params = st.query_params

if "code" in query_params and not (st.session_state.google_connected and st.session_state.notion_connected):
    raw_state = query_params.get("state", "")
    state_data = {}
    if raw_state:
        try:
            state_data = json.loads(raw_state)
        except Exception:
            if raw_state.startswith("sk-"):
                st.session_state.user_openai_key = raw_state

    provider = state_data.get("provider", "google")  # default keeps old links working

    if state_data.get("key"):
        st.session_state.user_openai_key = state_data["key"]
    if state_data.get("messages") and len(st.session_state.messages) <= 1:
        st.session_state.messages = state_data["messages"]
    if state_data.get("pending_action"):
        if provider == "google":
            st.session_state.pending_google_action = state_data["pending_action"]
        elif provider == "notion":
            st.session_state.pending_notion_action = state_data["pending_action"]

    if provider == "google" and not st.session_state.google_connected:
        try:
            token_response = http_requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": query_params["code"],
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": REDIRECT_URI,
                    "grant_type": "authorization_code",
                }
            )
            token_data = token_response.json()

            if "error" in token_data:
                st.error(f"Google OAuth failed: {token_data}")
            else:
                creds = Credentials(
                    token=token_data["access_token"],
                    refresh_token=token_data.get("refresh_token"),
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=GOOGLE_CLIENT_ID,
                    client_secret=GOOGLE_CLIENT_SECRET,
                    scopes=GOOGLE_SCOPES
                )
                st.session_state.creds = creds
                st.session_state.google_connected = True
                st.session_state.show_google_success = True
        except Exception as e:
            st.error(f"Google OAuth failed: {e}")

    elif provider == "notion" and not st.session_state.notion_connected:
        try:
            token_data = exchange_notion_code_for_token(
                code=query_params["code"],
                client_id=NOTION_CLIENT_ID,
                client_secret=NOTION_CLIENT_SECRET,
                redirect_uri=REDIRECT_URI,
            )

            if "error" in token_data:
                st.error(f"Notion OAuth failed: {token_data}")
            else:
                notion_token = token_data["access_token"]
                st.session_state.notion_token = notion_token
                st.session_state.notion_page_id = find_accessible_page_id(notion_token)
                st.session_state.notion_connected = True
                st.session_state.show_notion_success = True
        except Exception as e:
            st.error(f"Notion OAuth failed: {e}")

    # Always rerun after OAuth — on second pass query_params still has the
    # code so key/messages/pending_action are restored again, but
    # google_connected/notion_connected are already True so token exchange
    # is skipped. Clear params only once connected AND key is present.
    if (st.session_state.google_connected or st.session_state.notion_connected) and st.session_state.user_openai_key:
        st.query_params.clear()
    st.rerun()

# -------------------------
# OPENAI API KEY CHECK + VALIDATION
# -------------------------
def validate_openai_key(key: str) -> bool:
    try:
        resp = http_requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5
        )
        return resp.status_code == 200
    except Exception:
        return False

if not st.session_state.user_openai_key:
    st.warning("⚠️ Please enter your OpenAI API key to use this app.")
    key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored — it lives only in your browser session."
    )
    if key_input:
        if not key_input.startswith("sk-"):
            st.error("❌ Invalid key format — must start with 'sk-'")
            st.stop()
        with st.spinner("Validating key..."):
            if validate_openai_key(key_input):
                st.session_state.user_openai_key = key_input
                st.rerun()
            else:
                st.error("❌ Key is invalid or has no credits — please check and try again.")
                st.stop()
    else:
        st.stop()

# -------------------------
# AGENT INIT / REBUILD — single source of truth
# -------------------------

with st.spinner("Starting up agent..."):
    ensure_agent_is_current()

# -------------------------
# AUTO-EXECUTE PENDING GOOGLE ACTION (after OAuth return)
# -------------------------
if (
    st.session_state.google_connected
    and st.session_state.pending_google_action
    and st.session_state.get("agent")
    and not st.session_state.auto_running
):
    st.session_state.auto_running = True
    pending = st.session_state.pending_google_action
    st.session_state.pending_google_action = None

    with st.chat_message("assistant"):
        with st.spinner("Google connected! Creating your doc now..."):
            try:
                auto_prompt = (
                    f"The user just connected Google. Please now create the Google Doc as previously discussed. "
                    f"Context from the conversation: {pending}"
                )
                result = run_agent(auto_prompt)
                display_agent_result(result)
            except Exception as e:
                import traceback
                st.error(f"Agent crashed: {e}")
                st.code(traceback.format_exc(), language="python")

    st.session_state.auto_running = False
    st.rerun()

# -------------------------
# AUTO-EXECUTE PENDING NOTION ACTION (after OAuth return)
# -------------------------
if (
    st.session_state.notion_connected
    and st.session_state.pending_notion_action
    and st.session_state.get("agent")
    and not st.session_state.auto_running
):
    st.session_state.auto_running = True
    pending = st.session_state.pending_notion_action
    st.session_state.pending_notion_action = None

    with st.chat_message("assistant"):
        with st.spinner("Notion connected! Building your content calendar now..."):
            try:
                auto_prompt = (
                    f"The user just connected Notion. Please now create the content calendar as "
                    f"previously discussed. Context from the conversation: {pending}"
                )
                result = run_agent(auto_prompt)
                display_agent_result(result)
            except Exception as e:
                import traceback
                st.error(f"Agent crashed: {e}")
                st.code(traceback.format_exc(), language="python")

    st.session_state.auto_running = False
    st.rerun()

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and i in st.session_state.generated_images:
            img_data = st.session_state.generated_images[i]
            if img_data.get("image_b64"):
                st.image(
                    base64.b64decode(img_data["image_b64"]),
                    caption="Generated image",
                    use_container_width=True
                )
            if img_data.get("drive_url"):
                st.markdown(f"📁 [View in Google Drive]({img_data['drive_url']})")
            for v in img_data.get("campaign_visuals", []):
                if v.get("image_b64"):
                    st.image(
                        base64.b64decode(v["image_b64"]),
                        caption=v.get("visual_type", ""),
                        use_container_width=True
                    )
                if v.get("drive_url"):
                    st.markdown(f"📁 [View in Google Drive]({v['drive_url']})")


# -------------------------
# CONNECT BUTTONS (Google + Notion)
# -------------------------
def trim_messages_for_url(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages[-6:]]

google_needed = (
    st.session_state.pending_google_action and
    not st.session_state.google_connected
)

notion_needed = (
    st.session_state.pending_notion_action and
    not st.session_state.notion_connected
)

active_provider = None

if google_needed and notion_needed:
    active_provider = st.session_state.get("last_pending_provider", "google")
elif google_needed:
    active_provider = "google"
elif notion_needed:
    active_provider = "notion"

if active_provider == "google":

    state_data = json.dumps({
        "provider": "google",
        "key": st.session_state.user_openai_key,
        "messages": trim_messages_for_url(st.session_state.messages),
        "pending_action": st.session_state.pending_google_action,
    })

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state_data,
    }

    auth_url = "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)

    if st.session_state.pending_google_action:
        components.html(
            f"""
            <div style="background:#1e3a5f;border:1px solid #4285F4;border-radius:10px;padding:1rem;font-family:sans-serif;">
                <p style="margin:0 0 0.5rem 0;color:#cfe8ff;font-weight:600;">
                📄 Ready to create your Google Doc — just connect first:
                </p>
                <a href="{auth_url}" target="_top"
                   style="display:block;text-align:center;background:#4285F4;color:white;
                   padding:0.6rem 1rem;border-radius:8px;font-weight:600;text-decoration:none;">
                   👉 Connect Google & Create Doc
                </a>
            </div>
            """,
            height=110,
        )
    else:
        with st.popover("🔗 Connect Google Drive"):
            st.write("Authorize Google Drive access")
            components.html(
                f'<a href="{auth_url}" target="_top" '
                f'style="display:block;text-align:center;background:#4285F4;color:white;'
                f'padding:0.5rem 1rem;border-radius:8px;font-weight:600;text-decoration:none;font-family:sans-serif;">'
                f'👉 Continue with Google</a>',
                height=50,
            )


elif active_provider == "notion":

    state_data = json.dumps({
        "provider": "notion",
        "key": st.session_state.user_openai_key,
        "messages": trim_messages_for_url(st.session_state.messages),
        "pending_action": st.session_state.pending_notion_action,
    })

    notion_params = {
        "client_id": NOTION_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "owner": "user",
        "state": state_data,
    }

    notion_auth_url = "https://api.notion.com/v1/oauth/authorize?" + urllib.parse.urlencode(notion_params)

    if st.session_state.pending_notion_action:
        components.html(
            f"""
            <div style="background:#2d2440;border:1px solid #9b87f5;border-radius:10px;padding:1rem;font-family:sans-serif;">
                <p style="margin:0 0 0.5rem 0;color:#e3d9ff;font-weight:600;">
                🗒️ Ready to build your content calendar — just connect first:
                </p>
                <a href="{notion_auth_url}" target="_top"
                   style="display:block;text-align:center;background:#000;color:white;
                   padding:0.6rem 1rem;border-radius:8px;font-weight:600;text-decoration:none;">
                   👉 Connect Notion & Build Calendar
                </a>
            </div>
            """,
            height=110,
        )
    else:
        with st.popover("🔗 Connect Notion"):
            st.write("Authorize Notion access — you'll pick which page to share")
            components.html(
                f'<a href="{notion_auth_url}" target="_top" '
                f'style="display:block;text-align:center;background:#000;color:white;'
                f'padding:0.5rem 1rem;border-radius:8px;font-weight:600;text-decoration:none;font-family:sans-serif;">'
                f'👉 Continue with Notion</a>',
                height=50,
            )
# -------------------------
# IMAGE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    if uploaded_file.size > 2 * 1024 * 1024:
        st.error("Please upload an image smaller than 2 MB.")
        st.stop()

    img = Image.open(uploaded_file)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.thumbnail((1024, 1024))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    st.session_state.uploaded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    st.markdown(f"📎 {uploaded_file.name}")

# -------------------------
# CHAT INPUT
# -------------------------
user_input = st.chat_input("Describe your goals or ask for a marketing plan...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    st.session_state.pending_image_for_prompt = st.session_state.get("uploaded_image")

    with st.chat_message("user"):
        st.markdown(user_input)

    user_said_yes = user_input.strip().lower() in [
        "yes", "yeah", "yep", "sure", "ok", "okay", "do it", "yes please", "please do", "go ahead"
    ]
    last_assistant = next(
        (m["content"] for m in reversed(st.session_state.messages[:-1]) if m["role"] == "assistant"),
        ""
    )

    # ── Google Doc flow: user said yes + doc was suggested ────────────
    doc_was_suggested = is_google_doc_request(last_assistant)
    if st.session_state.pending_google_action and not st.session_state.google_connected:
        still_wants_doc = user_said_yes or is_google_doc_request(user_input)
        if not still_wants_doc:
            st.session_state.pending_google_action = None
    if user_said_yes and doc_was_suggested and not st.session_state.google_connected:
        st.session_state.pending_google_action = last_assistant
        msg = (
            "To create Google Docs I need access to your Google account. "
            "Please click the **🔗 Connect Google Drive** button below — "
            "you'll come right back here and I'll create the doc automatically! 🚀"
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.rerun()

    # ── Notion flow: user said yes + calendar was suggested ───────────
    notion_was_suggested = is_notion_request(last_assistant)
    if st.session_state.pending_notion_action and not st.session_state.notion_connected:
        still_wants_notion = user_said_yes or is_notion_request(user_input)
        if not still_wants_notion:
            st.session_state.pending_notion_action = None
    if user_said_yes and notion_was_suggested and not st.session_state.notion_connected:
        st.session_state.pending_notion_action = last_assistant
        msg = (
            "To build your content calendar I need access to your Notion workspace. "
            "Please click the **🔗 Connect Notion** button below — "
            "you'll come right back here and I'll build the calendar automatically! 🚀"
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.rerun()

    # ── Normal agent call ──────────────────────────────────────────────
    if not (
        (user_said_yes and doc_was_suggested and not st.session_state.google_connected)
        or (user_said_yes and notion_was_suggested and not st.session_state.notion_connected)
    ):
        with st.chat_message("assistant"):
            with st.spinner("Thinking and pulling data..."):
                try:
                    result = run_agent(user_input)
                    display_agent_result(result)

                    rerun_needed = False
                    if result.get("needs_google_connect") and not st.session_state.google_connected:
                        st.session_state.pending_google_action = user_input
                        rerun_needed = True
                    if result.get("needs_notion_connect") and not st.session_state.notion_connected:
                        st.session_state.pending_notion_action = user_input
                        rerun_needed = True
                    if rerun_needed:
                        st.rerun()
                except Exception as e:
                    import traceback
                    st.error(f"Agent crashed: {e}")
                    st.code(traceback.format_exc(), language="python")
                    st.session_state.messages.append({"role": "assistant", "content": "Something went wrong."})