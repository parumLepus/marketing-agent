import requests
import json
from langchain.tools import tool

NOTION_VERSION = "2022-06-28"


def _add_comment(headers: dict, page_id: str, text: str):
    """Best-effort: attach the full post detail as a comment on its page. Swallows
    failures (e.g. the integration lacks comment-insert capability) since the row
    itself was already created successfully and shouldn't be reported as failed."""
    try:
        requests.post(
            "https://api.notion.com/v1/comments",
            headers=headers,
            json={
                "parent": {"page_id": page_id},
                "rich_text": [{"text": {"content": text[:2000]}}],
            },
        )
    except Exception:
        pass


def make_content_calendar_tool(notion_token: str, notion_page_id: str):
    """
    Factory that builds a create_content_calendar tool bound to ONE specific
    user's Notion access token and the page they chose during OAuth consent.

    Mirrors make_google_doc_tool's pattern: each user gets a tool instance
    wired to their own credentials, instead of every user sharing one
    module-level NOTION_TOKEN / PAGE_ID.

    If notion_token or notion_page_id is missing, the tool returns a
    structured "auth_required" status instead of failing — the calling app
    is expected to catch this and prompt the user to connect Notion, the
    same way it already does for GOOGLE_NOT_CONNECTED.
    """

    headers = {
        "Authorization": f"Bearer {notion_token}" if notion_token else "",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }

    @tool
    def create_content_calendar(plan_json: str) -> str:
        """
        Creates a content calendar database in Notion from a marketing plan.
        Input must be a JSON string with a list of posts.
        Each post must have: date (YYYY-MM-DD), platform, name, content, status.
        "name" is a short idea title (a few words) - this becomes the row's
        title in Notion. "content" is the full post detail (hook/problem/
        solution/CTA, caption, whatever the format is) - this gets attached
        as a comment on the row instead of the title, so the calendar stays
        scannable at a glance.
        Call this when the user asks to save or create a content plan in Notion.

        Example input:
        [
            {"date": "2026-06-10", "platform": "Instagram", "name": "Feature walkthrough hook", "content": "Hook: ...\\nProblem: ...\\nSolution: ...\\nCTA: ...", "status": "Draft"},
            {"date": "2026-06-12", "platform": "LinkedIn", "name": "Customer proof post", "content": "Full post text here", "status": "Draft"}
        ]
        """
        if not notion_token or not notion_page_id:
            return json.dumps({
                "status": "auth_required",
                "message": "NOTION_NOT_CONNECTED"
            })

        try:
            posts = json.loads(plan_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format. Make sure the input is a valid JSON list of posts."

        try:
            # Step 1 — Create the database inside the page the user chose at OAuth time
            db_response = requests.post(
                "https://api.notion.com/v1/databases",
                headers=headers,
                json={
                    "parent": {"type": "page_id", "page_id": notion_page_id},
                    "title": [{"type": "text", "text": {"content": "Content Calendar"}}],
                    "properties": {
                        "Post Content": {"title": {}},
                        "Platform": {"select": {}},
                        "Date": {"date": {}},
                        "Status": {
                            "select": {
                                "options": [
                                    {"name": "Draft", "color": "gray"},
                                    {"name": "Ready", "color": "green"},
                                    {"name": "Published", "color": "blue"}
                                ]
                            }
                        }
                    }
                }
            )

            if db_response.status_code == 401:
                return json.dumps({
                    "status": "auth_required",
                    "message": "NOTION_NOT_CONNECTED"
                })

            if db_response.status_code != 200:
                error_body = db_response.json() if db_response.headers.get("content-type", "").startswith(
                    "application/json") else {}
                if "cannot have content" in error_body.get("message", "") or error_body.get(
                        "code") == "validation_error":
                    return json.dumps({
                        "status": "invalid_target",
                        "message": "The shared Notion page can't hold new content. Ask the user to share a different, blank page when reconnecting Notion."
                    })
                return f"Failed to create database: {db_response.text}"

            database_id = db_response.json()["id"]

            # Step 2 — Add each post as a row, name as the title and the full
            # detail as a comment on that row
            for post in posts:
                page_response = requests.post(
                    "https://api.notion.com/v1/pages",
                    headers=headers,
                    json={
                        "parent": {"database_id": database_id},
                        "properties": {
                            "Post Content": {
                                "title": [{"text": {"content": post.get("name", post["content"])}}]
                            },
                            "Platform": {
                                "select": {"name": post["platform"]}
                            },
                            "Date": {
                                "date": {"start": post["date"]}
                            },
                            "Status": {
                                "select": {"name": post.get("status", "Draft")}
                            }
                        }
                    }
                )

                if page_response.status_code != 200:
                    return f"Failed to add post: {page_response.text}"

                _add_comment(headers, page_response.json()["id"], post["content"])

            notion_url = f"https://www.notion.so/{database_id.replace('-', '')}"
            return json.dumps({
                "status": "success",
                "posts_created": len(posts),
                "notion_url": notion_url
            })

        except Exception as e:
            return f"Error: {e}"

    return create_content_calendar


def make_update_content_calendar_tool(notion_token: str):
    """
    Factory for an UPDATE tool that adds rows to an EXISTING content
    calendar database, instead of creating a brand new one.

    This is what the system prompt's "assets are stateful objects" rule
    needs: once create_content_calendar has run once, expand/update/extend
    requests should call THIS tool against the existing database_id rather
    than create_content_calendar again.
    """

    headers = {
        "Authorization": f"Bearer {notion_token}" if notion_token else "",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }

    @tool
    def update_content_calendar(database_id: str, plan_json: str) -> str:
        """
        Adds more posts to an EXISTING Notion content calendar database.
        Call this instead of create_content_calendar when a calendar already
        exists and the user asks to expand, extend, add more days, or add
        more posts to it.

        database_id: the id of the existing Notion database (from the
        notion_url returned by create_content_calendar).
        plan_json: JSON string, same shape as create_content_calendar's input
        — a list of new posts to add, each with date, platform, name,
        content, status.
        """
        if not notion_token:
            return json.dumps({
                "status": "auth_required",
                "message": "NOTION_NOT_CONNECTED"
            })

        try:
            posts = json.loads(plan_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format. Make sure the input is a valid JSON list of posts."

        try:
            for post in posts:
                page_response = requests.post(
                    "https://api.notion.com/v1/pages",
                    headers=headers,
                    json={
                        "parent": {"database_id": database_id},
                        "properties": {
                            "Post Content": {
                                "title": [{"text": {"content": post.get("name", post["content"])}}]
                            },
                            "Platform": {
                                "select": {"name": post["platform"]}
                            },
                            "Date": {
                                "date": {"start": post["date"]}
                            },
                            "Status": {
                                "select": {"name": post.get("status", "Draft")}
                            }
                        }
                    }
                )
                if page_response.status_code == 401:
                    return json.dumps({
                        "status": "auth_required",
                        "message": "NOTION_NOT_CONNECTED"
                    })
                if page_response.status_code != 200:
                    return f"Failed to add post: {page_response.text}"

                _add_comment(headers, page_response.json()["id"], post["content"])

            notion_url = f"https://www.notion.so/{database_id.replace('-', '')}"
            return json.dumps({
                "status": "success",
                "posts_added": len(posts),
                "notion_url": notion_url
            })

        except Exception as e:
            return f"Error: {e}"

    return update_content_calendar


def exchange_notion_code_for_token(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    """
    Exchanges an OAuth authorization code for a Notion access token.
    Returns the parsed JSON response, which (on success) includes:
      - access_token
      - workspace_id, workspace_name
      - owner (contains the user info)
      - duplicated_template_id (if relevant)
    Notion's OAuth does NOT hand back a single "page_id" directly in the
    token response in all cases — access is granted at the page/workspace
    level the user picked during consent, and which pages/databases are
    visible to the integration depends on what they shared. See app.py's
    callback handling for how we resolve a usable page_id afterward.
    """
    import base64

    basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    response = requests.post(
        "https://api.notion.com/v1/oauth/token",
        headers={
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/json",
        },
        json={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
    )
    return response.json()


def find_accessible_page_id(notion_token: str) -> str | None:
    """
    After OAuth, asks Notion's Search API for pages the integration can now
    see, and returns the first top-level page id found. Notion's OAuth
    consent screen lets the user pick a page (or whole workspace) to share
    with the integration — this call discovers what was actually shared,
    since the token response itself doesn't include a ready-to-use page_id.
    """
    try:
        response = requests.post(
            "https://api.notion.com/v1/search",
            headers={
                "Authorization": f"Bearer {notion_token}",
                "Content-Type": "application/json",
                "Notion-Version": NOTION_VERSION,
            },
            json={
                "filter": {"property": "object", "value": "page"},
                "page_size": 1,
            },
        )
        results = response.json().get("results", [])
        if results:
            return results[0]["id"]
        return None
    except Exception:
        return None