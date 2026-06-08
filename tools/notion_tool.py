import os
import requests
import json
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = os.getenv("NOTION_PAGE_ID")

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}


@tool
def create_content_calendar(plan_json: str) -> str:
    """
    Creates a content calendar database in Notion from a marketing plan.
    Input must be a JSON string with a list of posts.
    Each post must have: date (YYYY-MM-DD), platform, content, status.
    Call this when the user asks to save or create a content plan in Notion.

    Example input:
    [
        {"date": "2026-06-10", "platform": "Instagram", "content": "Post text here", "status": "Draft"},
        {"date": "2026-06-12", "platform": "LinkedIn", "content": "Post text here", "status": "Draft"}
    ]
    """
    try:
        # Parse the JSON input
        posts = json.loads(plan_json)

        # Step 1 — Create the database inside the Content Calendar page
        db_response = requests.post(
            "https://api.notion.com/v1/databases",
            headers=HEADERS,
            json={
                "parent": {"type": "page_id", "page_id": PAGE_ID},
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

        if db_response.status_code != 200:
            return f"Failed to create database: {db_response.text}"

        database_id = db_response.json()["id"]

        # Step 2 — Add each post as a row
        for post in posts:
            page_response = requests.post(
                "https://api.notion.com/v1/pages",
                headers=HEADERS,
                json={
                    "parent": {"database_id": database_id},
                    "properties": {
                        "Post Content": {
                            "title": [{"text": {"content": post["content"]}}]
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

        notion_url = f"https://www.notion.so/{database_id.replace('-', '')}"
        return json.dumps({
            "status": "success",
            "posts_created": len(posts),
            "notion_url": notion_url
        })

    except json.JSONDecodeError:
        return "Error: Invalid JSON format. Make sure the input is a valid JSON list of posts."
    except Exception as e:
        return f"Error: {e}"