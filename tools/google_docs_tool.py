from typing import Dict, Any, Optional
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from datetime import datetime
import traceback
import os


def make_google_doc_tool(creds=None):
    """Factory: returns a create_google_doc tool with credentials baked in."""

    def create_google_doc(title: str, content: str, doc_type: str = "strategy") -> Dict[str, str]:
        """Create a Google Doc using the authenticated user's credentials."""
        try:
            if creds is None:
                return {
                    "status": "auth_required", 
                    "message": "GOOGLE_NOT_CONNECTED: The user needs to connect Google Drive before I can create docs. They should click the 'Connect Google Drive' button at the top of the chat."
                }

            from googleapiclient.discovery import build

            docs_service = build("docs", "v1", credentials=creds)

            doc = docs_service.documents().create(
                body={"title": title}
            ).execute()

            doc_id = doc["documentId"]

            requests_body = [
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": (
                            f"{title}\n\n"
                            f"Generated: {datetime.now().strftime('%B %d, %Y')}\n"
                            f"Type: {doc_type}\n\n"
                            f"{content}\n\n"
                            "---\nCreated by Marketing AI Agent"
                        )
                    }
                }
            ]

            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": requests_body}
            ).execute()

            url = f"https://docs.google.com/document/d/{doc_id}/edit"
            return {
                "status": "success",
                "url": url,
                "message": f"Google Doc created successfully: {url}"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }

    return StructuredTool.from_function(
        func=create_google_doc,
        name="create_google_doc",
        description="Create a Google Doc with the given title and content. Use when the user asks to export or save to Google Docs.",
    )

@tool
def export_as_markdown(content: str) -> str:
    """Format strategy as markdown for download."""
    return f"""# Marketing Strategy\n\nGenerated: {datetime.now().strftime('%B %d, %Y')}\n\n---\n\n{content}\n\n---\n\n*Created by Marketing AI Agent*\n"""


@tool
def export_as_html(content: str) -> str:
    """Format strategy as HTML."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Marketing Strategy</title>
    <style>body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; }}</style>
</head>
<body>
    <h1>Marketing Strategy</h1>
    <p>Generated: {datetime.now().strftime('%B %d, %Y')}</p>
    <div>{content}</div>
</body>
</html>"""