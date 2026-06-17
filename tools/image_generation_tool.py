import os
import io
import base64
import requests
import tempfile
from datetime import datetime
from langchain_core.tools import StructuredTool

# Global temp store — app.py reads from here after tool runs
_last_generated_image = {}

def get_last_generated_image():
    return _last_generated_image.copy()

def make_image_generation_tool(creds=None, openai_api_key=None):
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    def generate_marketing_image(
        prompt: str,
        style: str = "professional marketing",
        size: str = "1024x1024"
    ) -> str:  # ✅ Returns SHORT string to LLM, not image data
        try:
            if not api_key:
                return "Error: Missing OpenAI API key"

            enhanced_prompt = (
                f"{prompt[:400]}. Style: {style}. "
                f"High quality, professional, clean composition, no text, suitable for marketing."
            )

            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-image-1",
                    "prompt": enhanced_prompt,
                    "n": 1,
                    "size": size,
                },
                timeout=120
            )

            if not response.ok:
                return f"Image API error {response.status_code}: {response.text[:200]}"

            data = response.json()
            image_b64 = data["data"][0]["b64_json"]
            image_bytes = base64.b64decode(image_b64)

            # ✅ Store image in module-level dict — NOT returned to LLM
            _last_generated_image["image_b64"] = image_b64
            _last_generated_image["drive_url"] = None

            # Upload to Drive if connected
            if creds:
                try:
                    from googleapiclient.discovery import build
                    from googleapiclient.http import MediaIoBaseUpload

                    drive_service = build("drive", "v3", credentials=creds)
                    filename = f"marketing_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    media = MediaIoBaseUpload(
                        io.BytesIO(image_bytes),
                        mimetype="image/png",
                        resumable=False
                    )
                    uploaded = drive_service.files().create(
                        body={"name": filename},
                        media_body=media,
                        fields="id, webViewLink"
                    ).execute()
                    _last_generated_image["drive_url"] = uploaded.get("webViewLink")
                except Exception as drive_err:
                    _last_generated_image["drive_warning"] = str(drive_err)

            # ✅ Only this short string goes back to the LLM
            drive_msg = f" Saved to Drive: {_last_generated_image['drive_url']}" if _last_generated_image.get("drive_url") else ""
            return f"Image generated successfully.{drive_msg} It will be displayed in the chat."

        except Exception as e:
            return f"Error generating image: {str(e)}"

    def generate_campaign_visuals(
        campaign_name: str,
        brand_description: str,
        visual_count: int = 2
    ) -> str:  # ✅ Also returns short string
        try:
            visual_count = min(visual_count, 3)
            prompts = [
                f"Hero banner for '{campaign_name}'. {brand_description}. Wide format, bold composition, no text.",
                f"Social media post for '{campaign_name}'. {brand_description}. Vibrant, modern square design, no text.",
                f"Email header for '{campaign_name}'. {brand_description}. Clean, minimal, wide format, no text.",
            ]
            labels = ["Hero Banner", "Social Post", "Email Header"]
            _last_generated_image["campaign_visuals"] = []

            for i in range(visual_count):
                result_msg = generate_marketing_image(
                    prompt=prompts[i],
                    style="professional marketing campaign visual",
                    size="1024x1024"
                )
                if "successfully" in result_msg and _last_generated_image.get("image_b64"):
                    _last_generated_image["campaign_visuals"].append({
                        "image_b64": _last_generated_image["image_b64"],
                        "visual_type": labels[i],
                        "drive_url": _last_generated_image.get("drive_url")
                    })

            count = len(_last_generated_image.get("campaign_visuals", []))
            return f"{count} campaign visuals generated successfully. They will be displayed in the chat."

        except Exception as e:
            return f"Error generating campaign visuals: {str(e)}"

    image_tool = StructuredTool.from_function(
        func=generate_marketing_image,
        name="generate_marketing_image",
        description=(
            "Generate a single marketing image. Use when user asks to create, generate, or design "
            "an image, visual, banner, logo, or graphic. Keep prompt under 200 words. "
            "Focus on composition, colours, mood. Never include people or faces in the prompt."
        ),
    )

    campaign_visuals_tool = StructuredTool.from_function(
        func=generate_campaign_visuals,
        name="generate_campaign_visuals",
        description="Generate multiple campaign visuals (hero banner, social post, email header) for a marketing campaign.",
    )

    return image_tool, campaign_visuals_tool