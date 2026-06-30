from openai import OpenAI
import config


def get_client():
    return OpenAI(
        api_key=config.MIMO_API_KEY,
        base_url=config.MIMO_BASE_URL
    )


def analyze_face(base64_image):
    if not config.MIMO_API_KEY:
        return {"error": "MIMO_API_KEY not set in .env"}

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=config.MIMO_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this person's face concisely: "
                                "approximate age range, facial expression, "
                                "distinguishing features (glasses, facial hair, etc), "
                                "and overall appearance. Keep it under 50 words."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150
        )
        return {"description": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
