import psycopg2
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

DB_DSN = os.getenv("DB_DSN")

prompts = [
    {"positive_prompt": "score_9, score_8_up, score_7_up, BREAK", "negative_prompt": "score_4, score_5, score_6, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs,(mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"},
    # {"positive_prompt": "Futuristic city", "negative_prompt": "Overexposed"},
]

def populate_prompts():
    try:
        conn = psycopg2.connect(DB_DSN)
        cursor = conn.cursor()

        for prompt in prompts:
            prompt_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO prompt_defs (id, positive_prompt, negative_prompt)
                VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;
                """,
                (prompt_id, prompt["positive_prompt"], prompt["negative_prompt"]),
            )

        conn.commit()
        print("Prompts populated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    populate_prompts()
