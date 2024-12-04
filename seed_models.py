import sqlite3
import uuid
import os
import json

# Path to the SQLite database
DB_PATH = os.path.join("data", "main.db")

models_to_seed = [
    {
        "name": "playground2.5",
        "source": "hf:playgroundai/playground-v2.5-1024px-aesthetic",
        "class_name": "StableDiffusionXLPipeline",
        "custom_pipeline": None,
        "metadata": {
            "display_name": "Playground 2.5",
            "description": "A high-quality aesthetic model for diffusion.",
            "tags": ["aesthetic", "diffusion", "high-resolution"],
        },
        "default_args": {
            "guidance_scale": 3.5,
            "num_inference_steps": 25,
        },
        "components": {
            "scheduler": {
                "class_name": "EDMDPMSolverMultistepScheduler",
                "kwargs": {"use_kwargs_sigmas": True},
            },
        #     "text_encoder_2": {
        #         # "class_name": "transformers.CLIPTextModelWithProjection",
        #         "source": "hf:playgroundai/playground-v2.5-1024px-aesthetic/text_encoder_2",
        #     },
        },
    },
    {
        "name": "juggernaut-xl-v9",
        "display_name": "Juggernaut XL v9",
        "source": "hf:RunDiffusion/Juggernaut-XL-v9",
    },
    {
        "name": "playground",
        "metadata": {
            "display_name": "Playground",
        },
        "class_name": "StableDiffusionXLPipeline",
        "source": "file:D:/models/playground-v2.5-1024px-aesthetic.fp16.safetensors",
    },
    {
        "name": "sd1.5",
        "class_name": "StableDiffusionPipeline",
        "source": "https://civitai.com/api/download/models/256588?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    },
    {
        "name": "juggernaut-lpw",
        "metadata": {
            "display_name": "Juggernaut LPW",
        },
        "source": "hf:RunDiffusion/Juggernaut-XL-v9",
        "custom_pipeline": "lpw_stable_diffusion",
    }
]

# Update models in the database
def update_database():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for model in models_to_seed:
            # Convert fields to JSON for storage
            default_args_json = json.dumps(model.get("default_args", {}))
            components_json = json.dumps(model.get("components", {}))
            metadata_json = json.dumps(model.get("metadata", {}))

            # Update the model if it exists, or insert it if it doesn't
            cursor.execute(
                """
                UPDATE models
                SET source = ?,
                    class_name = ?,
                    custom_pipeline = ?,
                    metadata = ?,
                    default_args = ?,
                    components = ?,
                    updated_at = datetime('now')
                WHERE name = ?
                """,
                (
                    model["source"],
                    model.get("class_name"),
                    model.get("custom_pipeline"),
                    metadata_json,
                    default_args_json,
                    components_json,
                    model["name"],
                ),
            )

            # Check if the model was updated; if not, insert it
            if cursor.rowcount == 0:
                model_id = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT INTO models (
                        id, name, source, class_name, custom_pipeline,
                        metadata, default_args, components, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                    """,
                    (
                        model_id,
                        model["name"],
                        model["source"],
                        model.get("class_name"),
                        model.get("custom_pipeline"),
                        metadata_json,
                        default_args_json,
                        components_json,
                    ),
                )

        # Commit the transaction
        conn.commit()
        print("Database updated successfully.")

    except sqlite3.Error as e:
        print(f"Error updating database: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    update_database()
