import sqlite3
import uuid
import os
import json

# Path to the SQLite database
DB_PATH = os.path.join("data", "main.db")

models_to_seed = [
    {
        "name": "flux.1-schnell-fp8",
        "source": "hf:black-forest-labs/FLUX.1-schnell",
        "components": {
            "transformer": {"source": "hf:cozy-creator/Flux.1-schnell-8bit/transformer"},
            "text_encoder_2": {"source": "hf:cozy-creator/Flux.1-schnell-8bit/text_encoder_2"},
        },
        "default_args": {"max_sequence_length": 256},
    },
    {
        "name": "flux.1-dev",
        "source": "hf:black-forest-labs/FLUX.1-dev",
        "default_args": {"max_sequence_length": 512},
    },
    {
        "name": "flux.1-dev-fp8",
        "source": "hf:black-forest-labs/FLUX.1-dev",
        "components": {
            "text_encoder_2": {"source": "hf:cozy-creator/FLUX.1-dev-8bit/text_encoder_2"},
            "transformer": {"source": "hf:cozy-creator/FLUX.1-dev-8bit/transformer"},
        },
        "default_args": {"max_sequence_length": 512},
    },
    {
        "name": "flux.1-dev-nf4",
        "source": "hf:black-forest-labs/FLUX.1-dev",
        "components": {
            "text_encoder_2": {"source": "hf:hf-internal-testing/flux.1-dev-nf4-pkg/text_encoder_2"},
            "transformer": {"source": "hf:hf-internal-testing/flux.1-dev-nf4-pkg/trasformer"},
        },
        "default_args": {"max_sequence_length": 512},
    },
    {
        "name": "openflux.1",
        "custom_pipeline": "pipeline_flux_with_cfg",
        "source": "hf:ostris/OpenFLUX.1",
        "default_args": {"max_sequence_length": 512},
    },
    {
        "name": "sd3.5-large-int8",
        "source": "hf:stabilityai/stable-diffusion-3.5-large",
        "components": {
            "transformer": {"source": "hf:cozy-creator/stable-diffusion-3.5-large-8bit/transformer"},
            "text_encoder_3": {"source": "hf:cozy-creator/stable-diffusion-3.5-large-8bit/text_encoder_3"},
        },
    },
    {
        "name": "sdxl.base",
        "source": "hf:stabilityai/stable-diffusion-xl-base-1.0",
    },
    {
        "name": "illustrious.xl",
        "class_name": "StableDiffusionXLPipeline",
        "source": "https://civitai.com/api/download/models/889818?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "metadata": {
            "display_name": "Illustrious XL",
            "lineage": "sdxl.base",
        },
        "components": {
            "scheduler": {"class_name": "EulerAncestralDiscreteScheduler"},
        },
    },
    {
        "name": "pony.v6",
        "class_name": "StableDiffusionXLPipeline",
        "source": "https://civitai.com/api/download/models/290640?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "components": {
            "scheduler": {"class_name": "EulerAncestralDiscreteScheduler"},
        },
    },
    {
        "name": "cyberrealistic.pony",
        "class_name": "StableDiffusionXLPipeline",
        "source": "https://civitai.com/api/download/models/953264?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "components": {
            "scheduler": {"class_name": "EulerAncestralDiscreteScheduler"},
        },
    },
    {
        "name": "wai.ani.ponyxl",
        "class_name": "StableDiffusionXLPipeline",
        "source": "https://civitai.com/api/download/models/1065370?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    },
    {
        "name": "real.dream.pony",
        "class_name": "StableDiffusionXLPipeline",
        "source": "https://civitai.com/api/download/models/832353?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "components": {
            "scheduler": {"class_name": "EulerAncestralDiscreteScheduler"},
        },
    },
    {
        "name": "pony.realism",
        "class_name": "StableDiffusionXLPipeline",
        "source": "https://civitai.com/api/download/models/914390?type=Model&format=SafeTensor&size=full&fp=fp16",
        "components": {
            "scheduler": {"class_name": "EulerAncestralDiscreteScheduler"},
        },
    },
    {
        "name": "babes_by_stable_yogi.v4.xl.fp16",
        "class_name": "StableDiffusionXLPipeline",
        "source": "https://civitai.com/api/download/models/984905?type=Model&format=SafeTensor&size=full&fp=fp16",
        "metadata": {
            "display_name": "Babes by Stable Yogi V4 XL",
            "lineage": "pony.v6",
        },
        "components": {
            "scheduler": {"class_name": "EulerAncestralDiscreteScheduler"},
        },
    },
    {
        "name": "playground2.5",
        "source": "hf:playgroundai/playground-v2.5-1024px-aesthetic",
    },
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
