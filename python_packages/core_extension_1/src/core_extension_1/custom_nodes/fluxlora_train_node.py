import os
import sys
from collections import OrderedDict
from typing import List, Optional
from gen_server.base_types import CustomNode
from gen_server.utils.paths import get_assets_dir, get_home_dir
import asyncio
import traceback

class FluxTrainNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.ai_toolkit_path = self._get_ai_toolkit_path()
        self.home_dir = get_home_dir()

    def _get_ai_toolkit_path(self):
        # Look for ai-toolkit in the custom_nodes directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ai_toolkit_path = os.path.join(current_dir, 'ai-toolkit')
        
        if os.path.exists(ai_toolkit_path):
            return ai_toolkit_path
        else:
            raise ImportError("AI Toolkit not found. Please run the setup script to install it.")

    async def __call__(self,
                       processed_directory: str,
                       lora_name: str,
                       flux_version: str = "dev",
                       training_steps: int = 2500,
                       resolution: List[int] = [1024],
                       batch_size: int = 1,
                       learning_rate: float = 1e-4,
                       trigger_word: Optional[str] = None,
                       low_vram: bool = False,
                       seed: Optional[int] = None,
                       walk_seed: bool = True) -> dict[str, str]:

        # Ensure we're in the AI Toolkit directory
        # os.chdir(self.ai_toolkit_path)
        # sys.path.append(self.ai_toolkit_path)
        try:
            from ostris_ai_toolkit.toolkit.job import run_job 
        except ImportError:
            raise ImportError("AI Toolkit not found. Please run the setup script to install it.")

        config = self._prepare_config(processed_directory, lora_name, flux_version, training_steps, resolution, batch_size, learning_rate, trigger_word, low_vram, seed, walk_seed)

        progress_queue = asyncio.Queue()
        
        def callback(progress):
            progress_queue.put_nowait(progress)

        config['config']['process'][0]['callback'] = callback
        

        # Run the job
        async def run_job_with_progress():
            job_task = asyncio.create_task(asyncio.to_thread(run_job, config))
            
            try:
                while not job_task.done():
                    try:
                        progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        yield progress
                        if progress.get('type') == 'finished':
                            break
                    except asyncio.TimeoutError:
                        # This allows for checking cancellation regularly
                        continue
                    except asyncio.CancelledError:
                        job_task.cancel()
                        yield {"type": "cancelled", "message": "Training was cancelled"}
                        break
                
                # Check if the job_task raised an exception
                if job_task.done() and not job_task.cancelled():
                    job_task.result()  # This will raise any exception that occurred in run_job

            except Exception as e:
                error_message = f"An error occurred during training: {str(e)}\n{traceback.format_exc()}"
                print(error_message)  # Print to console for immediate visibility
                yield {"type": "error", "message": error_message}
            finally:
                if not job_task.done():
                    job_task.cancel()
                    try:
                        await job_task
                    except asyncio.CancelledError:
                        pass
                yield {"type": "finished"}

        return run_job_with_progress()
    

    def _prepare_config(self,
                        processed_directory: str,
                        lora_name: str,
                        flux_version: str,
                        training_steps: int,
                        resolution: List[int],
                        batch_size: int,
                        learning_rate: float,
                        trigger_word: Optional[str],
                        low_vram: bool,
                        seed: Optional[int],
                        walk_seed: bool) -> OrderedDict:
        
        config = OrderedDict({
            "job": "extension",
            "config": {
                "name": lora_name,
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": f"{self.home_dir}/lora_output",
                    "device": "cuda:0",
                    "network": {
                        "type": "lora",
                        "linear": 16,
                        "linear_alpha": 16
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 4
                    },
                    "datasets": [{
                        "folder_path": processed_directory,
                        "caption_ext": "txt",
                        "caption_dropout_rate": 0.05,
                        "shuffle_tokens": False,
                        "cache_latents_to_disk": True,
                        "resolution": resolution
                    }],
                    "train": {
                        "batch_size": batch_size,
                        "steps": training_steps,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": learning_rate,
                        "linear_timesteps": True,
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.99
                        },
                        "dtype": "bf16"
                    },
                    "model": {
                        "name_or_path": f"black-forest-labs/FLUX.1-{flux_version}",
                        "is_flux": True,
                        "quantize": True
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": 250,
                        "width": 1024,
                        "height": 1024,
                        "prompts": [
                            "A person in a futuristic cityscape",
                            "A portrait of someone in a natural setting",
                            "An abstract representation of emotion"
                        ],
                        "neg": "",
                        "guidance_scale": 4,
                        "sample_steps": 20,
                        "walk_seed": walk_seed
                    }
                }]
            }
        })

        # Add trigger word if provided
        if trigger_word:
            config["config"]["process"][0]["trigger_word"] = trigger_word
            # Update sample prompts to include trigger word
            for i, prompt in enumerate(config["config"]["process"][0]["sample"]["prompts"]):
                config["config"]["process"][0]["sample"]["prompts"][i] = f"{trigger_word} {prompt}"

        # Add assistant_lora_path for schnell version
        if flux_version.lower() == "schnell":
            config["config"]["process"][0]["model"]["assistant_lora_path"] = "ostris/FLUX.1-schnell-training-adapter"

        # Add low_vram option
        if low_vram:
            config["config"]["process"][0]["model"]["low_vram"] = True

        # Add seed if provided
        if seed is not None:
            config["config"]["process"][0]["sample"]["seed"] = seed

        return config
        