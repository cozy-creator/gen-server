import os
import subprocess
import sys
from gen_server.base_types import CustomNode


# Make this node a generator or use callback fucntion to return the output
class FLUXLoRATrainNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.ai_toolkit_path = self._setup_ai_toolkit()

    def _setup_ai_toolkit(self):
        # Check if AI Toolkit is installed
        try:
            import ai_toolkit
            return os.path.dirname(ai_toolkit.__file__)
        except ImportError:
            print("AI Toolkit not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/ostris/ai-toolkit.git"])
            import ai_toolkit
            return os.path.dirname(ai_toolkit.__file__)
        

    async def __call__(self, config_path: str) -> dict[str, str]:
        # Ensure we're in the AI Toolkit directory
        os.chdir(self.ai_toolkit_path)

        # Run the training script
        result = subprocess.run([sys.executable, "run.py", config_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")

        # Determine the output directory based on the config file name
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        output_dir = os.path.join("output", config_name)
        lora_path = os.path.join(output_dir, f"{config_name}.safetensors")

        return {"lora_path": lora_path, "training_output": result.stdout}
