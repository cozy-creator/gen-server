import os
import subprocess
import sys
import stat
import shutil
import platform
import requests
import logging
import zipfile
import tarfile
import tempfile
import re
from pathlib import Path
from typing import Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

COZY_VERSION = "v0.0.0"
GITHUB_DOWNLOAD_BASE_URL = (
    "https://github.com/cozy-creator/gen-server/releases/download"
)
DOWNLOAD_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_file(file_path: Path, extract_to: Path = Path(".")):
    system = platform.system().lower()

    try:
        if system == "windows":
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            with tarfile.open(file_path, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
    except (zipfile.BadZipFile, tarfile.ReadError) as e:
        logger.error(f"Error extracting file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {e}")
        raise
    else:
        logger.info(f"File extracted successfully: {file_path}")


def get_platform() -> Tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()

    machines_map = {"x86_64": "amd64", "aarch64": "arm64"}
    if system == "windows":
        return "windows", "amd64"

    if system not in ["linux", "darwin"]:
        raise ValueError(f"Unsupported {system.capitalize()} machine: {machine}")
    if machine not in ["amd64", "arm64"]:
        machine = machines_map.get(machine)

    if machine is None:
        raise ValueError(f"Unsupported {system.capitalize()} machine: {machine}")

    return system, machine


def validate_version(version: Union[int, str]) -> str:
    version_str = str(version).lower()
    if not re.match(r"^v?\d+\.\d+\.\d+$", version_str):
        raise ValueError(f"Invalid version format: {version}")
    return f"v{version_str}" if not version_str.startswith("v") else version_str


def get_compressed_binary_name(version: Union[int, str] = COZY_VERSION) -> str:
    version = validate_version(version)
    system, arch = get_platform()
    ext = "zip" if system == "windows" else "tar.gz"
    return f"cozy-{version}-{system}-{arch}.{ext}"


def get_binary_name() -> str:
    system, _ = get_platform()
    return "cozy.exe" if system == "windows" else "cozy"


def get_binary_download_url(version: Union[int, str] = COZY_VERSION) -> str:
    version = validate_version(version)
    binary_name = get_compressed_binary_name(version)
    return f"{GITHUB_DOWNLOAD_BASE_URL}/{version}/{binary_name}"


def download_with_retry(url: str, max_retries: int = MAX_RETRIES) -> requests.Response:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached. Download failed.")
                raise


def install_binary(install_dir: Path):
    try:
        if is_go_installed():
            logger.info("Building cozy binary")
            output_path = install_dir / get_binary_name()
            subprocess.run(
                ["go", "build", "-o", str(output_path), "../.."],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Cozy binary built successfully")
        else:
            logger.info("Go not installed, downloading cozy binary")
            binary_name = get_binary_name()
            install_path = install_dir / binary_name
            # if install_path.exists():
            #     logger.info(f"Binary already exists at {install_path}")
            #     return

            download_url = get_binary_download_url()
            response = download_with_retry(download_url)

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                extract_file(Path(temp_file.name), temp_dir_path)

                binary_path = temp_dir_path / binary_name
                if not binary_path.exists():
                    raise ValueError(f"Binary not found at {binary_path}")

                shutil.copy2(binary_path, install_path)
                install_path.chmod(install_path.stat().st_mode | stat.S_IEXEC)
                logger.info(f"Binary installed to {install_path}")

            os.unlink(temp_file.name)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error building/installing cozy binary: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during binary installation: {e}")
        raise


def is_node_installed() -> bool:
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True
        )
        version = result.stdout.strip()
        logger.info(f"Node.js version: {version}")
        return True
    except subprocess.CalledProcessError:
        logger.warning("Node.js is installed but returned a non-zero exit code")
        return False
    except FileNotFoundError:
        logger.warning("Node.js is not installed")
        return False


def is_go_installed() -> bool:
    try:
        result = subprocess.run(
            ["go", "version"], capture_output=True, text=True, check=True
        )
        version = result.stdout.strip()
        logger.info(f"Go version: {version}")
        return True
    except subprocess.CalledProcessError:
        logger.warning("Go is installed but returned a non-zero exit code")
        return False
    except FileNotFoundError:
        logger.warning("Go is not installed")
        return False


# npm, yarn, pnpm, jsr, bun
def install_web(web_dir: Path):
    if not is_node_installed():
        logger.error(
            "Node.js is not installed. Please install Node.js before proceeding."
        )
        raise RuntimeError("Node.js is not installed")

    try:
        os.chdir(web_dir)

        logger.info("Installing web dependencies...")
        subprocess.run(["npm", "install"], check=True, timeout=300)
        logger.info("Successfully installed web dependencies")

        logger.info("Building web assets...")
        subprocess.run(["npm", "run", "build"], check=True, timeout=300)
        logger.info("Successfully built web assets")

    except subprocess.CalledProcessError as e:
        logger.error(f"Command '{e.cmd}' failed with code {e.returncode}")
        raise
    except subprocess.TimeoutExpired:
        logger.error("npm command timed out")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while running npm commands: {e}")
        raise


class CustomBuildHook(BuildHookInterface):
    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        try:
            web_directory = Path(__file__).parent.parent.parent / "web"
            bin_directory = Path(sys.executable).parent
            if not bin_directory.exists():
                logger.error(
                    f"No bin directory found at {bin_directory}... skipping hook"
                )
                return

            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(install_binary, bin_directory)
                executor.submit(install_web, web_directory)

        except Exception as e:
            logger.error(f"Error in CustomBuildHook: {e}")
            raise
