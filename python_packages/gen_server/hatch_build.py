import os
import subprocess
import sys
import stat
import shutil
import platform
import threading
import requests
import logging
import zipfile
import tarfile
import tempfile
from typing import Tuple, Union, Any
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

COZY_VERSION = "v0.0.0"
GITHUB_DOWNLOAD_BASE_URL = (
    "https://github.com/cozy-creator/gen-server/releases/download"
)

logger = logging.getLogger(__name__)


def extract_file(file_path, extract_to="."):
    system = platform.system().lower()

    try:
        if system == "windows":
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            with tarfile.open(file_path, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
    except Exception as e:
        logger.error(f"Error extracting file: {e}")
        raise
    finally:
        logger.info(f"File extracted successfully: {file_path}")


def get_platform() -> Tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return "windows", "amd64"
    elif system == "linux":
        if machine == "x86_64":
            return "linux", "amd64"
        elif machine == "aarch64":
            return "linux", "arm64"
        else:
            raise ValueError(f"Unsupported machine: {machine}")
    elif system == "darwin":
        if machine == "x86_64":
            return "darwin", "amd64"
        elif machine == "arm64":
            return "darwin", "arm64"
        else:
            raise ValueError(f"Unsupported machine: {machine}")
    else:
        raise ValueError(f"Unsupported system: {system}")


def get_compressed_binary_name(version: Union[int, str] = COZY_VERSION) -> str:
    version = str(version).lower()
    if not version.startswith("v"):
        version = f"v{version}"

    system, arch = get_platform()
    ext = "zip" if system == "windows" else "tar.gz"
    return f"cozy-{version}-{system}-{arch}.{ext}"


def get_binary_name() -> str:
    system, _ = get_platform()
    ext = ".exe" if system == "windows" else ""
    return f"cozy{ext}"


def get_binary_download_url(version: Union[int, str] = COZY_VERSION) -> str:
    version = str(version).lower()
    if not version.startswith("v"):
        version = f"v{version}"

    binary_name = get_compressed_binary_name(version)
    return f"{GITHUB_DOWNLOAD_BASE_URL}/{version}/{binary_name}"


def install_binary(install_dir: str):
    if is_go_installed():
        try:
            logger.info("Building cozy binary")
            output_path = os.path.join(install_dir, get_binary_name())
            result = subprocess.run(
                ["go", "build", "-o", output_path, "../.."],
                check=True,
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    result.stdout,
                    result.stderr,
                )
            logger.info("Cozy binary built successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building cozy binary: {e}")
            raise
    else:
        logger.info("Go not installed, downloading cozy binary")
        binary_name = get_binary_name()
        install_path = os.path.join(install_dir, binary_name)
        if os.path.exists(install_path):
            print(f"Binary already exists at {install_path}")
            return

        download_url = get_binary_download_url()

        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

        with tempfile.TemporaryDirectory() as temp_dir:
            extract_file(temp_file.name, temp_dir)

            binary_path = os.path.join(temp_dir, binary_name)

            if not os.path.exists(binary_path):
                raise ValueError(f"Binary not found at {binary_path}")

            install_path = os.path.join(install_dir, binary_name)
            shutil.copyfile(binary_path, install_path)

            mode = (
                stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
            )
            os.chmod(install_path, mode)
            print(f"Binary installed to {install_path}")

        os.remove(temp_file.name)


def run_install_binary(install_dir: str):
    threading.Thread(target=install_binary, args=(install_dir,)).start()


def is_node_installed() -> bool:
    try:
        _output = subprocess.check_output(
            ["node", "--version"],
            stderr=subprocess.PIPE,
        )
        return True
    except FileNotFoundError:
        return False


def is_go_installed() -> bool:
    try:
        _output = subprocess.run(
            ["go", "version"],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


def install_web(web_dir: str):
    if not is_node_installed():
        print("Node.js is not installed. Please install Node.js before proceeding.")
        sys.exit(1)

    try:
        os.chdir(web_dir)

        # Install web dependencies
        print("Installing web dependencies... \n")
        run = subprocess.run(["npm", "install"], check=True)
        if run.returncode != 0:
            print("'\nFailed to install web dependencies... \n")
            sys.exit(1)
        print("'\nSuccessfully installed web dependencies... \n")

        # Build the web directory
        print("Building web assets... \n")
        build = subprocess.run(["npm", "run", "build"], check=True)
        if build.returncode != 0:
            print("'\nFailed to build web assets... \n")
            sys.exit(1)
        print("'\nSuccessfully built web assets... \n")

    except FileNotFoundError:
        print("npm is not installed")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' failed with code {e.returncode}.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while running npm commands: {e}")
        sys.exit(1)


def run_install_web(web_dir: str):
    threading.Thread(target=install_web, args=(web_dir,)).start()


class CustomBuildHook(BuildHookInterface):
    def finalize(
        self,
        version: str,
        build_data: dict[str, Any],
        artifact_path: str,
    ) -> None:
        try:
            web_directory = os.path.join(os.path.dirname(__file__), "..", "..", "web")
            bin_directory = os.path.dirname(sys.executable)
            if not os.path.exists(bin_directory):
                logger.error(
                    f"No bin directory found at {bin_directory}... skipping hook"
                )
                return

            run_install_binary(bin_directory)
            run_install_web(web_directory)
        except Exception as e:
            logger.error(f"Error installing binary: {e}")
            raise
