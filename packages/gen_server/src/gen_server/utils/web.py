import os
import subprocess
import sys


def is_node_installed() -> bool:
    try:
        _output = subprocess.check_output(
            ["node", "--version"],
            stderr=subprocess.STDOUT,
        )
        return True
    except FileNotFoundError:
        return False


def install_and_build_web_dir(directory):
    if not is_node_installed():
        print("Node.js is not installed. Please install Node.js before proceeding.")
        sys.exit(1)

    try:
        os.chdir(directory)

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
        logger.error("npm is not installed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command '{e.cmd}' failed with code {e.returncode}.")
    except Exception as e:
        logger.error(f"An error occurred while running npm commands: {e}")
