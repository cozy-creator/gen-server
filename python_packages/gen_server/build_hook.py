from setuptools import setup
from setuptools.command.install import install
import os
import shutil


class CustomInstallCommand(install):
    def run(self):
        # Build or copy the Go binary
        go_binary_name = "cozy"
        go_binary_source = "./cozy"  # or build it dynamically
        go_binary_dest = os.path.join(self.install_scripts, go_binary_name)

        # Ensure the destination directory exists
        os.makedirs(self.install_scripts, exist_ok=True)

        # Copy the Go binary to the scripts directory
        shutil.copy(go_binary_source, go_binary_dest)

        # Make the binary executable
        os.chmod(go_binary_dest, 0o755)

        # Run the standard install process
        install.run(self)


setup(
    cmdclass={
        "install": CustomInstallCommand,
    },
)
