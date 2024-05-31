These are built-in core extensions.

In ComfyUI, you git-clone repos into an /extensions folder, and then ComfyUI has to search this folder to manually figure out what extensions are installed.

This is not a great approach. Instead we use the standard python-extension approach (like flash and jupyter-notebooks) and have users installed extensions as python packages. We then auto-detect these extension-packages installed in the currently running environment and activate them.

Note that `comfy-creator.extensions` is the package-group-name to use in your `setup.py` file's entry-points list.
