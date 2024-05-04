# Should not be limited to just CLIP, but any text encoder
# We should probably return a summary-embedding (pooled-output) in addition
# to the token-embeddings. ComfyUI 'conditioning' returns both the token
# embedding and pooled_output together as a single object.

class TextEncoder:
    def run(self, text: str):
        pass


