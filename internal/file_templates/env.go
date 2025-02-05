package templates

import "os"

const envTemplate = `
# This is an example .env file; it provides global configurations for the gen-server.
# Warning: DO NOT share this file or commit it to your git-history as it may contain
# sensitive fields.

# To configure where cozy-creator will locally store data.
# In particular, generated images, videos, audio, and downloaded models will be stored in this folder.
# Defaults to ~/.cozy-creator unless XDG_DATA_HOME is specified.
# COZY_HOME=~/.cozy-creator
# COZY_CONFIG_FILE=~/.cozy-creator/config.yaml

# Where the gen-server will search for extra model-files (state-dictionaries
# such as .safetensors), other than the MODELS_PATH.
# COZY_AUX_MODELS_PATHS='["C:/Users/User/git/ComfyUI/models"]'

# HTTP-location where the gen-server will be accessible from
# COZY_HOST=localhost
# COZY_PORT=8881

# enum for where files should be served from: 'LOCAL' or 'S3'
# LOCAL urls look like http://localhost:8881/media/<file-name>
# S3 urls look like https://storage.cozy.dev/<file-name>
# COZY_FILESYSTEM_TYPE=LOCAL


### Hugging Face Variables

# Used in place of any hugging-face token stored locally
# HF_TOKEN=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
# HF_HUB_CACHE=~/.cache/huggingface/hub


# You can also include custom environment variables not built into our gen-server
# MY_API_KEY=this-also-works!
`

func GetEnvTemplate() string {
	return envTemplate
}

func WriteEnv(path string) error {
	envTemplate := GetEnvTemplate()

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = file.WriteString(envTemplate)
	if err != nil {
		return err
	}

	return nil
}
