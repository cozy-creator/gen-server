package templates

import "os"

const configTemplate = `
home_dir: ~/.cozy-creator
filesystem_type: local
aux_models_paths: ['C:/Users/User/git/ComfyUI/models']

s3:
  endpoint_url: "https://nyc3.digitaloceanspaces.com"
  access_key: "DO00W9N964WMQC2MV6JK"
  region_name: "nyc3"
  bucket_name: "voidtech-storage-dev"
  folder: "public"
  vanity_url: "https://storage.cozy.dev"

enabled_models:
  playground2.5:
    source: hf:playgroundai/playground-v2.5-1024px-aesthetic
`

func GetConfigTemplate() string {
	return configTemplate
}

func WriteConfig(path string) error {
	configTemplate := GetConfigTemplate()

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = file.WriteString(configTemplate)
	if err != nil {
		return err
	}

	return nil
}
