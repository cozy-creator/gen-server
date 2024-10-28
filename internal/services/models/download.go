package models

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/hf-hub/hub"
)

const (
	ModelSourceTypeHuggingFace = "huggingface"
	ModelSourceTypeCivitai     = "civitai"
	ModelSourceTypeDirect      = "direct"
)

type ModelSource struct {
	Type     string `json:"type"`
	Location string `json:"location"`
}

var client = hub.DefaultClient()

func DownloadModels(config *config.Config) error {
	if config.EnabledModels == nil {
		return nil
	}

MainLoop:
	for _, model := range config.EnabledModels {
		if model.Source == "" {
			continue MainLoop
		}

		go func() {
			err := download(model.Source, "")
			if err != nil {
				fmt.Println("Error downloading model:", err)
			}
		}()

		if model.Components != nil {
		ComponentLoop:
			for name, component := range model.Components {
				if component.Source == "" {
					continue ComponentLoop
				}

				go func() {
					err := download(component.Source, name)
					if err != nil {
						fmt.Println("Error downloading component:", err)
					}
				}()
			}
		}
	}

	return nil
}

func download(model, name string) error {
	modelSource, err := toModelSource(model)
	if err != nil {
		return err
	}

	params := hub.DownloadParams{
		Repo:      hub.NewRepo(modelSource.Location),
		SubFolder: name,
	}
	if _, err := client.Download(&params); err != nil {
		return err
	}
	return nil
}

func toModelSource(source string) (*ModelSource, error) {
	var typeStr string
	var location string

	if strings.HasPrefix(source, "hf:") {
		typeStr = ModelSourceTypeHuggingFace
		location = strings.TrimPrefix(source, "hf:")
	} else if strings.Contains(source, "civitai.com") {
		typeStr = ModelSourceTypeCivitai
		location = source
	} else if strings.HasPrefix(source, "http") {
		typeStr = ModelSourceTypeDirect
		location = source
	} else {
		return nil, fmt.Errorf("unsupported model source: %s", source)
	}

	return &ModelSource{
		Type:     typeStr,
		Location: location,
	}, nil
}
