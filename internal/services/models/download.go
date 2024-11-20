package models

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

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

func DownloadEnabledModels(ctx context.Context, config *config.Config) error {
	modelJSON, _ := json.MarshalIndent(config.PipelineDefs, "", "  ")
    fmt.Printf("Enabled Model list:\n%s\n", string(modelJSON))

	if len(config.PipelineDefs) == 0 {
		fmt.Println("To download enabled models, please specify them in config.yaml")
		return nil
	}

	var wg sync.WaitGroup

MainLoop:
	for _, model := range config.PipelineDefs {
		select {
        case <-ctx.Done():
            return ctx.Err()
        default:
			modelJSON, _ = json.MarshalIndent(model, "", "  ")
    		fmt.Printf("Downloading model:\n%s\n", string(modelJSON))

			if model.Source == "" {
				continue MainLoop
			}

			// Create local copies for goroutine
			model := model
			wg.Add(1)
			go func() {
				defer wg.Done()

				downloadCtx, cancel := context.WithCancel(ctx)
                defer cancel()

				err := download(downloadCtx, model.Source, "")
                if err != nil {
                    if errors.Is(err, context.Canceled) {
                        fmt.Println("Download cancelled for model:", model.Source)
                    } else {
                        fmt.Println("Error downloading model:", err)
                    }
                }

				fmt.Println("Downloaded model:", model.Source)
			}()

			if model.Components != nil {
			ComponentLoop:
				for name, component := range model.Components {
					if component.Source == "" {
						continue ComponentLoop
					}

					// Create local copies for goroutine
					name, component := name, component
					wg.Add(1)
					go func() {
						defer wg.Done()

						downloadCtx, cancel := context.WithCancel(ctx)
                		defer cancel()

						err := download(downloadCtx, component.Source, name)
						if err != nil {
							if errors.Is(err, context.Canceled) {
								fmt.Println("Download cancelled for component:", component.Source)
							} else {
								fmt.Println("Error downloading component:", err)
							}
						}

						fmt.Println("Downloaded component:", component.Source)
					}()
				}
			}
		}
	}

	// Wait for all downloads to complete
	wg.Wait()

	return nil
}

// TO DO: actually use the context being passed in
func download(_ context.Context, model string, name string) error {
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
