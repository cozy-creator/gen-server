package generationnode

import (
	"fmt"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/scripts"
)

func GenerateReplicateImage(app *app.App, inputs map[string]interface{}) (map[string]interface{}, error) {
	prompt := inputs["positive_prompt"].(string)
	style := inputs["style"].(string)
	size := inputs["size"].(string)

	replicate := scripts.NewReplicateAI(app.Config().Replicate.APIKey)

	// Create generation
	gen, err := replicate.CreateRecraft(prompt, style, size)
	if err != nil {
		return nil, fmt.Errorf("failed to create generation: %w", err)
	}

	// Poll for completion
	finalGen, err := replicate.PollGeneration(gen.URLs.Get)
	if err != nil {
		return nil, fmt.Errorf("failed while polling generation: %w", err)
	}

	// Extract URLs from output
	var urls []string
	switch output := finalGen.Output.(type) {
	case []interface{}:
		for _, url := range output {
			if strURL, ok := url.(string); ok {
				urls = append(urls, strURL)
			}
		}
	case string:
		urls = append(urls, output)
	}

	return map[string]interface{}{
		"urls": urls,
	}, nil
}
