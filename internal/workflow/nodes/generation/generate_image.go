package generationnode

import (
	"context"
	"errors"
	"fmt"
	"image"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/equeue"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/worker"
	imagenode "github.com/cozy-creator/gen-server/internal/workflow/nodes/image"
)

type GenerateImageInputs struct {
	Model        string `json:"model"`
	NumImages    int    `json:"num_images"`
	RandomSeed   int    `json:"random_seed"`
	Prompt       string `json:"prompt"`
	OutputFormat string `json:"output_format"`
}

func GenerateImage(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	// log := logger.GetLogger()

	model := inputs["model"].(string)
	aspectRatio := inputs["aspect_ratio"].(string)
	outputFormat := inputs["output_format"].(string)
	numImages := int(inputs["num_images"].(float64))
	randomSeed := int(inputs["random_seed"].(float64))
	positivePrompt := inputs["positive_prompt"].(string)
	negativePrompt := inputs["negative_prompt"].(string)

	generateData := types.GenerateParams{
		RandomSeed:     randomSeed,
		AspectRatio:    aspectRatio,
		OutputFormat:   outputFormat,
		NegativePrompt: negativePrompt,
		PositivePrompt: positivePrompt,
		Models:         map[string]int{model: numImages},
	}

	requestId, err := worker.RequestGenerateImage(generateData)
	if err != nil {
		return nil, err
	}

	topic := config.DefaultGeneratePrefix + requestId
	queue := equeue.GetQueue("inmemory")

	output := make(chan []byte)
	errChan := make(chan error, 1)

	go func() {
		defer close(output)
		for {
			image, err := queue.Receive(context.Background(), topic)
			if err != nil {
				if errors.Is(err, equeue.ErrNoMessage) {
					continue
				}

				errChan <- err
				break
			}

			output <- image
		}
	}()

	images := make([]image.Image, 0, numImages)
	for len(images) < numImages {
		select {
		case err := <-errChan:
			return nil, err
		case <-context.Background().Done():
			return nil, fmt.Errorf("context cancelled")
		case out, ok := <-output:
			if !ok {
				break
			}

			// log.Info(fmt.Sprintf("image generated successfully, request id: %s", requestId))

			img, err := imagenode.DecodeImage(out, outputFormat)
			if err != nil {
				return nil, err
			}

			fmt.Println("Image generated and decoded successfully")

			// log.Info(fmt.Sprintf("image decoded to %s successfully", outputFormat))
			images = append(images, img)
		}
	}

	return map[string]interface{}{
		"images": images,
	}, nil
}
