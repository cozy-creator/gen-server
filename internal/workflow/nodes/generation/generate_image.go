package generationnode

import (
	"context"
	"errors"
	"fmt"
	"image"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/types"
	imagenode "github.com/cozy-creator/gen-server/internal/workflow/nodes/image"
)

type GenerateImageInputs struct {
	Model        string `json:"model"`
	NumImages    int    `json:"num_images"`
	RandomSeed   int    `json:"random_seed"`
	Prompt       string `json:"prompt"`
	OutputFormat string `json:"output_format"`
}

func GenerateImage(app *app.App, inputs map[string]interface{}) (map[string]interface{}, error) {
	model := inputs["model"].(string)
	aspectRatio := inputs["aspect_ratio"].(string)
	outputFormat := inputs["output_format"].(string)
	numImages := int(inputs["num_images"].(float64))
	randomSeed := int(inputs["random_seed"].(float64))
	positivePrompt := inputs["positive_prompt"].(string)
	negativePrompt := inputs["negative_prompt"].(string)

	params := types.GenerateParams{
		RandomSeed:     randomSeed,
		AspectRatio:    aspectRatio,
		OutputFormat:   outputFormat,
		NegativePrompt: negativePrompt,
		PositivePrompt: positivePrompt,
		Models:         map[string]int{model: numImages},
	}

	_, err := generation.NewRequest(&params, false, app.MQ())
	if err != nil {
		return nil, err
	}

	fmt.Println("params: ", params)
	images, err := receiveImages(params.ID, params.OutputFormat, app.Uploader(), app.MQ())
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"images": images,
	}, nil

	// images := make([]image.Image, 0, numImages)
	// for len(images) < numImages {
	// 	select {
	// 	case err := <-errChan:
	// 		return nil, err
	// 	case <-context.Background().Done():
	// 		return nil, fmt.Errorf("context cancelled")
	// 	case out, ok := <-output:
	// 		if !ok {
	// 			break
	// 		}

	// 		// log.Info(fmt.Sprintf("image generated successfully, request id: %s", requestId))

	// 		fmt.Println("Image generated and decoded successfully")

	// 		// log.Info(fmt.Sprintf("image decoded to %s successfully", outputFormat))
	// 		images = append(images, img)
	// 	}
	// }

	// return map[string]interface{}{
	// 	"images": images,
	// }, nil
}

func receiveImages(requestId string, outputFormat string, uploader *fileuploader.Uploader, queue mq.MQ) ([]image.Image, error) {
	topic := config.DefaultGeneratePrefix + requestId

	images := make([]image.Image, 0)
	fmt.Println("topic: ", topic)
	for {
		output, err := queue.Receive(context.Background(), topic)
		if err != nil {
			if errors.Is(err, mq.ErrTopicClosed) {
				break
			}
			return nil, err
		}

		image, err := imagenode.DecodeImage(output, "bmp")
		if err != nil {
			fmt.Println("error: ", err)
			return nil, err
		}

		fmt.Println("image: ", "decoded image")
		images = append(images, image)
	}

	fmt.Println("images: ", images)
	return images, nil
}
