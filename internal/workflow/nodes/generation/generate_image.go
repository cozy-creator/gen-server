package generationnode

import (
	"bytes"
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

// type GenerateImageInputs struct {
// 	Model        string `json:"model"`
// 	NumOutputs    int    `json:"num_outputs"`
// 	RandomSeed   int    `json:"random_seed"`
// 	Prompt       string `json:"prompt"`
// 	OutputFormat string `json:"output_format"`
// }

// TO DO: this code is all wrong and unsafe; it will panic if the inputs are not what we expect
func GenerateImage(app *app.App, inputs map[string]interface{}) (map[string]interface{}, error) {
	model := inputs["model"].(string)
	aspectRatio := inputs["aspect_ratio"].(string)
	outputFormat := inputs["output_format"].(string)
	numOutputs := int(inputs["num_outputs"].(float64))
	randomSeed := int(inputs["random_seed"].(float64))
	positivePrompt := inputs["positive_prompt"].(string)
	negativePrompt := inputs["negative_prompt"].(string)

	fmt.Println("Output format: ", outputFormat)

	params := types.GenerateParamsRequest{
		Model:          model,
		NumOutputs:     &numOutputs,
		RandomSeed:     &randomSeed,
		AspectRatio:    aspectRatio,
		OutputFormat:   outputFormat,
		NegativePrompt: negativePrompt,
		PositivePrompt: positivePrompt,
	}

	reqParams, err := generation.NewRequest(params, app)
	if err != nil {
		return nil, err
	}

	images, err := receiveImages(reqParams.ID, app.Uploader(), app.MQ())
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"images": images,
	}, nil
}

func receiveImages(requestId string, uploader *fileuploader.Uploader, queue mq.MQ) ([]image.Image, error) {
	topic := config.DefaultGeneratePrefix + requestId

	images := make([]image.Image, 0)
	for {
		output, err := queue.Receive(context.Background(), topic)
		if err != nil {
			if errors.Is(err, mq.ErrTopicClosed) {
				break
			}
			return nil, err
		}

		outputData, err := queue.GetMessageData(output)
		if err != nil {
			return nil, err
		}

		if bytes.Equal(outputData, []byte("END")) {
			queue.CloseTopic(topic)
			break
		}

		outputData, _, _ = generation.ParseImageOutput(outputData)
		image, err := imagenode.DecodeImage(outputData, "bmp")
		if err != nil {
			fmt.Println("error: ", err)
			return nil, err
		}

		images = append(images, image)
	}

	fmt.Println("images: ", images)
	return images, nil
}
