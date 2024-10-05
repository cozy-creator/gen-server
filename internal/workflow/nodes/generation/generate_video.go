package generationnode

import (
	"context"
	"fmt"
	"image"
	"net/http"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/types"
	imagenode "github.com/cozy-creator/gen-server/internal/workflow/nodes/image"
	"github.com/cozy-creator/gen-server/scripts"
)

func GenerateVideo(app *app.App, inputs map[string]interface{}) (map[string]interface{}, error) {
	video_url, err := invokeLumaAI(app, inputs)
	if err != nil {
		return nil, err
	}

	httpClient := &http.Client{Timeout: 2 * time.Minute}
	req, err := http.NewRequestWithContext(context.Background(), "GET", video_url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}

	video := types.Video{
		Content: resp.Body,
		Kind:    types.ContentKindReader,
	}

	return map[string]interface{}{
		"video": video,
	}, nil
	// return nil, fmt.Errorf("video generation failed")
}

func invokeLumaAI(app *app.App, inputs map[string]interface{}) (string, error) {
	prompt := inputs["prompt"].(string)
	image_ := inputs["images"].([]image.Image)[0]

	imageInput := map[string]interface{}{
		"images":  []image.Image{image_},
		"format":  "png",
		"is_temp": false,
	}

	output, err := imagenode.SaveImage(app, imageInput)
	if err != nil {
		fmt.Println("Error saving image: ", err)
		return "", err
	}

	apiKey := app.Config().LumaAI.APIKey
	luma := scripts.NewLumaAI(apiKey)
	gen, err := luma.ImageToVideo(prompt, output["urls"].([]string)[0], "1:1", false)
	if err != nil {
		fmt.Println("Error invoking luma ai: ", err)
		return "", err
	}

	gen, err = pollGeneration(apiKey, gen.ID)
	if err != nil {
		fmt.Println("Error getting video...")
		return "", err
	}

	return gen.Assets.Video, nil
}

func pollGeneration(apiKey, id string) (*scripts.Generation, error) {
	luma := scripts.NewLumaAI(apiKey)
	for {
		res, err := luma.GetGeneration(id)
		if err != nil {
			fmt.Println("Error invoking luma ai: ", err)
			return nil, err
		}

		if res.State == "completed" {
			return res, nil
		}
	}
}
