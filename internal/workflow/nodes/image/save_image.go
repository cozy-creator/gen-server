package imagenode

import (
	"fmt"
	"image"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/utils/imageutil"
)

type SaveImageInputs struct {
	Images []image.Image `json:"images"`
	Format string        `json:"format"`
	IsTemp bool          `json:"is_temp"`
}

type SaveImageOutput struct {
	Urls []string `json:"urls"`
}

func SaveImage(app *app.App, input map[string]interface{}) (map[string]interface{}, error) {
	images := input["images"].([]image.Image)
	format := input["output_format"].(string)
	isTemp := input["is_temp"].(bool)

	var urls []string
	for _, img := range images {
		content, err := imageutil.EncodeImage(img, format)
		if err != nil {
			return nil, fmt.Errorf("failed to encode image: %w", err)
		}

		response := make(chan string)
		extension := "." + format
		go app.Uploader().UploadBytes(content, extension, isTemp, response)

		url, ok := <-response
		if !ok {
			return nil, fmt.Errorf("failed to upload image")
		}

		urls = append(urls, url)
	}

	return map[string]interface{}{
		"urls": urls,
	}, nil
}
