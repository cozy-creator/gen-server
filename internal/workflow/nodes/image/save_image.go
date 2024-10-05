package imagenode

import (
	"bytes"
	"fmt"
	"image"

	"image/gif"
	"image/jpeg"
	"image/png"

	"github.com/cozy-creator/gen-server/internal/app"
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
	format := input["format"].(string)
	isTemp := input["is_temp"].(bool)

	var (
		urls []string
		err  error
	)

	for _, img := range images {
		var content bytes.Buffer
		switch format {
		case "png":
			err = png.Encode(&content, img)
		case "jpg":
		case "jpeg":
			err = jpeg.Encode(&content, img, nil)
		case "gif":
			err = gif.Encode(&content, img, nil)
		default:
			return nil, ErrInvalidFormat
		}

		if err != nil {
			return nil, err
		}

		response := make(chan string)
		extension := "." + format
		go app.Uploader().UploadBytes(content.Bytes(), extension, isTemp, response)

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
