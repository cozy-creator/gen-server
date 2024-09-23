package imagenode

import (
	"bytes"
	"context"
	"fmt"
	"image"

	"image/gif"
	"image/jpeg"
	"image/png"

	"github.com/cozy-creator/gen-server/pkg/logger"
)

type SaveImageInputs struct {
	Images []image.Image `json:"images"`
	Format string        `json:"format"`
	IsTemp bool          `json:"is_temp"`
}

type SaveImageOutput struct {
	Urls []string `json:"urls"`
}

func SaveImage(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	log := logger.GetLogger()
	images := input["images"].([]image.Image)
	format := input["format"].(string)
	// isTemp := input["is_temp"].(bool)

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

		// extension := "." + format
		// contentBytes := content.Bytes()
		// contentHash := hashutil.Blake3Hash(contentBytes)
		// _ := filehandler.FileInfo{
		// 	Name:      contentHash,
		// 	Extension: extension,
		// 	Content:   contentBytes,
		// 	IsTemp:    isTemp,
		// }

		response := make(chan string)
		// worker := worker.GetUploadWorker()
		// go worker.Upload(fileMeta, response)

		url := <-response
		urls = append(urls, url)
	}

	// return &SaveImageOutput{
	// 	Urls: urls,
	// }, nil

	log.Info(fmt.Sprintf("%d images saved successfully", len(urls)))

	return map[string]interface{}{
		"urls": urls,
	}, nil
}
