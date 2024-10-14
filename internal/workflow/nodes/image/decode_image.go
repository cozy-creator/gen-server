package imagenode

import (
	"bytes"
	"fmt"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"

	"golang.org/x/image/bmp"
)

var (
	ErrInvalidFormat = image.ErrFormat
)

func DecodeImage(data []byte, format string) (image.Image, error) {
	var (
		output image.Image
		err    error
	)

	switch format {
	case "bmp":
		output, err = bmp.Decode(bytes.NewReader(data))
	case "png":
		output, err = png.Decode(bytes.NewReader(data))
	case "jpg", "jpeg":
		output, err = jpeg.Decode(bytes.NewReader(data))
	case "gif":
		output, err = gif.Decode(bytes.NewReader(data))
	default:
		return nil, fmt.Errorf("unsupported image format: %s", format)
	}

	return output, err
}
