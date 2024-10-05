package imagenode

import (
	"bytes"
	"image"

	"image/gif"
	"image/jpeg"
	"image/png"

	"github.com/cozy-creator/gen-server/internal/utils/imageutil"
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

	reader := bytes.NewReader(data)
	imageutil.ConvertImageFromBitmap(data, format)
	switch format {
	case "bmp":
		output, err = bmp.Decode(reader)
	case "png":
		output, err = png.Decode(reader)
	case "gif":
		output, err = gif.Decode(reader)
	case "jpg":
		output, err = jpeg.Decode(reader)
	case "jpeg":
		output, err = jpeg.Decode(reader)
	default:
		return nil, ErrInvalidFormat
	}

	if err != nil {
		return nil, err
	}

	return output, nil
}
