package imageutil

import (
	"bytes"
	"image/jpeg"
	"image/png"

	"golang.org/x/image/bmp"
)

func ConvertImageFromBitmap(bmpBytes []byte, format string) ([]byte, error) {
	img, err := bmp.Decode(bytes.NewReader(bmpBytes))
	if err != nil {
		return nil, err
	}

	var output bytes.Buffer
	switch format {
	case "png":
		err = png.Encode(&output, img)
	case "jpg":
	case "jpeg":
		options := &jpeg.Options{Quality: 90}
		err = jpeg.Encode(&output, img, options)
	default:
		return nil, err
	}

	if err != nil {
		return nil, err
	}

	return output.Bytes(), nil
}

// func DecodeFromBitmap(bmpBytes []byte, format string) (image.Image, error) {
// 	img, err := bmp.Decode(bytes.NewReader(bmpBytes))
// 	if err != nil {
// 		return nil, err
// 	}

// 	var output bytes.Buffer
// 	switch format {
// 	case "png":
// 		err = png.Encode(&output, img)
// 	case "jpg":
// 	case "jpeg":
// 		options := &jpeg.Options{Quality: 90}
// 		err = jpeg.Encode(&output, img, options)
// 	case "gif":
// 		err = gif.Encode(&output, img, nil)
// 	default:
// 		return nil, err
// 	}

// 	if err != nil {
// 		return nil, err
// 	}

// 	return output, nil
// }
