package imageutil

import (
	"bytes"
	"fmt"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"

	"golang.org/x/image/bmp"
)

// func ConvertImageFromBitmap(bmpBytes []byte, format string) ([]byte, error) {
// 	img, err := bmp.Decode(bytes.NewReader(bmpBytes))
// 	if err != nil {
// 		return nil, err
// 	}

// 	var output bytes.Buffer
// 	switch format {
// 	case "png":
// 		err = png.Encode(&output, img)
// 	case "gif":
// 		err = gif.Encode(&output, img, nil)
// 	case "jpg":
// 		options := &jpeg.Options{Quality: 90}
// 		err = jpeg.Encode(&output, img, options)
// 	case "jpeg":
// 		options := &jpeg.Options{Quality: 90}
// 		err = jpeg.Encode(&output, img, options)
// 	default:
// 		fmt.Println("Default: ", "not supported", format)
// 		return nil, fmt.Errorf("not supported")
// 	}

// 	if err != nil {
// 		fmt.Println("Errrrrr: ", err)
// 		return nil, err
// 	}

// 	outputBytes := output.Bytes()
// 	fmt.Println("Image111: ", len(outputBytes))

// 	return outputBytes, nil
// }

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

func DecodeBmpToFormat(data []byte, format string) ([]byte, error) {
	img, err := DecodeImage(data, "bmp")
	if err != nil {
		return nil, err
	}

	return EncodeImage(img, format)
}

func EncodeImage(img image.Image, format string) ([]byte, error) {
	var (
		content bytes.Buffer
		err     error
	)

	switch format {
	case "bmp":
		err = bmp.Encode(&content, img)
	case "png":
		err = png.Encode(&content, img)
	case "jpg", "jpeg":
		err = jpeg.Encode(&content, img, nil)
	case "gif":
		err = gif.Encode(&content, img, nil)
	default:
		return nil, ErrInvalidFormat
	}

	if err != nil {
		return nil, err
	}

	return content.Bytes(), nil
}
