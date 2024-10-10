package imageutil

import (
	"bytes"
	"fmt"
	"image/gif"
	"image/jpeg"
	"image/png"

	"golang.org/x/image/bmp"
)

func ConvertImageFromBitmap(bmpBytes []byte, format string) ([]byte, error) {
	fmt.Println("Image00: ", len(bmpBytes))
	img, err := bmp.Decode(bytes.NewReader(bmpBytes))
	if err != nil {
		return nil, err
	}

	var output bytes.Buffer
	switch format {
	case "png":
		err = png.Encode(&output, img)
	case "gif":
		err = gif.Encode(&output, img, nil)
	case "jpg":
		options := &jpeg.Options{Quality: 90}
		err = jpeg.Encode(&output, img, options)
	case "jpeg":
		options := &jpeg.Options{Quality: 90}
		err = jpeg.Encode(&output, img, options)
	default:
		fmt.Println("Default: ", "not supported", format)
		return nil, fmt.Errorf("not supported")
	}

	if err != nil {
		fmt.Println("Errrrrr: ", err)
		return nil, err
	}

	outputBytes := output.Bytes()
	fmt.Println("Image111: ", len(outputBytes))

	return outputBytes, nil
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
