package imagenode

import (
	"context"
	"errors"
	"image"

	"github.com/anthonynsimon/bild/transform"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
)

type SizeHandling int

const (
	UniformSize SizeHandling = iota
	Resize
	BatchBySize
)

var (
	ErrUniformSize         = errors.New("images must have uniform size")
	ErrInvalidSizeHandling = errors.New("invalid size handling")
)

type ImageNodeInputs struct {
	Filenames    []string     `json:"filenames"`
	SizeHandling SizeHandling `json:"size_handling"`
	TargetSize   image.Point  `json:"target_size"`
}

type ImageNodeOutput struct {
	Images []image.Image `json:"images"`
}

func LoadImage(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	filenames, err := getFilenames(inputs)
	if err != nil {
		return nil, err
	}

	sizeHandling, err := getSizeHandling(inputs)
	if err != nil {
		return nil, err
	}

	targetSize, err := getTargetSize(inputs)
	if err != nil {
		return nil, err
	}

	var images []image.Image
	for _, filename := range filenames {
		image, err := loadImageFromFile(filename)
		if err != nil {
			return nil, err
		}

		images = append(images, image)
	}

	switch sizeHandling {
	case UniformSize:
		if !hasUniformSize(images) {
			return nil, ErrUniformSize
		}
	case Resize:
		resized := make([]image.Image, 0, len(images))
		for i, image := range images {
			image, err := resizeImage(image, targetSize)
			if err != nil {
				return nil, err
			}

			resized[i] = image
		}

		images = resized
	case BatchBySize:
		grouped, err := groupImagesBySize(images)
		if err != nil {
			return nil, err
		}

		images = grouped
	default:
		return nil, ErrInvalidSizeHandling
	}

	// return &ImageNodeOutput{
	// 	Images: images,
	// }, nil

	return map[string]interface{}{
		"images": images,
	}, nil
}

func hasUniformSize(images []image.Image) bool {
	if len(images) == 0 {
		return true
	}

	var initialWidth, initialHeight int
	for _, image := range images {
		size := image.Bounds().Size()

		width := size.X
		height := size.Y
		if initialWidth == 0 && initialHeight == 0 {
			initialWidth, initialHeight = width, height
		} else {
			if initialWidth != width || initialHeight != height {
				return false
			}
		}
	}

	return true
}

func resizeImage(image image.Image, targetSize image.Point) (image.Image, error) {
	resized := transform.Resize(image, targetSize.X, targetSize.Y, transform.Linear)
	return resized, nil
}

func groupImagesBySize(images []image.Image) ([]image.Image, error) {
	groups := make(map[image.Point][]image.Image)

	for _, image := range images {
		size := image.Bounds().Size()
		groups[size] = append(groups[size], image)
	}

	images = make([]image.Image, 0)
	for _, group := range groups {
		images = append(images, group...)
	}

	return images, nil
}

func loadImageFromFile(filename string) (image.Image, error) {
	// handler, err := filehandler.GetFileHandler()
	// if err != nil {
	// 	return nil, err
	// }

	// file, err := handler.GetFile(filename)
	// if err != nil {
	// 	fmt.Println("error decoding config0:", err)
	// 	return nil, err
	// }

	// reader := bytes.NewReader(file.Content)
	// image, _, err := image.Decode(reader)
	// if err != nil {
	// 	fmt.Println("error decoding config2:", err)
	// 	return nil, err
	// }

	return nil, nil
}

func getFilenames(inputs map[string]interface{}) ([]string, error) {
	filenames := inputs["filenames"].([]interface{})
	var filenamesStr []string
	for _, filename := range filenames {
		filenamesStr = append(filenamesStr, filename.(string))
	}

	return filenamesStr, nil
}

func getSizeHandling(inputs map[string]interface{}) (SizeHandling, error) {
	sizeHandling := SizeHandling(inputs["size_handling"].(float64))
	switch sizeHandling {
	case UniformSize:
		return UniformSize, nil
	case Resize:
		return Resize, nil
	case BatchBySize:
		return BatchBySize, nil
	default:
		return 0, ErrInvalidSizeHandling
	}
}

func getTargetSize(inputs map[string]interface{}) (image.Point, error) {
	target := inputs["target_size"].(map[string]interface{})

	return image.Point{
		X: int(target["x"].(float64)),
		Y: int(target["y"].(float64)),
	}, nil
}
