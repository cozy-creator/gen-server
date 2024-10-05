package videonode

import (
	"errors"
	"fmt"
	"io"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/types"
)

type SaveVideoNode struct{}

var (
	ErrInvalidVideoKind = errors.New("invalid video kind")
)

func SaveVideo(app *app.App, inputs map[string]interface{}) (map[string]interface{}, error) {
	fmt.Println("saving video: ", inputs)

	video := inputs["video"].(types.Video)
	isTemp := inputs["is_temp"].(bool)

	urlc := make(chan string)
	if video.Kind == types.ContentKindReader {
		app.Uploader().UploadReader(video.Content.(io.Reader), ".mp4", isTemp, urlc)
	} else if video.Kind == types.ContentKindBytes {
		app.Uploader().UploadBytes(video.Content.([]byte), ".mp4", isTemp, urlc)
	} else {
		return nil, ErrInvalidVideoKind
	}

	return map[string]interface{}{
		"url": <-urlc,
	}, nil
}
