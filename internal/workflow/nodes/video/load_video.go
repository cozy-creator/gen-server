package videonode

// import (
// 	"fmt"

// 	"gocv.io/x/gocv"
// )

// type Video struct {
// 	frames  int
// 	width   int
// 	height  int
// 	fps     float64
// 	capture *gocv.VideoCapture
// }

// func LoadVideo(filename string) (*Video, error) {
// 	v, err := loadVideo(filename)
// 	if err != nil {
// 		return nil, err
// 	}

// 	return v, nil
// }

// func loadVideo(filePath string) (*Video, error) {
// 	capture, err := gocv.VideoCaptureFile(filePath)
// 	if err != nil {
// 		return nil, fmt.Errorf("error opening video file: %w", err)
// 	}

// 	v := &Video{
// 		capture: capture,
// 		fps:     capture.Get(gocv.VideoCaptureFPS),
// 		frames:  int(capture.Get(gocv.VideoCaptureFrameCount)),
// 		width:   int(capture.Get(gocv.VideoCaptureFrameWidth)),
// 		height:  int(capture.Get(gocv.VideoCaptureFrameHeight)),
// 	}

// 	return v, nil
// }
