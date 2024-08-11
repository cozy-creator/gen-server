package types

import "fmt"

const (
	FileResponseType = "file"
	DataResponseType = "data"
)

type FileResponse struct {
	Path string `json:"path"`
}

type ErrorResponse struct {
	Message string `json:"message"`
}

type UploadResponse struct {
	Url string `json:"url"`
}

type HandlerResponse struct {
	Type string      `json:"type"`
	Data interface{} `json:"data"`
}

func NewDataResponse(data interface{}) (*HandlerResponse, error) {
	return &HandlerResponse{
		Type: DataResponseType,
		Data: data,
	}, nil
}

func NewFileResponse(path string) (*HandlerResponse, error) {
	return &HandlerResponse{
		Type: FileResponseType,
		Data: FileResponse{
			Path: path,
		},
	}, nil
}

func NewErrorResponse(message string, a ...any) (*HandlerResponse, error) {
	return nil, fmt.Errorf(message, a...)
}
