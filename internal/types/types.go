package types

import "fmt"

const (
	FileResponseType = "file"
	JSONResponseType = "json"
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

type GenerateData struct {
	Models         map[string]int `json:"models"`
	WebhookUrl     *string        `json:"webhook_url"`
	RandomSeed     int            `json:"random_seed"`
	AspectRatio    string         `json:"aspect_ratio"`
	PositivePrompt string         `json:"positive_prompt"`
	NegativePrompt string         `json:"negative_prompt"`
}

func NewJSONResponse(data interface{}) (*HandlerResponse, error) {
	return &HandlerResponse{
		Type: JSONResponseType,
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