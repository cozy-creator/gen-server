package types

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

type GenerateParams struct {
	Models         map[string]int `json:"models"`
	RandomSeed     int            `json:"random_seed"`
	AspectRatio    string         `json:"aspect_ratio"`
	PositivePrompt string         `json:"positive_prompt"`
	NegativePrompt string         `json:"negative_prompt"`
	WebhookUrl     string         `json:"webhook_url"`
	OutputFormat   string         `json:"output_format"`
}

type RequestGenerateParams struct {
	GenerateParams GenerateParams `json:"params"`
	RequestId      string         `json:"request_id"`
	OutputFormat   string         `json:"output_format,omitempty"`
}
