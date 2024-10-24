package types

const (
	FileResponseType = "file"
	JSONResponseType = "json"
)

const (
	ContentKindReader = "reader"
	ContentKindString = "string"
	ContentKindBytes  = "bytes"
)

type GenerateParams struct {
	ID             string         `json:"id"`
	Models         map[string]int `json:"models"`
	RandomSeed     int            `json:"random_seed"`
	AspectRatio    string         `json:"aspect_ratio"`
	PositivePrompt string         `json:"positive_prompt"`
	NegativePrompt string         `json:"negative_prompt,omitempty"`
	WebhookUrl     string         `json:"webhook_url"`
	OutputFormat   string         `json:"output_format"`
}

type RequestGenerateParams struct {
	GenerateParams GenerateParams `json:"params"`
	RequestId      string         `json:"request_id"`
	SaveOutput     bool           `json:"save_output"`
	OutputFormat   string         `json:"output_format,omitempty"`
}

type Video struct {
	Content interface{}
	Kind    string
}

type GeneratedOutput struct {
	URLs  []string `json:"urls"`
	Model string   `json:"model"`
}

type GenerationResponse struct {
	ID     string          `json:"id"`
	Index  int8            `json:"index"`
	Status string          `json:"status"`
	Output GeneratedOutput `json:"output"`
	Input  *GenerateParams `json:"input,omitempty"`
}
