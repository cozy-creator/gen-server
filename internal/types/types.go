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

type GenerateParams struct {
	ID             string         `json:"id,omitempty"`
	Models         map[string]int `json:"models"`
	Style          string         `json:"style,omitempty"`    // Added for Replicate
	Size           string         `json:"size,omitempty"`	// Added for Replicate
	RandomSeed     int            `json:"random_seed"`
	AspectRatio    string         `json:"aspect_ratio"`
	PositivePrompt string         `json:"positive_prompt"`
	NegativePrompt string         `json:"negative_prompt,omitempty"`
	WebhookUrl     string         `json:"webhook_url,omitempty"`
	OutputFormat   string         `json:"output_format"`
	SourceImage    interface{}    `json:"source_image,omitempty"`  // Added for image-to-image
	Strength       float32        `json:"strength,omitempty"`     // Added for image-to-image
}

type Video struct {
	Content interface{}
	Kind    string
}
