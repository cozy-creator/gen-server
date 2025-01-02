package types

type JobStatus string

const (
	StatusInProgress JobStatus = "IN_PROGRESS"
	StatusCompleted  JobStatus = "COMPLETED"
	StatusCancelled  JobStatus = "CANCELED"
	StatusInQueue    JobStatus = "IN_QUEUE"
	StatusFailed     JobStatus = "FAILED"
)

type EventType = string

const (
	StatusEventType EventType = "status"
	ErrorEventType  EventType = "error"
	OutputEventType EventType = "output"
)

type GenerationEvent struct {
	Type EventType   `msgpack:"type"`
	Data interface{} `msgpack:"data"`
}

type GenerationOutputData struct {
	Url       string `msgpack:"url"`
	JobID     string `msgpack:"job_id"`
	MimeType  string `msgpack:"mime_type"`
	FileBytes []byte `msgpack:"file_bytes"`
}

type GenerationStatusData struct {
	JobID        string    `msgpack:"job_id"`
	Status       JobStatus `msgpack:"status"`
	ErrorMessage string    `msgpack:"error_message"`
}

type GenerationErrorData struct {
	ErrorType    string `msgpack:"error_type"`
	ErrorMessage string `msgpack:"error_message"`
	JobID        string `msgpack:"job_id"`
}

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
	ID     string                 `json:"id"`
	Index  int8                   `json:"index"`
	Status JobStatus              `json:"status"`
	Output GeneratedOutput        `json:"output"`
	Input  *GenerateParamsRequest `json:"input,omitempty"`
}

// Request from client - no ID field
type GenerateParamsRequest struct {
	Model          string `json:"model" msgpack:"model"`
	NumOutputs     *int   `json:"num_outputs" msgpack:"num_outputs"`
	RandomSeed     *int   `json:"random_seed" msgpack:"random_seed"`
	AspectRatio    string `json:"aspect_ratio" msgpack:"aspect_ratio"`
	PositivePrompt string `json:"positive_prompt" msgpack:"positive_prompt"`
	NegativePrompt string `json:"negative_prompt,omitempty" msgpack:"negative_prompt,omitempty"`
	OutputFormat   string `json:"output_format" msgpack:"output_format"`
	PresignedURL   string `json:"presigned_url" msgpack:"presigned_url"`
	LoRAs         []LoRAParams `json:"loras,omitempty" msgpack:"loras,omitempty"`
	EnhancePrompt bool         `json:"enhance_prompt" msgpack:"enhance_prompt"`
}

// Internal type with server-generated ID
type GenerateParams struct {
	ID             string `json:"id" msgpack:"id"`
	Model          string `json:"model" msgpack:"model"`
	NumOutputs     int    `json:"num_outputs" msgpack:"num_outputs"`
	RandomSeed     int    `json:"random_seed" msgpack:"random_seed"`
	AspectRatio    string `json:"aspect_ratio" msgpack:"aspect_ratio"`
	PositivePrompt string `json:"positive_prompt" msgpack:"positive_prompt"`
	NegativePrompt string `json:"negative_prompt,omitempty" msgpack:"negative_prompt,omitempty"`
	OutputFormat   string `json:"output_format" msgpack:"output_format"`
	PresignedURL   string `json:"presigned_url" msgpack:"presigned_url"`
	LoRAs []LoRAParams `json:"loras,omitempty" msgpack:"loras,omitempty"`
}

type LoRAParams struct {
	URL      string  `json:"url" msgpack:"url"`
	Scale    float32 `json:"scale" msgpack:"scale"`
	FilePath string  `json:"file_path,omitempty" msgpack:"file_path,omitempty"`
}

type Video struct {
	Content interface{}
	Kind    string
}
