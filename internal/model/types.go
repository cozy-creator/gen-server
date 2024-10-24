package model

type ModelCommand string

const (
    LoadModel   ModelCommand = "LOAD"
    UnloadModel ModelCommand = "UNLOAD"
    CheckModel  ModelCommand = "CHECK"
)

type ModelLocation string

const (
    GPU  ModelLocation = "GPU"
    CPU  ModelLocation = "CPU"
    None ModelLocation = "NONE"
)

type ModelRequest struct {
    Command ModelCommand `json:"command"`
    ModelID string      `json:"model_id"`
}

type ModelResponse struct {
    Success  bool          `json:"success"`
    Message  string       `json:"message"`
    Location ModelLocation `json:"location,omitempty"`
    Error    string       `json:"error,omitempty"`
}