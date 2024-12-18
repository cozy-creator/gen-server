package models

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/pkg/tcpclient"
)

type ModelLoadRequest struct {
	Force    bool     `json:"force"`
	Command  string   `json:"command"`
	ModelIDs []string `json:"model_ids"`
	Priority bool     `json:"priority"`
}

type ModelStatus struct {
	Location    string  `json:"location"`
	ModelID     string  `json:"model_id"`
	IsLoaded    bool    `json:"is_loaded"`
	MemoryUsage float64 `json:"memory_usage"`
}

func LoadModels(app *app.App, modelIDs []string, priority bool) error {
	req := ModelLoadRequest{
		Command:  "load",
		ModelIDs: modelIDs,
		Priority: priority,
	}

	return sendModelCommand(app, req)
}

func EnableModels(app *app.App, modelIDs []string) error {
	req := ModelLoadRequest{
		Command:  "enable",
		ModelIDs: modelIDs,
	}

	return sendModelCommand(app, req)
}

func UnloadModels(app *app.App, modelIDs []string) error {
	req := ModelLoadRequest{
		Command:  "unload",
		ModelIDs: modelIDs,
	}

	return sendModelCommand(app, req)
}

func GetModelStatus(app *app.App, modelIDs []string) ([]ModelStatus, error) {
	req := ModelLoadRequest{
		Command:  "status",
		ModelIDs: modelIDs,
	}

	data, err := sendModelCommandWithResponse(app, req)
	if err != nil {
		return nil, err
	}

	var statuses []ModelStatus
	if err := json.Unmarshal(data, &statuses); err != nil {
		return nil, fmt.Errorf("failed to parse model statuses: %w", err)
	}

	return statuses, nil
}

func sendModelCommand(app *app.App, req ModelLoadRequest) error {
	timeout := time.Duration(500) * time.Second
	ctx := app.Context()
	cfg := app.Config()

	serverAddress := fmt.Sprintf("%s:%d", cfg.Host, config.TCPPort)
	client, err := tcpclient.NewTCPClient(serverAddress, timeout, 1)
	if err != nil {
		return fmt.Errorf("failed to create TCP client: %w", err)
	}
	defer client.Close()

	// Marshall request data
	data, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send request data
	if err := client.Send(ctx, string(data)); err != nil {
		return fmt.Errorf("failed to send command: %w", err)
	}

	// Read the size prefix
	sizePrefix, err := client.ReceiveFullBytes(ctx, 4)
	if err != nil {
		return fmt.Errorf("failed to receive size prefix: %w", err)
	}
	size := binary.BigEndian.Uint32(sizePrefix)

	// Read the full response based on size
	resp, err := client.ReceiveFullBytes(ctx, int(size))
	if err != nil {
		return fmt.Errorf("failed to receive full response: %w", err)
	}
	fmt.Printf("Response: %s\n", resp)

	// Parse the JSON response
	var response struct {
		Status  string `json:"status"`
		Error   string `json:"error,omitempty"`
		Message string `json:"message,omitempty"`
	}
	if err := json.Unmarshal(resp, &response); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	// Handle error response from server
	if response.Status == "error" {
		return fmt.Errorf("command failed: %s", response.Error)
	}

	return nil
}
func sendModelCommandWithResponse(app *app.App, req ModelLoadRequest) ([]byte, error) {
	timeout := time.Duration(500) * time.Second
	ctx := app.Context()
	cfg := app.Config()

	serverAddress := fmt.Sprintf("%s:%d", cfg.Host, config.TCPPort)
	client, err := tcpclient.NewTCPClient(serverAddress, timeout, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to create TCP client: %w", err)
	}
	defer client.Close()

	// Marshal request data
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send request data
	if err := client.Send(ctx, string(data)); err != nil {
		return nil, fmt.Errorf("failed to send command: %w", err)
	}

	// Receive size prefix
	sizePrefix, err := client.ReceiveFullBytes(ctx, 4)
	if err != nil {
		return nil, fmt.Errorf("failed to receive size prefix: %w", err)
	}
	size := binary.BigEndian.Uint32(sizePrefix)

	// Receive the full response based on size
	resp, err := client.ReceiveFullBytes(ctx, int(size))
	if err != nil {
		return nil, fmt.Errorf("failed to receive full response: %w", err)
	}

	return resp, nil
}
