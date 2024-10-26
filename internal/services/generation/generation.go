package generation

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/pkg/logger"
	"github.com/cozy-creator/gen-server/pkg/tcpclient"
	"github.com/google/uuid"
)

const (
	StatusInProgress = "IN_PROGRESS"
	StatusCompleted  = "COMPLETED"
	StatusCancelled  = "CANCELLED"
	StatusInQueue    = "IN_QUEUE"
	StatusFailed     = "FAILED"
)

const (
	MaxWebhookAttempts = 3
)

type ModelLoadRequest struct {
    Command     string   `json:"command"`      // "load", "unload", "warmup"
    ModelIDs    []string `json:"model_ids"`    // Models to operate on
    Priority    bool     `json:"priority"`     // Whether these are hot/priority models
    Force       bool     `json:"force"`        // Force load even if memory constrained
}

type ModelStatus struct {
    ModelID     string `json:"model_id"`
    IsLoaded    bool   `json:"is_loaded"`
    Location    string `json:"location"`    // "gpu", "cpu", "unloaded"
    MemoryUsage float64  `json:"memory_usage"`
}


func LoadModels(ctx context.Context, cfg *config.Config, modelIDs []string, priority bool) error {
    req := ModelLoadRequest{
        Command:  "load",
        ModelIDs: modelIDs,
        Priority: priority,
    }
    
    return sendModelCommand(ctx, cfg, req)
}

func WarmupModels(ctx context.Context, cfg *config.Config, modelIDs []string) error {
    req := ModelLoadRequest{
        Command:  "warmup",
        ModelIDs: modelIDs,
    }
    
    return sendModelCommand(ctx, cfg, req)
}

func UnloadModels(ctx context.Context, cfg *config.Config, modelIDs []string) error {
    req := ModelLoadRequest{
        Command:  "unload",
        ModelIDs: modelIDs,
    }
    
    return sendModelCommand(ctx, cfg, req)
}

func GetModelStatus(ctx context.Context, cfg *config.Config, modelIDs []string) ([]ModelStatus, error) {
    req := ModelLoadRequest{
        Command:  "status",
        ModelIDs: modelIDs,
    }
    
    data, err := sendModelCommandWithResponse(ctx, cfg, req)
    if err != nil {
        return nil, err
    }

    var statuses []ModelStatus
    if err := json.Unmarshal(data, &statuses); err != nil {
        return nil, fmt.Errorf("failed to parse model statuses: %w", err)
    }

    return statuses, nil
}

func sendModelCommand(ctx context.Context, cfg *config.Config, req ModelLoadRequest) error {
    timeout := time.Duration(500) * time.Second
    serverAddress := fmt.Sprintf("%s:%d", cfg.Host, cfg.TcpPort)
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
func sendModelCommandWithResponse(ctx context.Context, cfg *config.Config, req ModelLoadRequest) ([]byte, error) {
    timeout := time.Duration(500) * time.Second
    serverAddress := fmt.Sprintf("%s:%d", cfg.Host, cfg.TcpPort)
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


func RunProcessor(ctx context.Context, cfg *config.Config, mq mq.MQ) error {
	for {
		message, err := mq.Receive(ctx, config.DefaultGenerateTopic)
		if err != nil {
			return err
		}

		data, err := mq.GetMessageData(message)
		if err != nil {
			return err
		}

		var request types.GenerateParams
		if err := json.Unmarshal(data, &request); err != nil {
			logger.Error("Failed to parse request data", err)
			continue
		}

		outputs, errorc := requestHandler(ctx, cfg, &request)
		generationTopic := getGenerationTopic(request.ID)

		select {
		case err := <-errorc:
			mq.Publish(ctx, generationTopic, []byte("END"))
			return err
		case <-ctx.Done():
			mq.Publish(ctx, generationTopic, []byte("END"))
			break
		default:
			for output := range outputs {
				if err := mq.Publish(ctx, generationTopic, output); err != nil {
					return err
				}
			}
		}

		if err := mq.Publish(ctx, generationTopic, []byte("END")); err != nil {
			return fmt.Errorf("error publishing end message: %w", err)
		}
	}
}

func requestHandler(ctx context.Context, cfg *config.Config, data *types.GenerateParams) (chan []byte, chan error) {
	output := make(chan []byte)
	errorc := make(chan error, 1)

	go func() {
		defer close(output)
		params, err := json.Marshal(data)
		if err != nil {
			errorc <- err
			return
		}

		timeout := time.Duration(500) * time.Second
		// timeout := time.Duration(cfg.TcpTimeout) * time.Second
		serverAddress := fmt.Sprintf("%s:%d", cfg.Host, cfg.TcpPort)
		client, err := tcpclient.NewTCPClient(serverAddress, timeout, 1)
		if err != nil {
			errorc <- err
			return
		}

		defer func() {
			if err := client.Close(); err != nil {
				fmt.Printf("Failed to close connection: %v\n", err)
			}
		}()

		client.Send(ctx, string(params))
		for {
			sizeBytes, err := client.ReceiveFullBytes(ctx, 4)
			if handleReceiveError(err, errorc) {
				break
			}

			size := int(binary.BigEndian.Uint32(sizeBytes))
			if size != 0 {
				var outputBytes []byte
				outputBytes, err = client.ReceiveFullBytes(ctx, size)
				if handleReceiveError(err, errorc) {
					break
				}

				output <- outputBytes
			}
		}
	}()

	return output, errorc
}

func NewRequest(params *types.GenerateParams, mq mq.MQ) (string, error) {
	if params.ID == "" {
		params.ID = uuid.NewString()
	}
	// request := RequestGenerateParams{
	// 	RequestId:      params.ID,
	// 	GenerateParams: *params,
	// 	OutputFormat:   "png",
	// }

	data, err := json.Marshal(params)
	if err != nil {
		return "", fmt.Errorf("failed to marshal json data: %w", err)
	}

	err = mq.Publish(context.Background(), config.DefaultGenerateTopic, data)

	if err != nil {
		return "", fmt.Errorf("failed to publish message to queue: %w", err)
	}

	return params.ID, nil
}

func handleReceiveError(err error, errorc chan error) bool {
	if err != nil {
		if !(errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF)) {
			errorc <- err
		}
		return true
	}

	return false
}

func getGenerationTopic(requestId string) string {
	return config.DefaultGeneratePrefix + requestId
}
