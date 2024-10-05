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
	"github.com/cozy-creator/gen-server/pkg/tcpclient"
	"github.com/google/uuid"
)

const (
	StatusInProgress = "IN_PROGRESS"
	StatusCompleted  = "COMPLETED"
	StatusInQueue    = "IN_QUEUE"
	StatusFailed     = "FAILED"
)

const (
	MaxWebhookAttempts = 3
)

type AsyncGenerateResponse struct {
	ID     string                        `json:"id"`
	Index  int                           `json:"index"`
	Output []AsyncGenerateResponseOutput `json:"output"`
	Status string                        `json:"status,omitempty"`
}

type AsyncGenerateResponseOutput struct {
	Format string `json:"format"`
	URL    string `json:"url"`
}

func RunProcessor(ctx context.Context, cfg *config.Config, mq mq.MQ) error {
	for {
		message, err := mq.Receive(ctx, config.DefaultGenerateTopic)
		if err != nil {
			return err
		}

		request, err := parseRequestData(message)
		if err != nil {
			continue
		}

		outputs, errorc := requestHandler(ctx, cfg, request.GenerateParams)
		topic := config.DefaultGeneratePrefix + request.RequestId

		select {
		case err := <-errorc:
			mq.CloseTopic(topic)
			return err
		case <-ctx.Done():
			break
		default:
			for output := range outputs {
				mq.Publish(context.Background(), topic, output)
			}
		}

		mq.CloseTopic(topic)
	}
}

func requestHandler(ctx context.Context, cfg *config.Config, data types.GenerateParams) (chan []byte, chan error) {
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
			if err != nil {
				if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
					break
				}

				errorc <- err
				break
			}

			size := binary.BigEndian.Uint32(sizeBytes)
			if size == 0 {
				continue
			}

			// Receive the actual data based on the size
			outputBytes, err := (client.ReceiveFullBytes(ctx, int(size)))
			if err != nil {
				if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
					break
				}

				errorc <- err
				break
			}

			output <- outputBytes
		}

	}()

	return output, errorc
}

func NewRequest(params *types.GenerateParams, saveOutput bool, mq mq.MQ) (string, error) {
	if params.ID == "" {
		params.ID = uuid.NewString()
	}
	request := types.RequestGenerateParams{
		RequestId:      params.ID,
		GenerateParams: *params,
		OutputFormat:   "png",
	}

	data, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal json data: %w", err)
	}

	err = mq.Publish(context.Background(), config.DefaultGenerateTopic, data)

	if err != nil {
		return "", fmt.Errorf("failed to publish message to queue: %w", err)
	}

	return params.ID, nil
}

func parseRequestData(message []byte) (types.RequestGenerateParams, error) {
	var request types.RequestGenerateParams
	if err := json.Unmarshal(message, &request); err != nil {
		return types.RequestGenerateParams{}, fmt.Errorf("failed to unmarshal request data: %w", err)
	}

	return request, nil
}
