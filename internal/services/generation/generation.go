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
	StatusCancelled  = "CANCELLED"
	StatusInQueue    = "IN_QUEUE"
	StatusFailed     = "FAILED"
)

const (
	MaxWebhookAttempts = 1
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

type GeneratedOutput struct {
	ModelName string `json:"model_name"`
	Output    []byte `json:"output"`
}

func RunProcessor(ctx context.Context, cfg *config.Config, mq mq.MQ) error {
	for {
		message, err := mq.Receive(ctx, config.DefaultGenerateTopic)
		if err != nil {
			return err
		}

		messageData, err := mq.GetMessageData(message)
		if err != nil {
			return err
		}

		request, err := parseRequestData(messageData)
		if err != nil {
			continue
		}

		outputs, errorc := requestHandler(ctx, cfg, request.GenerateParams)
		topic := config.DefaultGeneratePrefix + request.RequestId

		select {
		case err := <-errorc:
			mq.Publish(ctx, topic, []byte("END"))
			return err
		case <-ctx.Done():
			break
		default:
			for output := range outputs {
				mq.Publish(ctx, topic, output)
			}
		}

		// Send a termination message to the queue
		mq.Publish(ctx, topic, []byte("END"))
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

func receiveWithSize(ctx context.Context, client *tcpclient.TCPClient, expectedSize int) ([]byte, error, bool) {
	data, err := client.ReceiveFullBytes(ctx, expectedSize)
	if err != nil {
		// Handle EOF or UnexpectedEOF gracefully
		if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
			return nil, err, true
		}

		return nil, err, false
	}

	return data, nil, false
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
