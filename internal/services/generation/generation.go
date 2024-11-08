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

		var request types.GenerateParamsRequest
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

func requestHandler(ctx context.Context, cfg *config.Config, data *types.GenerateParamsRequest) (chan []byte, chan error) {
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

func NewRequest(params types.GenerateParams, mq mq.MQ) (*types.GenerateParamsRequest, error) {
	if params.ID == "" {
		params.ID = uuid.NewString()
	}

	reqParams := types.GenerateParamsRequest{
		ID:             params.ID,
		WebhookUrl:     params.WebhookUrl,
		RandomSeed:     params.RandomSeed,
		AspectRatio:    params.AspectRatio,
		OutputFormat:   params.OutputFormat,
		PositivePrompt: params.PositivePrompt,
		NegativePrompt: params.NegativePrompt,
		Models:         map[string]int{params.Model: params.NumImages},
	}

	fmt.Println("reqParams: ", reqParams)

	data, err := json.Marshal(reqParams)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal json data: %w", err)
	}

	if err := mq.Publish(context.Background(), config.DefaultGenerateTopic, data); err != nil {
		return nil, fmt.Errorf("failed to publish message to queue: %w", err)
	}

	return &reqParams, nil
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
