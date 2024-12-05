package generation

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/ethicalfilter"
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

type GenerationEventType = string

const (
	OutputType GenerationEventType = "output"
	StatusType GenerationEventType = "status"
	ErrorType  GenerationEventType = "error"
)

type GenerationEvent struct {
	Type GenerationEventType `msgpack:"type"`
	Data interface{}         `msgpack:"data"`
}

type GenerationOutputData struct {
	Url       string `msgpack:"url"`
	JobID     string `msgpack:"job_id"`
	MimeType  string `msgpack:"mime_type"`
	FileBytes []byte `msgpack:"file_bytes"`
}

type GenerationStatusData struct {
	Status       string `msgpack:"status"`
	JobID        string `msgpack:"job_id"`
	ErrorMessage string `msgpack:"error_message"`
}

type GenerationErrorData struct {
	ErrorType    string `msgpack:"error_type"`
	ErrorMessage string `msgpack:"error_message"`
	JobID        string `msgpack:"job_id"`
}

func RunProcessor(ctx context.Context, cfg *config.Config, mq mq.MQ, app *app.App, downloader types.ModelDownloader) error {
	// start downloading queue processor
	go processDownloadingQueue(ctx, mq, downloader)

	// main processing loop for ready models
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

		modelState := downloader.GetModelState(request.Model)
		generationTopic := getGenerationTopic(request.ID)

		// If model is downloading, move to downloading queue
		if modelState == types.ModelStateDownloading {
			if err := mq.Publish(ctx, config.DefaultDownloadingTopic, data); err != nil {
				logger.Error("Failed to move request to downloading queue", err)
			}
			continue
		}

		// process normal request
		outputs, errorc := requestHandler(ctx, cfg, &request)

		select {
		case err := <-errorc:
			if err != nil {
				logger.Error("Error in request handler", err)
				if pubErr := mq.Publish(ctx, generationTopic, []byte("END")); pubErr != nil {
					return fmt.Errorf("error publishing end message after error: %w", pubErr)
				}
				continue
			}
		case <-ctx.Done():
			if err := mq.Publish(ctx, generationTopic, []byte("END")); err != nil {
				return fmt.Errorf("error publishing end message on shutdown: %w", err)
			}
			return ctx.Err()
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

func processDownloadingQueue(ctx context.Context, mq mq.MQ, downloader types.ModelDownloader) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
			message, err := mq.Receive(ctx, config.DefaultDownloadingTopic)
			if err != nil {
				logger.Error("Error receiving from downloading queue", err)
				continue
			}

			data, err := mq.GetMessageData(message)
			if err != nil {
				logger.Error("Error getting message data", err)
				continue
			}

			var request types.GenerateParams
			if err := json.Unmarshal(data, &request); err != nil {
				logger.Error("Failed to parse request data", err)
				continue
			}

			// waiting for model to be ready
			if err := downloader.WaitForModelReady(ctx, request.Model); err != nil {
				logger.Error("Failed waiting for model", err)
				continue
			}

			// move back to main queue
			if err := mq.Publish(ctx, config.DefaultGenerateTopic, data); err != nil {
				logger.Error("Failed to requeue request", err)
			}
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
		serverAddress := fmt.Sprintf("%s:%d", cfg.Host, config.TCPPort)
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

func NewRequest(params types.GenerateParamsRequest, app *app.App) (*types.GenerateParams, error) {
	newParams := types.GenerateParams{
		ID:             uuid.NewString(),
		OutputFormat:   params.OutputFormat,
		Model:          params.Model,
		NumOutputs:     params.NumOutputs,
		RandomSeed:     params.RandomSeed,
		AspectRatio:    params.AspectRatio,
		PositivePrompt: params.PositivePrompt,
		NegativePrompt: params.NegativePrompt,
		PresignedURL:   params.PresignedURL,
	}

	mq := app.MQ()
	cfg := app.Config()
	ctx := app.Context()
	response, err := ethicalfilter.FilterPrompt(ctx, cfg, newParams.PositivePrompt, newParams.NegativePrompt)
	fmt.Println("filterres")
	fmt.Println(response)
	if err != nil {
		return nil, err
	}

	if response.Type == ethicalfilter.PromptFilterResponseTypeRejected {
		return nil, fmt.Errorf("rejected by ethical filter: %s", response.Reason)
	}

	data, err := json.Marshal(&newParams)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal json data: %w", err)
	}

	if err := mq.Publish(context.Background(), config.DefaultGenerateTopic, data); err != nil {
		return nil, fmt.Errorf("failed to publish message to queue: %w", err)
	}

	return &newParams, nil
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

func getStreamTopic(requestId string) string {
	return config.DefaultStreamsTopic + "/" + requestId
}
