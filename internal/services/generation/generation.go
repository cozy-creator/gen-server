package generation

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/pkg/ethical_filter"
	"github.com/cozy-creator/gen-server/pkg/logger"
	"github.com/cozy-creator/gen-server/pkg/tcpclient"
	"github.com/google/uuid"
)

const (
	MaxWebhookAttempts = 3
)

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

		// handle loras if present
		if len(request.LoRAs) > 0 {
			fmt.Println("Downloading LoRAs")
			loraURLs := make([]string, len(request.LoRAs))
			for i, lora := range request.LoRAs {
				loraURLs[i] = lora.URL
			}

			lorasWithPaths, err := downloader.DownloadMultipleLoRAs(loraURLs)
			if err != nil {
				logger.Error("Failed to download LoRAs", err)
				continue
			}

			fmt.Println("LoRAs downloaded", lorasWithPaths)

			for i := range request.LoRAs {
				request.LoRAs[i].FilePath = lorasWithPaths[i]
			}
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

		fmt.Println(string(params))

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
	// check if enhanced prompt is enabled 
	// pipelineDef, exists := app.Config().PipelineDefs[params.Model]
	// if !exists {
	// 	return nil, fmt.Errorf("pipeline definition not found for model: %s", params.Model)
	// }

	// // fetch default positive prompt
	// defaultPrompt, ok := pipelineDef.DefaultArgs["positive_prompt"].(string)
	// if !ok {
	// 	defaultPrompt = "" 
	// }

	// // append default positive prompt if enhancePrompt is true
	// combinedPrompt := params.PositivePrompt
	// if params.EnhancePrompt && defaultPrompt != "" {
	// 	combinedPrompt = fmt.Sprintf("%s %s", defaultPrompt, params.PositivePrompt)
	// }

	// fmt.Println("combinedPrompt", combinedPrompt)
	
	newParams := types.GenerateParams{
		ID:             uuid.NewString(),
		OutputFormat:   params.OutputFormat,
		Model:          params.Model,
		NumOutputs:     func() int { if params.NumOutputs == nil { return 1 } else { return *params.NumOutputs } }(),
		RandomSeed:     func() int { if params.RandomSeed == nil { return rand.Intn(1 << 32) } else { return *params.RandomSeed } }(),
		AspectRatio:    params.AspectRatio,
		PositivePrompt: params.PositivePrompt,
		NegativePrompt: params.NegativePrompt,
		PresignedURL:   params.PresignedURL,
		LoRAs:          params.LoRAs,
		EnhancePrompt:  params.EnhancePrompt,
		Style:          params.Style,
	}

	mq := app.MQ()
	ctx := app.Context()

	// Use safety filter if it's enabled for every request
	if app.SafetyFilter != nil {
		response, err := app.SafetyFilter.EvaluatePrompt(ctx,  newParams.PositivePrompt, newParams.NegativePrompt)
		if err != nil {
			return nil, err
		}

		if response.Type == ethical_filter.PromptFilterResponseTypeRejected {
			return nil, fmt.Errorf("request rejected by safety filter: %s", response.Reason)
		}
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
