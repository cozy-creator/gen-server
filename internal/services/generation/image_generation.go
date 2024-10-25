package generation

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/utils/imageutil"
	"github.com/cozy-creator/gen-server/internal/utils/webhookutil"
	"github.com/cozy-creator/gen-server/pkg/logger"
)

func receiveImage(requestId string, outputFormat string, uploader *fileuploader.Uploader, mq mq.MQ) (string, string, error) {
	generationTopic := getGenerationTopic(requestId)
	output, err := mq.Receive(context.Background(), generationTopic)
	if err != nil {
		return "", "", err
	}

	outputData, err := mq.GetMessageData(output)
	if err != nil {
		return "", "", err
	}

	if bytes.Equal(outputData, []byte("END")) {
		logger.Info("Received end message...")
		mq.CloseTopic(generationTopic)
		return "", "", nil
	}

	outputData, modelName, err := ParseImageOutput(outputData)
	image, err := imageutil.DecodeBmpToFormat(outputData, outputFormat)
	if err != nil {
		return "", "", err
	}

	uploadUrl := make(chan string)
	go func() {
		extension := "." + "png"
		uploader.UploadBytes(image, extension, false, uploadUrl)
	}()

	url, ok := <-uploadUrl
	if !ok {
		return "", "", nil
	}

	return url, modelName, nil
}

func GenerateImageSync(app *app.App, params *types.GenerateParams) (chan types.GenerationResponse, error) {
	ctx := app.Context()
	errc := make(chan error, 1)
	outputc := make(chan types.GenerationResponse)

	sendResponse := func(urls []string, index int8, currentModel string, status string) {
		if len(urls) > 0 {
			outputc <- types.GenerationResponse{
				Output: types.GeneratedOutput{
					URLs:  urls,
					Model: currentModel,
				},
				Index:  index,
				Status: status,
			}
		}
	}

	go func() {
		defer func() {
			close(outputc)
			close(errc)
		}()

		if err := processImageGen(ctx, params, app.Uploader(), app.MQ(), sendResponse); err != nil {
			errc <- err
		}
	}()

	select {
	case err := <-errc:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return outputc, nil
	}
}

func GenerateImageAsync(app *app.App, params *types.GenerateParams) {
	ctx := app.Context()
	ctx, _ = context.WithTimeout(ctx, 5*time.Minute)
	invoke := func(response types.GenerationResponse) {
		if err := webhookutil.InvokeWithRetries(ctx, params.WebhookUrl, response, MaxWebhookAttempts); err != nil {
			fmt.Println("Failed to invoke webhook:", err)
		}
	}

	sendResponse := func(urls []string, index int8, currentModel, status string) {
		response := types.GenerationResponse{
			Index:  index,
			Input:  params,
			Status: status,
			ID:     params.ID,
			Output: types.GeneratedOutput{
				URLs:  urls,
				Model: currentModel,
			},
		}

		invoke(response)
	}

	if err := processImageGen(ctx, params, app.Uploader(), app.MQ(), sendResponse); err != nil {
		if !errors.Is(err, mq.ErrTopicClosed) {
			invoke(types.GenerationResponse{Status: StatusFailed})
		}
	}
}

func readModelName(buffer *bytes.Buffer) (string, error) {
	var modelNameSize uint32
	if err := binary.Read(buffer, binary.BigEndian, &modelNameSize); err != nil {
		return "", err
	}

	if buffer.Len() < int(modelNameSize) {
		return "", fmt.Errorf("buffer does not contain enough data")
	}

	modelName := make([]byte, modelNameSize)
	if err := binary.Read(buffer, binary.BigEndian, &modelName); err != nil {
		return "", err
	}
	return string(modelName), nil
}

func ParseImageOutput(output []byte) ([]byte, string, error) {
	outputBuffer := bytes.NewBuffer(output)
	modelName, err := readModelName(outputBuffer)
	if err != nil {
		return nil, "", err
	}

	return outputBuffer.Bytes(), modelName, nil
}

func processImageGen(ctx context.Context, params *types.GenerateParams, uploader *fileuploader.Uploader, queue mq.MQ, callback func(urls []string, index int8, currentModel, status string)) error {
	var (
		index        int8
		currentModel string
		urls         []string
	)

	for {
		select {
		case <-ctx.Done():
			if len(urls) > 0 {
				callback(urls, index, currentModel, StatusCancelled)
			}
			return ctx.Err()
		default:
			fmt.Println("Receiving image... iii")
			url, model, err := receiveImage(params.ID, params.OutputFormat, uploader, queue)
			fmt.Println("url....", url)

			if err != nil {
				fmt.Println("Error receiving image111: ", err)
				if errors.Is(err, mq.ErrTopicClosed) {
					// sleep
					time.Sleep(time.Second)
					callback(urls, index, currentModel, StatusCompleted)
					return nil
				}

				fmt.Println("Error receiving image: ", err)
				return err
			}

			if url == "" && model == "" {
				time.Sleep(time.Second)
				callback(urls, index, currentModel, StatusCompleted)
				return nil
			}

			fmt.Println("Received image: ", url)

			if currentModel == model {
				urls = append(urls, url)
				if len(urls) == cap(urls) {
					imgUrls := append([]string(nil), urls...)
					callback(imgUrls, index, currentModel, StatusInProgress)

					index++
					urls = nil
					currentModel = ""
				}
			} else if currentModel == "" {
				numImages, ok := params.Models[model]
				if !ok {
					return fmt.Errorf("model %s not found in models", model)
				}

				urls = make([]string, 0, numImages)
				urls = append(urls, url)
				currentModel = model
			}
		}
	}
}
