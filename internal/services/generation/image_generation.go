package generation

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/utils/imageutil"
	"github.com/cozy-creator/gen-server/internal/utils/webhookutil"
)

type GeneratedImage struct {
	URLs      []string `json:"urls"`
	Index     int8     `json:"index"`
	ID        string   `json:"id,omitempty"`
	Status    string   `json:"status,omitempty"`
	ModelName string   `json:"model_name,omitempty"`
}

func receiveImage(requestId string, outputFormat string, uploader *fileuploader.Uploader, mq mq.MQ) (string, string, error) {
	topic := config.DefaultGeneratePrefix + requestId
	output, err := mq.Receive(context.Background(), topic)
	if err != nil {
		return "", "", err
	}

	output, modelName, err := ParseImageOutput(output)
	image, err := imageutil.DecodeBmpToFormat(output, outputFormat)
	if err != nil {
		return "", "", err
	}

	uploadUrl := make(chan string)

	go func() {
		extension := "." + "png"
		uploader.UploadBytes(image, extension, false, uploadUrl)
	}()

	url, ok := <-uploadUrl
	fmt.Println("URL--: ", url)
	if !ok {
		return "", "", nil
	}

	return url, modelName, nil
}

func GenerateImageSync(ctx context.Context, params *types.GenerateParams, uploader *fileuploader.Uploader, queue mq.MQ) (chan GeneratedImage, error) {
	outputc := make(chan GeneratedImage)
	errc := make(chan error, 1)

	sendResponse := func(urls []string, index int8, currentModel string, status string) {
		if len(urls) > 0 {
			outputc <- GeneratedImage{
				URLs:      urls,
				Index:     index,
				Status:    status,
				ModelName: currentModel,
			}
		}
	}

	go func() {
		defer func() {
			close(outputc)
			close(errc)
		}()

		if err := processImageGen(ctx, params, uploader, queue, sendResponse); err != nil {
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

func GenerateImageAsync(ctx context.Context, params *types.GenerateParams, uploader *fileuploader.Uploader, queue mq.MQ) {
	ctx, _ = context.WithTimeout(ctx, 5*time.Minute)

	invoke := func(response GeneratedImage) {
		if err := webhookutil.InvokeWithRetries(ctx, params.WebhookUrl, response, MaxWebhookAttempts); err != nil {
			fmt.Println("Failed to invoke webhook:", err)
		}
	}

	sendResponse := func(urls []string, index int8, currentModel, status string) {
		response := GeneratedImage{
			ID:        params.ID,
			URLs:      urls,
			Index:     index,
			ModelName: currentModel,
			Status:    status,
		}

		invoke(response)
	}

	if err := processImageGen(ctx, params, uploader, queue, sendResponse); err != nil {
		if !errors.Is(err, mq.ErrTopicClosed) {
			response := GeneratedImage{
				Status: StatusFailed,
			}

			invoke(response)
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
			url, model, err := receiveImage(params.ID, params.OutputFormat, uploader, queue)
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) {
					// sleep
					time.Sleep(time.Second)
					callback(urls, index, currentModel, StatusCompleted)
					return nil
				}

				return err
			}

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
