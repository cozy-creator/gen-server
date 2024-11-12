package generation

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/utils/imageutil"
	"github.com/cozy-creator/gen-server/internal/utils/webhookutil"
	"github.com/vmihailenco/msgpack"

	"github.com/google/uuid"
)

func receiveImage(requestId string, outputFormat string, app *app.App) (string, string, error) {
	uploader := app.Uploader()
	mq := app.MQ()

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

		if err := processImageGen(ctx, params, app, sendResponse); err != nil {
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

	if err := processImageGen(ctx, params, app, sendResponse); err != nil {
		if !errors.Is(err, mq.ErrTopicClosed) {
			fmt.Println("Error processing image gen: ", err)
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

func processImageGen(ctx context.Context, params *types.GenerateParams, app *app.App, callback func(urls []string, index int8, currentModel, status string)) error {
	var (
		index int8
		urls  []string
	)

	if _, err := uuid.Parse(params.ID); err != nil {
		return err
	}
	job, err := app.JobsRepo.GetByID(app.Context(), params.ID)
	if err != nil {
		return err
	}

	if job.Status != models.JobStatusQueued {
		return fmt.Errorf("job is not in queue")
	}

	if err := app.JobsRepo.UpdateJobStatusByID(app.Context(), params.ID, models.JobStatusProgress); err != nil {
		return err
	}

	for {
		select {
		case <-ctx.Done():
			if len(urls) > 0 {
				callback(urls, index, params.Model, StatusCancelled)
			}
			return ctx.Err()
		default:
			url, _, err := receiveImage(params.ID, params.OutputFormat, app)
			if err != nil {
				statusEvent := GenerationEvent{
					Type: "status",
					Data: GenerationStatusData{
						JobID:        params.ID,
						Status:       StatusFailed,
						ErrorMessage: err.Error(),
					},
				}

				if err := publishEvent(params.ID, statusEvent, app); err != nil {
					return err
				}

				errorEvent := GenerationEvent{
					Type: "error",
					Data: GenerationErrorData{
						JobID:        params.ID,
						ErrorMessage: err.Error(),
						ErrorType:    "output_failed",
					},
				}

				if err := publishEvent(params.ID, errorEvent, app); err != nil {
					return err
				}
				return err
			}

			if url == "" {
				fmt.Println("gnot!!")
				time.Sleep(time.Second)
				if err := app.JobsRepo.UpdateJobStatusByID(app.Context(), params.ID, models.JobStatusCompleted); err != nil {
					return err
				}

				if err := app.MQ().Publish(app.Context(), config.DefaultStreamsTopic+"/"+params.ID, []byte("END")); err != nil {
					fmt.Println("Error publishing end message to MQ: ", err)
					return err
				}

				event := GenerationEvent{
					Type: "status",
					Data: GenerationStatusData{
						JobID:  params.ID,
						Status: StatusCompleted,
					},
				}

				if err := publishEvent(params.ID, event, app); err != nil {
					return err
				}

				callback(urls, index, params.Model, StatusCompleted)
				return nil
			} else {
				fmt.Println("got!!")
				urls = append(urls, url)
				imageData := models.Image{
					ID:    uuid.Must(uuid.NewRandom()),
					JobID: uuid.MustParse(params.ID),
					Url:   url,
				}

				event := GenerationEvent{
					Type: "output",
					Data: GenerationOutputData{
						Url:       url,
						JobID:     params.ID,
						MimeType:  "image/png",
						FileBytes: []byte{},
					},
				}

				if err := publishEvent(params.ID, event, app); err != nil {
					return err
				}

				if _, err := app.ImagesRepo.Create(app.Context(), &imageData); err != nil {
					fmt.Println("Error creating image: ", err)
					return err
				}
			}

			if len(urls) == cap(urls) {
				imgUrls := append([]string(nil), urls...)
				callback(imgUrls, index, params.Model, StatusInProgress)

				index++
				urls = nil
			}
		}
	}
}

func publishEvent(id string, event GenerationEvent, app *app.App) error {
	data, err := msgpack.Marshal(&event)
	if err != nil {
		return err
	}

	if err := app.MQ().Publish(app.Context(), config.DefaultStreamsTopic+"/"+id, data); err != nil {
		fmt.Println("Error publishing image to MQ: ", err)
		return err
	}

	return nil
}
