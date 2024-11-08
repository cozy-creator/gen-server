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
	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/utils/imageutil"
	"github.com/cozy-creator/gen-server/internal/utils/webhookutil"
	"github.com/cozy-creator/gen-server/pkg/logger"
	"github.com/google/uuid"
	"github.com/vmihailenco/msgpack"
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

func GenerateImageSync(app *app.App, params *types.GenerateParamsRequest) (chan types.GenerationResponse, error) {
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

func GenerateImageAsync(app *app.App, params *types.GenerateParamsRequest) {
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

func processImageGen(ctx context.Context, params *types.GenerateParamsRequest, app *app.App, callback func(urls []string, index int8, currentModel, status string)) error {
	var (
		index        int8
		currentModel string
		urls         []string
	)

	job, err := app.JobsRepo.Get(app.Context(), uuid.MustParse(params.ID))
	if err != nil {
		return err
	}

	if job.Status != db.JobStatusEnumINQUEUE {
		return fmt.Errorf("job is not in queue")
	}

	updateArg := db.UpdateJobStatusParams{ID: uuid.MustParse(params.ID), Status: db.JobStatusEnumINPROGRESS}
	if err := app.JobsRepo.UpdateStatus(app.Context(), updateArg); err != nil {
		fmt.Println("Error updating job status: ", err)
		return err
	}

	for {
		select {
		case <-ctx.Done():
			if len(urls) > 0 {
				callback(urls, index, currentModel, StatusCancelled)
			}
			fmt.Println("Error receiving image: ", err)

			return ctx.Err()
		default:
			url, model, err := receiveImage(params.ID, params.OutputFormat, app)
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) {
					if err := app.MQ().Publish(app.Context(), config.DefaultStreamsTopic+"/"+params.ID, []byte("END")); err != nil {
						fmt.Println("Error publishing end message to MQ: ", err)
						return err
					}
					callback(urls, index, currentModel, StatusCompleted)
					return nil
				}

				fmt.Println("Error receiving image: ", err)

				return err
			}

			fmt.Println("url....", url)

			if url != "" {
				fmt.Println("ddddd: ", url)
				createArgs := db.CreateImageParams{
					ID:    uuid.Must(uuid.NewRandom()),
					JobID: uuid.MustParse(params.ID),
					Url:   url,
				}

				output := struct {
					Url   string `msgpack:"url"`
					Model string `msgpack:"model"`
					JobID string `msgpack:"job_id"`
				}{
					Url:   url,
					Model: model,
					JobID: params.ID,
				}

				mapO := struct {
					Type string      `msgpack:"type"`
					Data interface{} `msgpack:"data"`
				}{
					Type: "output",
					Data: output,
				}

				fmt.Println("mapO: ", mapO)

				data, err := msgpack.Marshal(&mapO)
				if err != nil {
					fmt.Println("Error marshaling output: ", err)
					return err
				}

				if err := app.MQ().Publish(app.Context(), config.DefaultStreamsTopic+"/"+params.ID, data); err != nil {
					fmt.Println("Error publishing image to MQ: ", err)
					return err
				}

				if _, err := app.ImagesRepo.Create(app.Context(), createArgs); err != nil {
					fmt.Println("Error creating image: ", err)
					return err
				}
			}

			if url == "" && model == "" {
				time.Sleep(time.Second)
				updateArg := db.UpdateJobStatusParams{Status: db.JobStatusEnumCOMPLETED, ID: uuid.MustParse(params.ID)}
				if err := app.JobsRepo.UpdateStatus(app.Context(), updateArg); err != nil {
					return err
				}

				callback(urls, index, currentModel, StatusCompleted)
				return nil
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
					fmt.Println("Error receiving image: ", "model not found in models")
					return fmt.Errorf("model %s not found in models", model)
				}

				urls = make([]string, 0, numImages)
				urls = append(urls, url)
				currentModel = model
			}
		}
	}
}
