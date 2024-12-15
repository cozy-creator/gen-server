package generation

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/utils/imageutil"
	"github.com/cozy-creator/gen-server/pkg/logger"
	"github.com/uptrace/bun"
	"github.com/vmihailenco/msgpack"

	"github.com/google/uuid"
)

func receiveImage(params *types.GenerateParams, app *app.App) (string, string, error) {
	uploader := app.Uploader()
	mq := app.MQ()

	generationTopic := getGenerationTopic(params.ID)
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

	outputData, modelName, _ := ParseImageOutput(outputData)
	image, err := imageutil.DecodeBmpToFormat(outputData, params.OutputFormat)
	if err != nil {
		return "", "", err
	}

	uploadUrl := make(chan string)
	go func() {
		if params.PresignedURL != "" {
			uploader.UploadBytesPresigned(image, params.PresignedURL, uploadUrl)
		} else {
			extension := "." + "png"
			uploader.UploadBytes(image, extension, false, uploadUrl)
		}
	}()

	url, ok := <-uploadUrl
	if !ok {
		return "", "", nil
	}

	return url, modelName, nil
}

// This function is not currently used
func GenerateImageSync(app *app.App, params *types.GenerateParams) (chan types.GenerationResponse, error) {
	ctx := app.Context()
	errc := make(chan error, 1)
	outputc := make(chan types.GenerationResponse)

	sendResponse := func(urls []string, index int8, currentModel string, status types.JobStatus) {
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
	// TO DO: do not discard cancel here
	ctx, _ = context.WithTimeout(ctx, 5*time.Minute)
	// invoke := func(response types.GenerationResponse) {
	// 	if err := webhookutil.InvokeWithRetries(ctx, params.WebhookUrl, response, MaxWebhookAttempts); err != nil {
	// 		fmt.Println("Failed to invoke webhook:", err)
	// 	}
	// }

	sendResponse := func(urls []string, index int8, currentModel string, status types.JobStatus) {
		// 	response := types.GenerationResponse{
		// 		Index:  index,
		// 		Input:  params,
		// 		Status: status,
		// 		ID:     params.ID,
		// 		Output: types.GeneratedOutput{
		// 			URLs:  urls,
		// 			Model: currentModel,
		// 		},
		// 	}

		// invoke(response)
	}

	if err := processImageGen(ctx, params, app, sendResponse); err != nil {
		if !errors.Is(err, mq.ErrTopicClosed) {
			fmt.Println("Error processing image gen: ", err)
			// invoke(types.GenerationResponse{Status: StatusFailed})
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

func processImageGen(
    ctx context.Context,
    params *types.GenerateParams,
    app *app.App,
    callback func(
        urls []string,
        index int8,
        currentModel string,
        status types.JobStatus,
    ),
) error {
	var (
		index int8
		urls  = make([]string, 0, params.NumOutputs)  	// For collecting all URLs
        batch = make([]string, 0, params.NumOutputs) 	// For progress tracking (i.e. the batch of urls we've received)
	)

	if err := handleGenerationBegin(app, params.ID); err != nil {
		// logger.Error("error handling generation begin: ", err)
		return err
	}

	for {
		select {
		case <-ctx.Done():
			if len(urls) > 0 {
				callback(urls, index, params.Model, types.StatusCancelled)
			}
			return ctx.Err()
		default:
			url, _, err := receiveImage(params, app)
			fmt.Println("url: ", url)
			if err != nil {
				if err := handleGenerationError(app, params.ID, err.Error(), "output_failed"); err != nil {
					logger.Error("error handling generation error: ", err)
					return err
				}
				logger.Error("error receiving image: ", err)
				return err
			}

			if url == "" {                
                // if we have any successful outputs, mark as completed
                if len(urls) > 0 {
                    if err := handleGenerationCompletion(app, params.ID); err != nil {
                        logger.Error("error handling generation completion: ", err)
                        return err
                    }
                    callback(urls, index, params.Model, types.StatusCompleted)
                    return nil
                }

                // if no URLs, it means generation failed
                if err := handleGenerationError(app, params.ID, "No valid outputs received", "no_output"); err != nil {
                    logger.Error("error handling generation error: ", err)
                }
                return fmt.Errorf("no valid outputs received from generation")
            }

            // Successfully received a URL
            urls = append(urls, url)
            batch = append(batch, url)

            if err := handleImageOutput(app, params.ID, url, "image/png"); err != nil {
                logger.Error("error handling image output: ", err)
                return err
            }

			// if batch is full, send progress update
            if len(batch) == params.NumOutputs {
                batchCopy := make([]string, len(batch))
                copy(batchCopy, batch)
                callback(batchCopy, index, params.Model, types.StatusInProgress)
                index++
                batch = batch[:0]
            }
		}
	}
}

func publishStatusEvent(app *app.App, tx *bun.Tx, id string, status types.JobStatus, errMsg string) error {
	eventData := types.GenerationStatusData{
		JobID:        id,
		Status:       status,
		ErrorMessage: errMsg,
	}

	ctx := app.Context()
	event := models.NewEvent(uuid.MustParse(id), types.StatusEventType, eventData)
	if tx != nil {
		if _, err := app.EventRepository.WithTx(tx).Create(ctx, event); err != nil {
			if err := tx.Rollback(); err != nil {
				logger.Error("Error rolling back transaction:", err.Error())
				return err
			}
			logger.Error("error creating event: ", err)
			return err
		}
	} else {
		if _, err := app.EventRepository.Create(ctx, event); err != nil {
			logger.Error("error creating event: ", err)
			return err
		}
	}

	data, err := msgpack.Marshal(&types.GenerationEvent{Type: types.StatusEventType, Data: eventData})
	if err != nil {
		logger.Error("error marshaling event: ", err)
		return err
	}
	if err := app.MQ().Publish(ctx, getStreamTopic(id), data); err != nil {
		logger.Error("error publishing event: ", err)
		return err
	}

	// print message data
	// stateMsg := fmt.Sprintf(`Message: {"data": {"error_message": "%s", "job_id": "%s", "status": "%s"}, "type": "status"}`, 
    //     errMsg,
    //     id, 
    //     status,
    // )
    // fmt.Println(stateMsg)

	return nil
}

func handleImageOutput(app *app.App, id, url, mimeType string) error {
	ctx := app.Context()
	tx, err := app.DB().BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	// Ensure rollback in case of error
	var committed bool
	defer func() {
		if !committed {
			if err := tx.Rollback(); err != nil {
				logger.Error("Error rolling back transaction:", err.Error())
			}
		}
	}()

	image := models.NewImage(url, uuid.MustParse(id), mimeType)
	if _, err := app.ImageRepository.WithTx(&tx).Create(ctx, image); err != nil {
		logger.Error("error creating image: ", err)
		return err
	}

	eventData := types.GenerationOutputData{
		Url:       url,
		JobID:     id,
		FileBytes: []byte{},
		MimeType:  mimeType,
	}
	event := models.NewEvent(uuid.MustParse(id), types.OutputEventType, eventData)
	if _, err := app.EventRepository.WithTx(&tx).Create(ctx, event); err != nil {
		if err := tx.Rollback(); err != nil {
			logger.Error("Error rolling back transaction:", err.Error())
			return err
		}
		logger.Error("error creating event in output: ", err)
		return err
	}

	if err := tx.Commit(); err != nil {
		logger.Error("Error committing transaction:", err.Error())
		return err
	}

	committed = true

	data, err := msgpack.Marshal(&types.GenerationEvent{Data: eventData, Type: types.OutputEventType})
	if err != nil {
		logger.Error("error marshaling event: ", err)
		return err
	}

	if err := app.MQ().Publish(ctx, getStreamTopic(id), data); err != nil {
		logger.Error("error publishing event: ", err)
		return err
	}

	return nil
}

func handleGenerationCompletion(app *app.App, id string) error {
	ctx := app.Context()
	tx, err := app.DB().BeginTx(ctx, nil)
	if err != nil {
		logger.Error("error beginning transaction: ", err)
		return err
	}

	if err := app.JobRepository.WithTx(&tx).UpdateJobStatusByID(ctx, id, types.StatusCompleted); err != nil {
		if err := tx.Rollback(); err != nil {
			logger.Error("Error rolling back transaction:", err.Error())
			return err
		}
		return err
	}

	if err := publishStatusEvent(app, &tx, id, types.StatusCompleted, ""); err != nil {
		logger.Error("error publishing status event: ", err)
		return err
	}

	if err := tx.Commit(); err != nil {
		logger.Error("Error committing transaction:", err.Error())
		return err
	}

	if err := app.MQ().Publish(ctx, getStreamTopic(id), []byte("END")); err != nil {
		logger.Error("error publishing end event: ", err)
		return err
	}

	return nil
}

func handleGenerationError(app *app.App, id, errMsg, errType string) error {
	ctx := app.Context()
	tx, err := app.DB().BeginTx(ctx, nil)
	if err != nil {
		logger.Error("error beginning transaction: ", err)
		return err
	}

	// ensure rollback in case of error
    var committed bool
    defer func() {
        if !committed {
            if err := tx.Rollback(); err != nil {
                logger.Error("Error rolling back transaction:", err.Error())
            }
        }
    }()

	eventData := types.GenerationErrorData{
		JobID:        id,
		ErrorType:    errType,
		ErrorMessage: errMsg,
	}

	if err := app.JobRepository.WithTx(&tx).UpdateJobStatusByID(ctx, id, types.StatusFailed); err != nil {
		if err := tx.Rollback(); err != nil {
			logger.Error("Error rolling back transaction:", err.Error())
			return err
		}

		logger.Error("error updating job status in output: ", err)
		return err
	}

	event := models.NewEvent(uuid.MustParse(id), types.ErrorEventType, eventData)
	if _, err := app.EventRepository.WithTx(&tx).Create(ctx, event); err != nil {
		if err := tx.Rollback(); err != nil {
			logger.Error("Error rolling back transaction:", err.Error())
			return err
		}

		logger.Error("error creating event in output: ", err)
		return err
	}

	if err := publishStatusEvent(app, &tx, id, types.StatusFailed, errMsg); err != nil {
		logger.Error("error publishing status event in output: ", err)
		return err
	}

	data, err := msgpack.Marshal(&types.GenerationEvent{Type: types.ErrorEventType, Data: eventData})
	if err != nil {
		logger.Error("error marshaling event in output: ", err)
		return err
	}

	if err := app.MQ().Publish(ctx, getStreamTopic(id), data); err != nil {
		logger.Error("error publishing event in output: ", err)
		return err
	}

	// commit the transaction
	if err := tx.Commit(); err != nil {
		logger.Error("Error committing transaction:", err.Error())
		return err
	}

	committed = true

	return nil
}

func handleGenerationBegin(app *app.App, id string) error {
	ctx := app.Context()
	job, err := app.JobRepository.GetByID(ctx, id)
	if err != nil {
		fmt.Println("Error getting job: ", err)
		return err
	}

	if job.Status != types.StatusInQueue {
		logger.Error("job is not in queue")
		return fmt.Errorf("job is not in queue")
	}

	tx, err := app.DB().BeginTx(ctx, nil)
	if err != nil {
		logger.Error("error beginning transaction in gen begin: ", err)
		return err
	}

	if err := app.JobRepository.WithTx(&tx).UpdateJobStatusByID(ctx, id, types.StatusInProgress); err != nil {
		if err := tx.Rollback(); err != nil {
			logger.Error("Error rolling back transaction:", err.Error())
			return err
		}

		logger.Error("error updating job status in gen begin: ", err)
		return err
	}

	if err := publishStatusEvent(app, &tx, id, types.StatusInProgress, ""); err != nil {
		logger.Error("error publishing status event in gen begin: ", err)
		return err
	}

	if err := tx.Commit(); err != nil {
		logger.Error("Error committing transaction:", err.Error())
		return err
	}

	return nil
}
