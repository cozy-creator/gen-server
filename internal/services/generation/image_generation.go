package generation

import (
	"context"
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

func receiveImage(requestId string, outputFormat string, uploader *fileuploader.Uploader, mq mq.MQ) (string, error) {
	topic := config.DefaultGeneratePrefix + requestId
	output, err := mq.Receive(context.Background(), topic)
	image, _ := imageutil.ConvertImageFromBitmap(output, outputFormat)

	if err != nil {
		fmt.Println("Error receiving image from queue:", err)
		return "", err
	}

	uploadUrl := make(chan string)

	go func() {
		extension := "." + "png"
		uploader.UploadBytes(image, extension, false, uploadUrl)
	}()

	url, ok := <-uploadUrl
	if !ok {
		return "", nil
	}

	return url, nil
}

func GenerateImageSync(ctx context.Context, params *types.GenerateParams, uploader *fileuploader.Uploader, queue mq.MQ) (chan string, error) {
	urlc := make(chan string)
	errc := make(chan error, 1)

	go func() {
		defer func() {
			close(urlc)
			close(errc)
		}()

		for {
			url, err := receiveImage(params.ID, params.OutputFormat, uploader, queue)
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) {
					errc <- nil
				}

				errc <- err
				break
			}

			if url != "" {
				urlc <- url
			}
		}
	}()

	select {
	case err := <-errc:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return urlc, nil
	}
}

func GenerateImageAsync(ctx context.Context, params *types.GenerateParams, uploader *fileuploader.Uploader, queue mq.MQ) {
	ctx, _ = context.WithTimeout(ctx, 5*time.Minute)
	// defer cancel()

	invoke := func(response AsyncGenerateResponse) {
		if err := webhookutil.InvokeWithRetries(ctx, params.WebhookUrl, response, MaxWebhookAttempts); err != nil {
			fmt.Println("Failed to invoke webhook:", err)
		}
	}

	index := 0
	for {
		select {
		case <-ctx.Done():
			return
		default:
			url, err := receiveImage(params.ID, params.OutputFormat, uploader, queue)
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) {
					response := AsyncGenerateResponse{
						ID:     params.ID,
						Status: StatusCompleted,
						Output: []AsyncGenerateResponseOutput{},
					}

					go invoke(response)
					break
				} else {
					response := AsyncGenerateResponse{
						ID:     params.ID,
						Status: StatusFailed,
						Output: []AsyncGenerateResponseOutput{},
					}

					go invoke(response)
				}

				return
			}

			if url != "" {
				response := AsyncGenerateResponse{
					Index:  index,
					ID:     params.ID,
					Output: []AsyncGenerateResponseOutput{{Format: "png", URL: url}},
				}

				go invoke(response)
				index++
			}
		}
	}
}
