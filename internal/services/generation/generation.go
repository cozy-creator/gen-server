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
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/utils/hashutil"
	"github.com/cozy-creator/gen-server/internal/utils/imageutil"
	"github.com/cozy-creator/gen-server/pkg/tcpclient"
	"github.com/google/uuid"
)

func StartGenerationRequestProcessor(ctx context.Context, cfg *config.Config, mq mq.MQ) error {
	for {
		message, err := mq.Receive(ctx, "generation")
		if err != nil {
			return err
		}

		request, err := parseRequestData(message)
		if err != nil {
			continue
		}

		outputs, errorc := requestHandler(ctx, cfg, request.GenerateParams)
		topic := config.DefaultGeneratePrefix + request.RequestId

		select {
		case err := <-errorc:
			fmt.Println("Error:", err)
			mq.CloseTopic(topic)
			return err
		case <-ctx.Done():
			mq.CloseTopic(topic)
			break
		default:
			for output := range outputs {
				image, _ := imageutil.ConvertImageFromBitmap(output, request.OutputFormat)
				mq.Publish(context.Background(), topic, image)
			}
		}
		mq.CloseTopic(topic)
		return nil
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
			if err != nil {
				if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
					break
				}

				errorc <- err
				break
			}

			size := binary.BigEndian.Uint32(sizeBytes)
			if size == 0 {
				continue
			}

			// Receive the actual data based on the size
			imageBytes, err := (client.ReceiveFullBytes(ctx, int(size)))
			if err != nil {
				if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
					break
				}

				errorc <- err
				break
			}

			output <- imageBytes
		}

	}()

	return output, errorc
}

func NewRequest(params types.GenerateParams, mq mq.MQ) (string, error) {
	requestId := uuid.NewString()
	request := types.RequestGenerateParams{
		RequestId:      requestId,
		GenerateParams: params,
		OutputFormat:   "png",
	}

	data, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal json data: %w", err)
	}

	topic := config.DefaultGenerateTopic
	err = mq.Publish(context.Background(), topic, data)

	if err != nil {
		return "", fmt.Errorf("failed to publish message to queue: %w", err)
	}

	return requestId, nil
}

func ReceiveImage(requestId string, uploader *fileuploader.Uploader, mq mq.MQ) (string, error) {
	topic := config.DefaultGeneratePrefix + requestId
	image, err := mq.Receive(context.Background(), topic)
	if err != nil {
		fmt.Println("Error receiving image from queue:", err)
		return "", err
	}

	uploadUrl := make(chan string)

	go func() {
		extension := "." + "png"
		imageHash := hashutil.Blake3Hash(image)
		fileMeta := filestorage.FileInfo{
			Name:      imageHash,
			Extension: extension,
			Content:   image,
			IsTemp:    false,
		}

		uploader.Upload(fileMeta, uploadUrl)
	}()

	url, ok := <-uploadUrl
	if !ok {
		return "", nil
	}

	return url, nil
}

func parseRequestData(message []byte) (types.RequestGenerateParams, error) {
	var request types.RequestGenerateParams
	if err := json.Unmarshal(message, &request); err != nil {
		return types.RequestGenerateParams{}, fmt.Errorf("failed to unmarshal request data: %w", err)
	}

	return request, nil
}
