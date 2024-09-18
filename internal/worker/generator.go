package worker

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/equeue"
	"github.com/cozy-creator/gen-server/internal/services"
	"github.com/cozy-creator/gen-server/internal/services/filehandler"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/utils/hashutil"
	"github.com/cozy-creator/gen-server/internal/utils/imageutil"

	"github.com/google/uuid"
)

func RequestGenerateImage(generateParams types.GenerateParams) (string, error) {
	requestId := uuid.NewString()
	params, err := json.Marshal(
		types.RequestGenerateParams{
			RequestId:      requestId,
			GenerateParams: generateParams,
		},
	)

	if err != nil {
		return "", fmt.Errorf("failed to marshal json data: %w", err)
	}

	// TODO: make this configurable and dynamic
	queue := equeue.GetQueue("inmemory")
	topic := config.DefaultGenerateTopic
	err = queue.Publish(context.Background(), topic, params)

	if err != nil {
		return "", fmt.Errorf("failed to publish message to queue: %w", err)
	}

	return requestId, nil
}

func ReceiveGenerateImage(requestId, outputFormat string) (string, error) {
	worker := GetUploadWorker()
	queue := equeue.GetQueue("inmemory")

	topic := config.DefaultGeneratePrefix + requestId
	image, err := queue.Receive(context.Background(), topic)
	if err != nil {
		return "", err
	}

	uploadUrl := make(chan string)

	go func() {
		extension := "." + outputFormat
		imageHash := hashutil.Blake3Hash(image)
		fileMeta := filehandler.FileInfo{
			Name:      imageHash,
			Extension: extension,
			Content:   image,
			IsTemp:    false,
		}

		worker.Upload(fileMeta, uploadUrl)
	}()

	url, ok := <-uploadUrl
	if !ok {
		return "", nil
	}

	return url, nil
}

func StartGeneration(ctx context.Context) error {
	queue := equeue.GetQueue("inmemory")

	for {
		message, err := queue.Receive(ctx, config.DefaultGenerateTopic)
		if err != nil {
			if errors.Is(err, equeue.ErrNoMessage) {
				continue
			}
			break
		}

		var data map[string]any
		if err := json.Unmarshal(message, &data); err != nil {
			log.Println("failed to unmarshal message:", err)
			continue
		}

		err = generateImage(data)
		queue.Ack(ctx, "generation", nil)
		if err != nil {
			return err
		}
	}

	return nil
}

func generateImage(data map[string]any) error {
	cfg := config.GetConfig()
	queue := equeue.GetQueue("inmemory")
	timeout := time.Duration(cfg.TcpTimeout) * time.Second
	fmt.Println(cfg.TcpPort)
	if cfg.TcpPort == 0 {
		cfg.TcpPort = 8881
		// return fmt.Errorf("tcp port is not set")
	}

	serverAddress := fmt.Sprintf("%s:%d", cfg.Host, cfg.TcpPort)

	jsonData, err := json.Marshal(data["params"])
	if err != nil {
		return fmt.Errorf("failed to marshal json data: %w", err)
	}

	client, err := services.NewTCPClient(serverAddress, timeout)
	if err != nil {
		return err
	}

	defer func() {
		if err := client.Close(); err != nil {
			fmt.Printf("Failed to close connection: %v\n", err)
		}
	}()

	client.Send(string(jsonData))
	requestId := data["request_id"].(string)
	format := data["params"].(map[string]any)["output_format"].(string)

	topic := config.DefaultGeneratePrefix + requestId
	defer queue.CloseTopic(topic)

	for {
		sizeBytes, err := client.ReceiveFullBytes(4)
		if err != nil {
			if err == io.EOF {
				break
			}

			return err
		}

		contentsize := binary.BigEndian.Uint32(sizeBytes)
		if contentsize == 0 {
			continue
		}

		// Receive the actual data based on the size
		response, err := (client.ReceiveFullBytes(int(contentsize)))
		if err != nil {
			if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
				break
			}

			break
		}

		imageBytes, err := imageutil.ConvertImageFromBitmap(response, format)
		if err != nil {
			return err
		}

		queue.Publish(context.Background(), topic, imageBytes)
	}

	return nil
}
