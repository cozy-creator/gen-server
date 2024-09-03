package worker

import (
	"context"
	"cozy-creator/gen-server/internal/config"
	"cozy-creator/gen-server/internal/services"
	"cozy-creator/gen-server/internal/types"
	"cozy-creator/gen-server/internal/utils"
	"cozy-creator/gen-server/pkg/mq"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/google/uuid"
)

var inMemoryQueue = mq.GetDefaultInMemoryQueue()

func RequestGenerateImage(data types.GenerateData) (string, error) {
	requestId := uuid.NewString()

	bytes, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to marshal json data: %w", err)
	}

	err = inMemoryQueue.Publish(context.Background(), "generation", bytes)
	fmt.Println("message published to queue:", err)
	if err != nil {
		fmt.Println("failed to publish message to queue:", err)
		return "", fmt.Errorf("failed to publish message to queue: %w", err)
	}

	return requestId, nil
}

func ReceiveGenerateImage(requestId, outputFormat string) (string, error) {
	uploader := GetUploadWorker()
	topic := fmt.Sprintf("generation/%s", requestId)
	image, err := inMemoryQueue.Receive(context.Background(), topic)

	if err != nil {
		fmt.Println("failed to receive message from queue:", err)
		return "", fmt.Errorf("failed to receive message from queue: %w", err)
	}

	if image == nil {
		return "", fmt.Errorf("no image")
	}

	uploadUrl := make(chan string)

	go func() {
		imageHash := utils.Blake3Hash(image)
		extension := fmt.Sprintf(".%s", outputFormat)
		fileMeta := services.NewFileMeta(hex.EncodeToString(imageHash[:]), extension, image, false)

		uploader.Upload(fileMeta, uploadUrl)
	}()

	url, ok := <-uploadUrl
	if !ok {
		return "", nil
	}

	return url, nil
}

func StartGeneration(ctx context.Context) error {
	queue := mq.GetDefaultInMemoryQueue()

	for {
		message, err := queue.Receive(ctx, "generation")
		if err != nil {
			log.Println("failed to receive message from queue:", err)
			break
		}

		if message == nil {
			continue
		}

		var data map[string]any
		if err := json.Unmarshal(message, &data); err != nil {
			log.Println("failed to unmarshal message:", err)
			continue
		}

		err = generateImage(data)
		queue.Ack(ctx, "generation", nil)
		if err != nil {
			log.Println("failed to generate image:", err)
			continue
		}
		fmt.Println("image generated successfully, request id:", data["request_id"])
	}

	return nil
}

func generateImage(data map[string]any) error {
	cfg := config.GetConfig()
	queue := mq.GetDefaultInMemoryQueue()
	timeout := time.Duration(cfg.TcpTimeout) * time.Second
	serverAddress := fmt.Sprintf("%s:%d", cfg.Host, cfg.TcpPort)

	jsonData, err := json.Marshal(data)
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
	format := data["format"].(string)

	topic := fmt.Sprintf("generation/%s", requestId)
	// defer queue.CloseTopic(topic)

	for {
		sizeBytes, err := client.ReceiveFullBytes(4)
		// Receive the size of the incoming data (4 bytes for size)
		if err != nil {
			if err == io.EOF {
				fmt.Println("EOF reached")
				break // End of stream
			}
			fmt.Printf("Error reading size header: %v\n", err)
			break
		}

		contentsize := binary.BigEndian.Uint32(sizeBytes)
		if contentsize == 0 {
			fmt.Println("Received a chunk with size 0, skipping")
			continue
		}

		// Receive the actual data based on the size
		response, err := (client.ReceiveFullBytes(int(contentsize)))
		if err != nil {
			if errors.Is(err, io.ErrUnexpectedEOF) {
				fmt.Println("Unexpected EOF reached while reading data")
				break
			}
			if errors.Is(err, io.EOF) {
				fmt.Println("EOF reached while reading data")
				break
			}

			fmt.Println("error receiving data: %w", err)
			break
		}

		imageBytes, err := utils.ConvertImageFromBitmap(response, format)
		if err != nil {
			fmt.Println("Error converting image format: %w", err)
			continue
		}

		queue.Publish(context.Background(), topic, imageBytes)
	}
	time.Sleep(time.Second * 3)
	err = queue.CloseTopic(topic)
	if err != nil {
		fmt.Println("Error closing topic:", err)
		return err
	}

	return nil
}
