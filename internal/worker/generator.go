package worker

import (
	"context"
	"cozy-creator/gen-server/internal/config"
	"cozy-creator/gen-server/internal/services"
	"cozy-creator/gen-server/internal/utils"
	"cozy-creator/gen-server/pkg/mq"
	"cozy-creator/gen-server/tools"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"time"
)

func StartGeneration(ctx context.Context, queue mq.MessageQueue, mapChan *tools.MapChan[[]byte]) error {
	for {
		message, err := queue.Receive(ctx, "generation")
		if err != nil {
			if err.Error() == "no message available" {
				continue
			}

			log.Println("failed to receive message from queue:", err)
			break
		}

		var data map[string]any
		if err := json.Unmarshal(message, &data); err != nil {
			log.Println("failed to unmarshal message:", err)
			continue
		}

		err = generateImage(data, mapChan)
		queue.Ack(ctx, "generation", nil)
		if err != nil {
			log.Println("failed to generate image:", err)
			continue
		}
		fmt.Println("image generated successfully, request id:", data["request_id"])
	}

	return nil
}

func generateImage(data map[string]any, mapChan *tools.MapChan[[]byte]) error {
	cfg := config.GetConfig()
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

		mapChan.Send(requestId, imageBytes)
	}

	mapChan.Delete(requestId)

	return nil
}
