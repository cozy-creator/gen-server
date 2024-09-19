package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"time"

	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/filehandler"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/internal/utils/hashutil"
	"github.com/cozy-creator/gen-server/internal/worker"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func UploadFile(c *gin.Context) (*types.HandlerResponse, error) {
	file, err := c.FormFile("file")
	if err != nil {
		return types.NewErrorResponse("failed to get file: %w", err)
	}

	content, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer content.Close()

	fileBytes, err := io.ReadAll(content)
	if err != nil {
		return types.NewErrorResponse("failed to read file: %w", err)
	}

	fileHash := hashutil.Blake3Hash(fileBytes)
	fileInfo := filehandler.FileInfo{
		Name:      fileHash,
		Extension: filepath.Ext(file.Filename),
		Content:   fileBytes,
		IsTemp:    false,
	}

	response := make(chan string)
	worker := worker.GetUploadWorker()
	worker.Upload(fileInfo, response)

	return types.NewJSONResponse(types.UploadResponse{Url: <-response})
}

func GetFile(c *gin.Context) (*types.HandlerResponse, error) {
	filename := c.Param("filename")
	handler, err := filehandler.GetFileHandler()
	if err != nil {
		return types.NewErrorResponse("failed to get handler: %w", err)
	}

	file, err := handler.ResolveFile(filename, "", false)
	if err != nil {
		return types.NewErrorResponse("failed to get file: %w", err)
	}

	return types.NewFileResponse(file)
}

func GenerateImageSync(c *gin.Context) {
	requestId := uuid.NewString()
	data := types.GenerateParams{OutputFormat: "png"}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	requestId, err := worker.RequestGenerateImage(data)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")
		for {
			imageUrl, err := worker.ReceiveGenerateImage(requestId, data.OutputFormat)
			// fmt.Println("Received image from queue:", imageUrl)
			if err != nil {
				if errors.Is(err, mq.ErrNoMessage) {
					continue
				}

				fmt.Println("Error receiving image from queue:", err)

				c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
				return false
			}

			if imageUrl == "" {
				break
			}

			c.Writer.Write([]byte(fmt.Sprintf(`{"status": "pending", "url": "%s"}\n`, imageUrl)))
		}

		c.Writer.Write([]byte(`{"status": "finished"}\n`))
		c.Writer.Flush()
		return false
	})
}

func GenerateImageAsync(c *gin.Context) {
	// ctx, _ := context.WithTimeout(c.Request.Context(), 5*time.Minute)

	data := types.GenerateParams{OutputFormat: "png"}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	if data.WebhookUrl == "" {
		c.JSON(http.StatusBadRequest, gin.H{"message": "webhook_url is required"})
		return
	}

	requestId, err := worker.RequestGenerateImage(data)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	ctx := context.Background()
	go func() {
		// defer cancel()

		// defer invokeWebhook(ctx, data.WebhookUrl, "error", "", "Generation process ended unexpectedly")

		for {
			select {
			case <-ctx.Done():
				invokeWebhook(ctx, data.WebhookUrl, "timeout", "", "Image generation timed out")
				return
			default:
				imageUrl, err := worker.ReceiveGenerateImage(requestId, data.OutputFormat)
				fmt.Println("Received image from queue:", imageUrl)
				if err != nil {
					if errors.Is(err, mq.ErrNoMessage) {
						time.Sleep(time.Second) // Avoid tight loop
						continue
					}
					invokeWebhook(ctx, data.WebhookUrl, "error", "", err.Error())
					return
				}
				invokeWebhook(ctx, data.WebhookUrl, "success", imageUrl, "")
				return
			}
		}
	}()

	c.JSON(http.StatusOK, gin.H{"status": "pending", "request_id": requestId})
}

func invokeWebhook(ctx context.Context, url, status, imageUrl, message string) {
	httpClient := &http.Client{Timeout: 10 * time.Second}
	respData := struct {
		Status   string `json:"status"`
		ImageURL string `json:"image_url,omitempty"`
		Message  string `json:"message,omitempty"`
	}{
		Status:   status,
		ImageURL: imageUrl,
		Message:  message,
	}

	jsonData, err := json.Marshal(respData)
	if err != nil {
		log.Printf("Failed to marshal webhook data: %v", err)
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Failed to create webhook request: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("Failed to invoke webhook: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Webhook returned non-200 status: %d", resp.StatusCode)
	}
}
