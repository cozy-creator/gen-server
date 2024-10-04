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
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/types"
	"go.uber.org/zap"

	"github.com/gin-gonic/gin"
)

func GenerateImageSync(c *gin.Context) {
	data := types.GenerateParams{}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)
	requestId, err := generation.NewRequest(data, app.MQ())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")
		for {
			url, err := generation.ReceiveImage(requestId, app.Uploader(), app.MQ())
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) {
					break
				}

				c.JSON(http.StatusInternalServerError, gin.H{"message": "Unable to complete image generation"})
				return false
			}

			if url != "" {
				c.Writer.Write([]byte(fmt.Sprintf(`{"status": "pending", "url": "%s"}\n`, url)))
			}
		}

		c.Writer.Write([]byte(`{"status": "finished"}\n`))
		c.Writer.Flush()
		return false
	})
}

func GenerateImageAsync(c *gin.Context) {
	app := c.MustGet("app").(*app.App)

	data := types.GenerateParams{}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	if data.WebhookUrl == "" {
		c.JSON(http.StatusBadRequest, gin.H{"message": "webhook_url is required"})
		return
	}

	requestId, err := generation.NewRequest(data, app.MQ())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	go generateImageAsync(app, &data, requestId)
	c.JSON(http.StatusOK, gin.H{"status": "pending", "id": requestId})
}

func generateImageAsync(app *app.App, data *types.GenerateParams, requestId string) {
	ctx, _ := context.WithTimeout(app.Context(), 5*time.Minute)
	// defer cancel()

	index := 0
	for {
		select {
		case <-ctx.Done():
			return
		default:
			url, err := generation.ReceiveImage(requestId, app.Uploader(), app.MQ())
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) {
					invokeWebhook(ctx, data.WebhookUrl, "success", "", "", requestId, 0)
					break
				}

				app.Logger.Error("Error receiving image from queue", zap.Error(err))
				return
			}

			if url != "" {
				invokeWebhook(ctx, data.WebhookUrl, "", url, "", requestId, index)
				index++
			}
		}
	}
}

func invokeWebhook(ctx context.Context, url, status, imageUrl, message, requestId string, index int) {
	httpClient := &http.Client{Timeout: time.Minute}
	respData := struct {
		ID      string `json:"id"`
		Index   int    `json:"index"`
		Status  string `json:"status,omitempty"`
		URL     string `json:"url,omitempty"`
		Message string `json:"message,omitempty"`
	}{
		ID:      requestId,
		Status:  status,
		URL:     imageUrl,
		Message: message,
		Index:   index,
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
