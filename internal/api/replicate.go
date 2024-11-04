package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/gen-server/scripts"
	"github.com/gin-gonic/gin"
)

func GenerateReplicateImageSync(c *gin.Context) {
	data := &types.GenerateParams{}
	if err := c.BindJSON(data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)
	replicate := scripts.NewReplicateAI(app.Config().Replicate.APIKey)

	// Get number of images from the first model
	var numImages int
	for _, num := range data.Models {
		numImages = num
		break
	}

	// Create generation
	gen, err := replicate.CreateRecraft(data.PositivePrompt, data.NegativePrompt, numImages)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")
		finalGen, err := replicate.PollGeneration(gen.URLs.Get)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"message": "unable to complete image generation"})
			return false
		}

		var urls []string
		switch output := finalGen.Output.(type) {
		case []interface{}:
			for _, url := range output {
				if strURL, ok := url.(string); ok {
					urls = append(urls, strURL)
				}
			}
		case string:
			urls = append(urls, output)
		}

		// Match your existing response format
		output := types.GenerationResponse{
			ID:     gen.ID,
			Status: "COMPLETED",
			Output: types.GeneratedOutput{
				URLs:  urls,
				Model: "replicate-recraft",
			},
			Input: data,
		}

		outputc, _ := json.Marshal(output)
		c.Writer.Write(outputc)
		c.Writer.Flush()
		return false
	})
}

func GenerateReplicateImageAsync(c *gin.Context) {
	data := &types.GenerateParams{}
	if err := c.BindJSON(data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	if data.WebhookUrl == "" {
		c.JSON(http.StatusBadRequest, gin.H{"message": "webhook_url is required"})
		return
	}

	app := c.MustGet("app").(*app.App)
	replicate := scripts.NewReplicateAI(app.Config().Replicate.APIKey)

	// Get number of images from the first model
	var numImages int
	for _, num := range data.Models {
		numImages = num
		break
	}

	// Create generation
	gen, err := replicate.CreateRecraft(data.PositivePrompt, data.NegativePrompt, numImages)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	// Immediately return the generation ID
	response := types.GenerationResponse{
		ID:     gen.ID,
		Status: "IN_QUEUE",
		Input:  data,
	}

	c.JSON(http.StatusOK, response)

	// Process async
	go func() {
		finalGen, err := replicate.PollGeneration(gen.URLs.Get)
		if err != nil {
			// Send error webhook
			webhookData := types.GenerationResponse{
				ID:     gen.ID,
				Status: "FAILED",
				Input:  data,
			}
			sendWebhookNotification(data.WebhookUrl, webhookData)
			return
		}

		var urls []string
		switch output := finalGen.Output.(type) {
		case []interface{}:
			for _, url := range output {
				if strURL, ok := url.(string); ok {
					urls = append(urls, strURL)
				}
			}
		case string:
			urls = append(urls, output)
		}

		// Send success webhook
		webhookData := types.GenerationResponse{
			ID:     gen.ID,
			Status: "COMPLETED",
			Output: types.GeneratedOutput{
				URLs:  urls,
				Model: "replicate-recraft",
			},
			Input: data,
		}
		sendWebhookNotification(data.WebhookUrl, webhookData)
	}()
}

func sendWebhookNotification(webhookUrl string, data types.GenerationResponse) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return
	}

	resp, err := http.Post(webhookUrl, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return
	}
	defer resp.Body.Close()
}