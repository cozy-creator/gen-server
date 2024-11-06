package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"go.uber.org/zap"

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
    
    if app.Config().Replicate.APIKey == "" {
        c.JSON(http.StatusInternalServerError, gin.H{"message": "Replicate API key not configured"})
        return
    }

    // Default values for style and size
    style := "any"
    size := "1024x1024"
    
    if data.Style != "" {
        style = data.Style
    }
    if data.Size != "" {
        size = data.Size
    }

    app.Logger.Info("Starting Replicate generation",
        zap.String("prompt", data.PositivePrompt),
        zap.String("style", style),
        zap.String("size", size))
    
    replicate := scripts.NewReplicateAI(app.Config().Replicate.APIKey)

    gen, err := replicate.CreateRecraft(data.PositivePrompt, style, size)
    if err != nil {
        app.Logger.Error("Failed to create Replicate generation", zap.Error(err))
        c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
        return
    }

    c.Stream(func(w io.Writer) bool {
        c.Header("Content-Type", "application/json")
        finalGen, err := replicate.PollGeneration(gen.URLs.Get)
        if err != nil {
            app.Logger.Error("Generation failed",
                zap.String("generation_id", gen.ID),
                zap.Error(err))
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

        app.Logger.Info("Generation completed successfully",
            zap.String("generation_id", gen.ID),
            zap.Int("num_urls", len(urls)))

        // Match existing response format
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
	if app.Config().Replicate.APIKey == "" {
        c.JSON(http.StatusInternalServerError, gin.H{"message": "Replicate API key not configured"})
		return
	}

	// Default values for style and size
    style := "any"
    size := "1024x1024"

	if data.Style != "" {
        style = data.Style
    }
    if data.Size != "" {
        size = data.Size
    }
	replicate := scripts.NewReplicateAI(app.Config().Replicate.APIKey)



	// Create generation
	gen, err := replicate.CreateRecraft(data.PositivePrompt, style, size)
    if err != nil {
        app.Logger.Error("Failed to create Replicate generation", zap.Error(err))
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