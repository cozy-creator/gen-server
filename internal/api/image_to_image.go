package api

import (
	"io"
	"encoding/json"
	"net/http"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/gin-gonic/gin"
)

type ImageToImageParams struct {
	SourceImage    interface{}      `json:"source_image"`
	Models         map[string]int    `json:"models"`
	Prompt         string            `json:"prompt"`
	NegativePrompt string            `json:"negative_prompt,omitempty"`
	Strength       float32           `json:"strength"`
	RandomSeed     int               `json:"random_seed"`
	GuidanceScale  float32           `json:"guidance_scale,omitempty"`
	WebhookUrl     string            `json:"webhook_url,omitempty"`
}

func GenerateImageToImageSync(c *gin.Context) {
	data := &ImageToImageParams{}
	if err := c.BindJSON(data); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body", "error": err.Error()})
        return
    }

    // Validate source_image
    switch v := data.SourceImage.(type) {
    case string:
        // Single image path is fine
    case []interface{}:
        // Multiple images
        for _, img := range v {
            if _, ok := img.(string); !ok {
                c.JSON(http.StatusBadRequest, gin.H{"message": "source_image must be a string or array of strings"})
                return
            }
        }
    default:
        c.JSON(http.StatusBadRequest, gin.H{"message": "source_image must be a string or array of strings"})
        return
    }

    app := c.MustGet("app").(*app.App)

	// Create generation request
	genParams := &types.GenerateParams{
		ID:             "",  // Will be generated
		Models:         data.Models,
		PositivePrompt: data.Prompt,
		NegativePrompt: data.NegativePrompt,
		RandomSeed:     data.RandomSeed,
		OutputFormat:   "png",
		SourceImage:    data.SourceImage,
        Strength:       data.Strength,
	}

	if _, err := generation.NewRequest(genParams, app.MQ()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")
		output, err := generation.GenerateImageToImageSync(app, genParams, data.SourceImage, data.Strength)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"message": "unable to complete image generation"})
			return false
		}

		for url := range output {
			urlc, _ := json.Marshal(url)
			c.Writer.Write(urlc)
		}

		c.Writer.Flush()
		return false
	})
}

func GenerateImageToImageAsync(c *gin.Context) {
	data := &ImageToImageParams{}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	if data.WebhookUrl == "" {
		c.JSON(http.StatusBadRequest, gin.H{"message": "webhook_url is required"})
		return
	}

	app := c.MustGet("app").(*app.App)

	genParams := &types.GenerateParams{
		ID:             "", 
		Models:         data.Models,
		PositivePrompt: data.Prompt,
		NegativePrompt: data.NegativePrompt,
		RandomSeed:     data.RandomSeed,
		WebhookUrl:     data.WebhookUrl,
		OutputFormat:   "png",
	}

	if _, err := generation.NewRequest(genParams, app.MQ()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	go generation.GenerateImageToImageAsync(app, genParams, data.SourceImage, data.Strength)
	
	c.JSON(http.StatusOK, types.GenerationResponse{
		Input:  genParams,
		ID:     genParams.ID,
		Status: generation.StatusInQueue,
	})
}