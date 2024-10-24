package api

import (
	"encoding/json"
	"io"
	"net/http"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/model"


	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/types"

	"github.com/gin-gonic/gin"
)

func GenerateImageSync(c *gin.Context) {
	data := types.GenerateParams{}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)

	// First check and load model if needed
    for modelID := range data.Models {
        location, err := app.ModelManager().CheckModel(c.Request.Context(), modelID)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"message": fmt.Sprintf("failed to check model status: %v", err)})
            return
        }

        if location == model.None {
            if err := app.ModelManager().LoadModel(c.Request.Context(), modelID); err != nil {
                c.JSON(http.StatusInternalServerError, gin.H{"message": fmt.Sprintf("failed to load model: %v", err)})
                return
            }
        }
    }

	// Continue with your existing generation code
	_, err := generation.NewRequest(&data, true, app.MQ())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")
		output, err := generation.GenerateImageSync(app.Context(), &data, app.Uploader(), app.MQ())
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"message": "Unable to complete image generation"})
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

	requestId, err := generation.NewRequest(&data, true, app.MQ())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	go generation.GenerateImageAsync(app.Context(), &data, app.Uploader(), app.MQ())
	c.JSON(http.StatusOK, gin.H{"status": generation.StatusInQueue, "id": requestId})
}
