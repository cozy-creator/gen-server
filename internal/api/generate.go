package api

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/types"

	"github.com/gin-gonic/gin"
)

func GenerateImageSync(c *gin.Context) {
	data := &types.GenerateParams{}
	if err := c.BindJSON(data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)
	if _, err := generation.NewRequest(data, app.MQ()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")
		output, err := generation.GenerateImageSync(app, data)
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

func GenerateImageAsync(c *gin.Context) {
	data := &types.GenerateParams{}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	if data.WebhookUrl == "" {
		c.JSON(http.StatusBadRequest, gin.H{"message": "webhook_url is required"})
		return
	}

	app := c.MustGet("app").(*app.App)
	if _, err := generation.NewRequest(data, app.MQ()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	go generation.GenerateImageAsync(app, data)
	c.JSON(http.StatusOK, types.GenerationResponse{
		Input:  data,
		ID:     data.ID,
		Status: generation.StatusInQueue,
	})
}
