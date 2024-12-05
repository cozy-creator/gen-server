package api

import (
	"net/http"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/services/models"
	"github.com/gin-gonic/gin"
)

type ModelRequest struct {
	ModelIDs []string `json:"model_ids"`
	Priority bool     `json:"priority"`
}

func LoadModelsHandler(c *gin.Context) {
	var req ModelRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)

	if err := models.LoadModels(app, req.ModelIDs, req.Priority); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func GetModelStatusHandler(c *gin.Context) {
	app := c.MustGet("app").(*app.App)
	modelIDs := c.QueryArray("model_ids")

	statuses, err := models.GetModelStatus(app, modelIDs)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok", "loaded_models": statuses})
}

func UnloadModelsHandler(c *gin.Context) {
	var req ModelRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)

	if err := models.UnloadModels(app, req.ModelIDs); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// This is supposed to add models to the list of enabled models
func EnableModelsHandler(c *gin.Context) {
	var req ModelRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)
	if err := models.EnableModels(app, req.ModelIDs); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}
