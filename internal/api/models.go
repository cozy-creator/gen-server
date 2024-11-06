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

func LoadModels(c *gin.Context) {
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

func GetModelStatus(c *gin.Context) {
	app := c.MustGet("app").(*app.App)
	modelIDs := c.QueryArray("model_ids")

	statuses, err := models.GetModelStatus(app, modelIDs)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok", "loaded_models": statuses})
}

func UnloadModels(c *gin.Context) {
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

func WarmupModels(c *gin.Context) {
	var req ModelRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)
	if err := models.WarmupModels(app, req.ModelIDs); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}