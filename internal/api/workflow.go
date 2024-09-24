package api

import (
	"encoding/json"
	"net/http"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/services/workflow"
	"github.com/gin-gonic/gin"
)

func ExecuteWorkflow(c *gin.Context) {
	var workflow workflow.Workflow
	if err := c.BindJSON(&workflow); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	data, err := json.Marshal(workflow)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "failed to marshal workflow data"})
		return
	}

	app := c.MustGet("app").(*app.App)
	if err := app.MQ().Publish(c.Request.Context(), "workflows", data); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "failed to publish message to queue"})
		return
	}

	c.JSON(http.StatusAccepted, gin.H{"status": "pending"})
}
