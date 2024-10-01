package api

import (
	"encoding/json"
	"net/http"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/services/workflow"
	"github.com/gin-gonic/gin"
)

func ExecuteWorkflow(ctx *gin.Context) {
	var workflow workflow.Workflow
	if err := ctx.BindJSON(&workflow); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	data, err := json.Marshal(workflow)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"message": "failed to marshal workflow data"})
		return
	}

	app := ctx.MustGet("app").(*app.App)
	if err := app.MQ().Publish(app.Context(), "workflows", data); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"message": "failed to publish message to queue"})
		return
	}

	ctx.JSON(http.StatusAccepted, gin.H{"status": "pending"})
}
