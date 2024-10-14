package api

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/workflow"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func ExecuteWorkflow(ctx *gin.Context) {
	var workflow workflow.Workflow
	if err := ctx.BindJSON(&workflow); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	if workflow.ID == "" {
		workflow.ID = uuid.NewString()
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

	ctx.JSON(http.StatusAccepted, gin.H{"status": "pending", "id": workflow.ID})
}

func StreamWorkflow(ctx *gin.Context) {
	id := ctx.Param("id")
	if id == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"message": "missing workflow id"})
		return
	}

	app := ctx.MustGet("app").(*app.App)

	ctx.Writer.Header().Set("Content-Type", "text/event-stream")
	ctx.Writer.Header().Set("Cache-Control", "no-cache")
	ctx.Writer.Header().Set("Connection", "keep-alive")
	ctx.Writer.WriteHeader(http.StatusOK)

	for {
		select {
		case <-ctx.Request.Context().Done():
			return
		default:
			topic := "workflows:" + id
			message, err := app.MQ().Receive(app.Context(), topic)
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) || errors.Is(err, mq.ErrQueueClosed) {
					break
				}

				continue
			}

			if _, err := ctx.Writer.Write(message); err != nil {
				continue
			}
			ctx.Writer.Flush()
		}
	}
}
