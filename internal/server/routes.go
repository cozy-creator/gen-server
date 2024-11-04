package server

import (
	"net/http"

	"github.com/cozy-creator/gen-server/internal/api"
	"github.com/cozy-creator/gen-server/internal/api/middleware"
	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/gin-gonic/gin"
)

func (s *Server) SetupRoutes(app *app.App) {
	// Health check endpoint
	s.ginEngine.GET("/healthz", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// Not an API, just a simple file server endpoint
	s.ginEngine.GET("/file/:filename", handlerWrapper(app, api.GetFile))

	apiV1 := s.ginEngine.Group("/api/v1")

	// Authentication middleware
	apiV1.Use(handlerWrapper(app, middleware.AuthenticationMiddleware))

	apiV1.POST("/upload", handlerWrapper(app, api.UploadFile))
	apiV1.POST("/generate", handlerWrapper(app, api.GenerateImageSync))
	apiV1.POST("/generate_async", handlerWrapper(app, api.GenerateImageAsync))

	apiV1.POST("/workflow/execute", handlerWrapper(app, api.ExecuteWorkflow))
	apiV1.GET("/workflow/:id/stream", handlerWrapper(app, api.StreamWorkflow))

	apiV1.POST("/models/load", handlerWrapper(app, api.LoadModels))
	apiV1.GET("/models/status", handlerWrapper(app, api.GetModelStatus))
	apiV1.POST("/models/unload", handlerWrapper(app, api.UnloadModels))
	apiV1.POST("/models/warmup", handlerWrapper(app, api.WarmupModels))

	// Replicate
	apiV1.POST("/generate/replicate", handlerWrapper(app, api.GenerateReplicateImageSync))
	apiV1.POST("/generate_async/replicate", handlerWrapper(app, api.GenerateReplicateImageAsync))
}

func handlerWrapper(app *app.App, f func(c *gin.Context)) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.Set("app", app)
		f(ctx)
	}
}
