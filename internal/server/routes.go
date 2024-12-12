package server

import (
	"fmt"
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

	apiV1 := s.ginEngine.Group("/v1")

	// Authentication middleware. Auth is required in prod.
	if !app.Config().DisableAuth || app.Config().Environment == "prod" {
		apiV1.Use(handlerWrapper(app, middleware.AuthenticationMiddleware))
	} else {
		fmt.Println("Warning: authentication is disabled for all requests")
	}

	apiV1.POST("/upload", handlerWrapper(app, api.UploadFileHandler))

	apiV1.GET("/jobs/:id", handlerWrapper(app, api.GetJobHandler))
	apiV1.GET("/jobs/:id/stream", handlerWrapper(app, api.StreamJobHandler))
	apiV1.POST("/jobs/submit", handlerWrapper(app, api.SubmitRequestHandler))
	apiV1.POST("/jobs/stream", handlerWrapper(app, api.SubmitAndStreamRequestHandler))

	apiV1.POST("/workflow/execute", handlerWrapper(app, api.ExecuteWorkflowHandler))
	apiV1.GET("/workflow/:id/stream", handlerWrapper(app, api.StreamWorkflowHandler))

	// Admin-only API endpoints
	// TO DO: these should not be exposed to the public API--currently they are
	// TO DO: none of these endpoints are well thought out
	apiV1.GET("/models/status", handlerWrapper(app, api.GetModelStatusHandler))
	apiV1.POST("/models/load", handlerWrapper(app, api.LoadModelsHandler))
	apiV1.POST("/models/unload", handlerWrapper(app, api.UnloadModelsHandler))
	apiV1.POST("/models/enable", handlerWrapper(app, api.EnableModelsHandler))
}

func handlerWrapper(app *app.App, f func(c *gin.Context)) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.Set("app", app)
		f(ctx)
	}
}
