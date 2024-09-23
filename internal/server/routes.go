package server

import (
	"net/http"

	"github.com/cozy-creator/gen-server/internal/api"
	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/gin-gonic/gin"
)

func (s *Server) SetupRoutes(app *app.App) {
	// Health check endpoint
	s.ginEngine.GET("/healthz", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// Not an API, just a simple file server endpoint
	s.ginEngine.GET("/file/:filename", routeWrapper(app, api.GetFile))

	apiV1 := s.ginEngine.Group("/api/v1")
	apiV1.POST("/upload", routeWrapper(app, api.UploadFile))
	apiV1.POST("/generate", routeWrapper(app, api.GenerateImageSync))
	apiV1.POST("/generate_async", routeWrapper(app, api.GenerateImageAsync))
}

func routeWrapper(app *app.App, f func(c *gin.Context)) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Set("app", app)
		f(c)
	}
}
