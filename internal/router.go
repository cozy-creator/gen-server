package internal

import (
	"net/http"

	"github.com/cozy-creator/gen-server/internal/api"
	"github.com/cozy-creator/gen-server/internal/types"

	"github.com/gin-gonic/gin"
)

func (s *Server) SetupRoutes() {
	// Not an API, just a simple file server
	s.engine.GET("/file/:filename", wrapper(api.GetFile))

	apiV1 := s.engine.Group("/api/v1")
	apiV1.POST("/upload", wrapper(api.UploadFile))
	apiV1.POST("/generate", api.GenerateImageSync)
	apiV1.POST("/generate_async", api.GenerateImageAsync)
}

func wrapper(f func(c *gin.Context) (*types.HandlerResponse, error)) gin.HandlerFunc {
	return func(c *gin.Context) {
		data, err := f(c)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		}

		if data.Type == types.FileResponseType {
			c.File(data.Data.(types.FileResponse).Path)
		} else if data.Type == types.JSONResponseType {
			c.JSON(http.StatusOK, data.Data)
		}
	}
}
