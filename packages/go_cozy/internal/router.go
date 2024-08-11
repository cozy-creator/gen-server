package internal

import (
	"cozy-creator/go-cozy/internal/api"
	"cozy-creator/go-cozy/internal/types"
	"net/http"

	"github.com/gin-gonic/gin"
)

func (s *HTTPServer) SetupRouter() {
	// Not an API, just a simple file server
	s.engine.GET("/file/:filename", wrapper(api.GetFile))

	apiV1 := s.engine.Group("/api/v1")
	apiV1.POST("/upload", wrapper(api.UploadFile))
}

func wrapper(f func(c *gin.Context) (*types.HandlerResponse, error)) gin.HandlerFunc {
	return func(c *gin.Context) {
		data, err := f(c)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		}

		if data.Type == types.FileResponseType {
			c.File(data.Data.(types.FileResponse).Path)
		} else if data.Type == types.DataResponseType {
			c.JSON(http.StatusOK, data.Data)
		}
	}
}
