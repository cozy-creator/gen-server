package api

import (
	"io"
	"net/http"
	"path/filepath"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/gabriel-vasile/mimetype"
	"github.com/gin-gonic/gin"
)

func UploadFileHandler(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	content, err := file.Open()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to open file"})
		return
	}
	defer content.Close()

	fileBytes, err := io.ReadAll(content)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to read file"})
		return
	}

	url := make(chan string)
	app := c.MustGet("app").(*app.App)
	app.Uploader().UploadBytes(fileBytes, filepath.Ext(file.Filename), false, url)

	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"data": map[string]string{
			"url": <-url,
		},
	})
}

func GetFile(c *gin.Context) {
	filename := c.Param("filename")
	app := c.MustGet("app").(*app.App)

	storage, err := filestorage.NewFileStorage(app.Config())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	if app.Config().FilesystemType == config.FilesystemLocal {
		file, err := storage.ResolveFile(filename, "", false)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"message": "file not found"})
			return
		}

		c.File(file)
		return
	} else {
		file, err := storage.GetFile(filename)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"message": "file not found"})
			return
		}

		content := file.Content.([]byte)
		mimeType := mimetype.Detect(content).String()
		c.Data(http.StatusOK, mimeType, content)
	}
}
