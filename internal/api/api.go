package api

import (
	"cozy-creator/gen-server/internal/services"
	"cozy-creator/gen-server/internal/types"
	"cozy-creator/gen-server/internal/utils"
	"cozy-creator/gen-server/internal/worker"
	"cozy-creator/gen-server/pkg/mq"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

var inMemoryQueue = mq.GetDefaultInMemoryQueue()

func UploadFile(c *gin.Context) (*types.HandlerResponse, error) {
	file, err := c.FormFile("file")
	if err != nil {
		return types.NewErrorResponse("failed to get file: %w", err)
	}

	content, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer content.Close()

	fileBytes, err := io.ReadAll(content)
	if err != nil {
		return types.NewErrorResponse("failed to read file: %w", err)
	}

	fileHash := utils.Blake3Hash(fileBytes)
	fileMeta := services.NewFileMeta(hex.EncodeToString(fileHash[:]), filepath.Ext(file.Filename), fileBytes, false)

	response := make(chan string)
	uploader := worker.GetUploadWorker()
	uploader.Upload(fileMeta, response)

	return types.NewJSONResponse(types.UploadResponse{Url: <-response})
}

func GetFile(c *gin.Context) (*types.HandlerResponse, error) {
	filename := c.Param("filename")
	uploader, err := services.GetUploader()
	if err != nil {
		return types.NewErrorResponse("failed to get uploader: %w", err)
	}

	file, err := uploader.ResolveFile(filename, "", false)
	if err != nil {
		return types.NewErrorResponse("failed to get file: %w", err)
	}

	return types.NewFileResponse(file)
}

func GenerateImageSync(c *gin.Context) {
	requestId := uuid.NewString()
	data := types.GenerateData{OutputFormat: "png"}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	requestId, err := worker.RequestGenerateImage(data)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")
		for {
			imageUrl, err := worker.ReceiveGenerateImage(requestId, data.OutputFormat)
			if err != nil {
				if err.Error() == "no image" {
					continue
				}

				c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
				return false
			}

			if imageUrl == "" {
				break
			}

			c.Writer.Write([]byte(fmt.Sprintf(`{"status": "pending", "url": "%s"}\n`, imageUrl)))
		}

		c.Writer.Write([]byte(`{"status": "finished"}\n`))
		c.Writer.Flush()
		return false
	})
}
