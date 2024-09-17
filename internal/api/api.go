package api

import (
	"cozy-creator/gen-server/internal/services/filehandler"
	"cozy-creator/gen-server/internal/types"
	"cozy-creator/gen-server/internal/utils/hashutil"
	"cozy-creator/gen-server/internal/worker"
	"fmt"
	"io"
	"net/http"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

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

	fileHash := hashutil.Blake3Hash(fileBytes)
	fileInfo := filehandler.FileInfo{
		Name:      fileHash,
		Extension: filepath.Ext(file.Filename),
		Content:   fileBytes,
		IsTemp:    false,
	}

	response := make(chan string)
	worker := worker.GetUploadWorker()
	worker.Upload(fileInfo, response)

	return types.NewJSONResponse(types.UploadResponse{Url: <-response})
}

func GetFile(c *gin.Context) (*types.HandlerResponse, error) {
	filename := c.Param("filename")
	handler, err := filehandler.GetFileHandler()
	if err != nil {
		return types.NewErrorResponse("failed to get handler: %w", err)
	}

	file, err := handler.ResolveFile(filename, "", false)
	if err != nil {
		return types.NewErrorResponse("failed to get file: %w", err)
	}

	return types.NewFileResponse(file)
}

func GenerateImageSync(c *gin.Context) {
	requestId := uuid.NewString()
	data := types.GenerateParams{OutputFormat: "png"}
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
