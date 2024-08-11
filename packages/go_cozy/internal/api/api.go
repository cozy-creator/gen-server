package api

import (
	"cozy-creator/go-cozy/internal/services"
	"cozy-creator/go-cozy/internal/types"
	"cozy-creator/go-cozy/internal/utils"
	"encoding/hex"
	"fmt"
	"io"
	"mime/multipart"
	"path/filepath"

	"github.com/gin-gonic/gin"
)

func UploadFile(c *gin.Context) (*types.HandlerResponse, error) {
	file, err := c.FormFile("file")
	if err != nil {
		return types.NewErrorResponse("failed to get file: %w", err)
	}

	uploader, err := services.GetUploader()
	if err != nil {
		return types.NewErrorResponse("failed to create uploader: %w", err)
	}

	fileBytes, err := readFileContent(file)
	if err != nil {
		return types.NewErrorResponse("failed to read file: %w", err)
	}

	fileHash := utils.Blake3Hash(fileBytes)
	fileMeta := services.NewFileMeta(hex.EncodeToString(fileHash[:]), filepath.Ext(file.Filename), fileBytes, false)

	url, err := uploader.Upload(fileMeta)
	if err != nil {
		return types.NewErrorResponse("failed to upload file: %w", err)
	}

	return types.NewDataResponse(
		types.UploadResponse{
			Url: url,
		},
	)
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

func readFileContent(file *multipart.FileHeader) ([]byte, error) {
	content, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer content.Close()

	return io.ReadAll(content)
}
