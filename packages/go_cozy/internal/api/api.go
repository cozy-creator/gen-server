package api

import (
	"bytes"
	"cozy-creator/go-cozy/internal/services"
	"cozy-creator/go-cozy/internal/types"
	"cozy-creator/go-cozy/internal/utils"
	"cozy-creator/go-cozy/internal/worker"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"

	"github.com/gin-gonic/gin"
)

func UploadFile(c *gin.Context) (*types.HandlerResponse, error) {
	file, err := c.FormFile("file")
	if err != nil {
		return types.NewErrorResponse("failed to get file: %w", err)
	}

	fileBytes, err := readFileContent(file)
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

func GenerateImageSync(c *gin.Context) (*types.HandlerResponse, error) {
	data := types.GenerateData{}
	if err := c.BindJSON(&data); err != nil {
		return types.NewErrorResponse("failed to parse request body: %w", err)
	}

	result, err := generateImage(data)
	if err != nil {
		return types.NewErrorResponse("failed to generate image: %w", err)
	}

	print(result)
	return types.NewJSONResponse(result)
}

func GenerateImageAsync(c *gin.Context) (*types.HandlerResponse, error) {
	data := types.GenerateData{}
	if err := c.BindJSON(&data); err != nil {
		return types.NewErrorResponse("failed to parse request body: %w", err)
	}

	go func() {
		result, err := generateImage(data)

		if err != nil {
			return
		}
		fmt.Println("Image generated successfully")
		fmt.Println(result)
	}()

	return types.NewJSONResponse(map[string]interface{}{"status": "pending"})
}

func readFileContent(file *multipart.FileHeader) ([]byte, error) {
	content, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer content.Close()

	return io.ReadAll(content)
}

func generateImage(data types.GenerateData) (interface{}, error) {
	endpointUrl := ("http://127.0.0.1:8881/generate_async")

	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal json data: %w", err)
	}

	r := bytes.NewReader(jsonData)
	client, err := http.NewRequest(http.MethodPost, endpointUrl, r)
	if err != nil {
		return types.NewErrorResponse("failed to create http request: %w", err)
	}

	client.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(client)
	if err != nil {
		return types.NewErrorResponse("failed to make http request: %w", err)
	}

	defer response.Body.Close()
	// if response.Header.Get("Content-Type") != "application/json" {
	// 	return nil, fmt.Errorf("unexpected content type: %s", response.Header.Get("Content-Type"))
	// }
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return types.NewErrorResponse("failed to read response body: %w", err)
	}

	return body, nil
}
