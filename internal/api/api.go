package api

import (
	"bytes"
	"cozy-creator/gen-server/internal/services"
	"cozy-creator/gen-server/internal/types"
	"cozy-creator/gen-server/internal/utils"
	"cozy-creator/gen-server/internal/worker"
	"cozy-creator/gen-server/tools"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

var mapChan = tools.DefaultBytesMap()

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

func GenerateImageSync(c *gin.Context) {
	data := make(map[string]any)
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
	}

	requestId := uuid.NewString()
	data["request_id"] = requestId

	mapChan.Set(requestId, make(chan []byte))
	go generateImage(data)

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")

		responseChan := mapChan.Get(requestId)

		for {
			image, ok := <-responseChan
			fmt.Println("responseChan->", "image received")
			if !ok {
				break
			}

			uploadUrl := make(chan string)
			go func() {
				fmt.Println("uploadUrl->", "uploading image")
				imageHash := utils.Blake3Hash(image)
				fileMeta := services.NewFileMeta(hex.EncodeToString(imageHash[:]), ".png", image, false)

				uploader := worker.GetUploadWorker()
				uploader.Upload(fileMeta, uploadUrl)
			}()

			url, ok := <-uploadUrl
			if !ok {
				break
			}

			fmt.Println("url", url)
			c.Writer.Write([]byte(fmt.Sprintf(`{"status": "pending", "url": "%s"}\n`, url)))
		}

		c.Writer.Write([]byte(`{"status": "finished"}\n`))
		c.Writer.Flush()
		return false
	})
}

func GenerateImageAsync(c *gin.Context) (*types.HandlerResponse, error) {
	data := map[string]any{}
	if err := c.BindJSON(&data); err != nil {
		return types.NewErrorResponse("failed to parse request body: %w", err)
	}

	go func() {
		imageChan := make(chan []byte)
		go generateImage(data)

		for {
			image, ok := <-imageChan
			fmt.Println("imageChan->", "image received")
			if !ok {
				break
			}

			uploadUrl := make(chan string)
			go func() {
				imageHash := utils.Blake3Hash(image)
				fileMeta := services.NewFileMeta(hex.EncodeToString(imageHash[:]), ".png", image, false)

				uploader := worker.GetUploadWorker()
				uploader.Upload(fileMeta, uploadUrl)
			}()

			url, ok := <-uploadUrl
			if !ok {
				break
			}

			response, err := http.Post(
				data["webhook_url"].(string),
				"application/json",
				bytes.NewBuffer([]byte(fmt.Sprintf(`{"status": "pending", "url": "%s"}`, url))),
			)

			if err != nil {
				fmt.Println("Error sending webhook response:", err)
				continue
			}

			defer response.Body.Close()
			fmt.Println("Webhook response:", response.Status)
		}
	}()

	return types.NewJSONResponse(map[string]interface{}{"status": "pending"})
}

func GenerationCallback(c *gin.Context) {
	fmt.Println("GenerationCallback")
	image, err := io.ReadAll(c.Request.Body)
	fmt.Println("image len", len(image))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to read request body"})
		return
	}

	requestId := c.Param("request_id")
	fmt.Println("requestId", requestId)
	if requestId == "" {
		c.JSON(http.StatusBadRequest, gin.H{"message": "request_id is required"})
		return
	}

	mapChan.Send(requestId, image)
	fmt.Println("mapChan->", "image sent")
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func readFileContent(file *multipart.FileHeader) ([]byte, error) {
	content, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer content.Close()

	return io.ReadAll(content)
}

func generateImage(data map[string]any) error {
	endpointUrl := ("http://127.0.0.1:8881/generate")

	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal json data: %w", err)
	}

	r := bytes.NewReader(jsonData)
	client, err := http.NewRequest(http.MethodPost, endpointUrl, r)
	if err != nil {
		return fmt.Errorf("failed to create http request: %w", err)
	}

	client.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(client)
	if err != nil {
		return fmt.Errorf("failed to make http request: %w", err)
	}

	defer response.Body.Close()
	if response.StatusCode != 200 {
		return fmt.Errorf("non-200 status code: %d", response.StatusCode)
	}

	return nil
}
