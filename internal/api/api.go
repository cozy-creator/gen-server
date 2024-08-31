package api

import (
	"context"
	"cozy-creator/gen-server/internal/services"
	"cozy-creator/gen-server/internal/types"
	"cozy-creator/gen-server/internal/utils"
	"cozy-creator/gen-server/internal/worker"
	"cozy-creator/gen-server/pkg/mq"
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

var mapChan = tools.GetDefaultBytesMap()
var inMemoryQueue = mq.GetDefaultInMemoryQueue()

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
		return
	}

	if _, ok := data["format"]; !ok {
		data["format"] = "png"
	}

	requestId := uuid.NewString()
	data["request_id"] = requestId
	format := data["format"].(string)

	mapChan.Set(requestId, make(chan []byte))

	bytesData, err := json.Marshal(data)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to serialize request data"})
		return
	}

	err = inMemoryQueue.Publish(context.TODO(), "generation", bytesData)
	fmt.Println("message published to queue:", err)
	if err != nil {
		fmt.Println("failed to publish message to queue:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "failed to publish message to queue"})
		return
	}

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
				fileMeta := services.NewFileMeta(hex.EncodeToString(imageHash[:]), fmt.Sprintf(".%s", format), image, false)

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

func CompleteGeneration(c *gin.Context) {
	fmt.Println("CompleteGeneration")
	requestId := c.Param("request_id")
	if requestId == "" {
		c.JSON(http.StatusBadRequest, gin.H{"message": "request_id is required"})
		return
	}

	mapChan.Delete(requestId)
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

// func GenerateImageAsync(c *gin.Context) (*types.HandlerResponse, error) {
// 	data := map[string]any{}
// 	if err := c.BindJSON(&data); err != nil {
// 		return types.NewErrorResponse("failed to parse request body: %w", err)
// 	}

// 	go func() {
// 		imageChan := make(chan []byte)
// 		go generateImage(data)

// 		for {
// 			image, ok := <-imageChan
// 			fmt.Println("imageChan->", "image received")
// 			if !ok {
// 				break
// 			}

// 			uploadUrl := make(chan string)
// 			go func() {
// 				imageHash := utils.Blake3Hash(image)
// 				fileMeta := services.NewFileMeta(hex.EncodeToString(imageHash[:]), ".png", image, false)

// 				uploader := worker.GetUploadWorker()
// 				uploader.Upload(fileMeta, uploadUrl)
// 			}()

// 			url, ok := <-uploadUrl
// 			if !ok {
// 				break
// 			}

// 			response, err := http.Post(
// 				data["webhook_url"].(string),
// 				"application/json",
// 				bytes.NewBuffer([]byte(fmt.Sprintf(`{"status": "pending", "url": "%s"}`, url))),
// 			)

// 			if err != nil {
// 				fmt.Println("Error sending webhook response:", err)
// 				continue
// 			}

// 			defer response.Body.Close()
// 			fmt.Println("Webhook response:", response.Status)
// 		}
// 	}()

// 	return types.NewJSONResponse(map[string]interface{}{"status": "pending"})
// }
