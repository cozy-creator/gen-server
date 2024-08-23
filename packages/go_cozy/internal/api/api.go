package api

import (
	"bytes"
	"cozy-creator/go-cozy/internal/services"
	"cozy-creator/go-cozy/internal/types"
	"cozy-creator/go-cozy/internal/utils"
	"cozy-creator/go-cozy/internal/worker"
	"encoding/binary"
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

func GenerateImageSync(c *gin.Context) {
	data := types.GenerateData{}
	if err := c.BindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
	}

	imageChan := make(chan []byte)
	go generateImage(data, imageChan)

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")

	Loop:
		for {
			image, ok := <-imageChan
			if !ok {
				break Loop
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
				break Loop
			}

			c.Writer.Write([]byte(fmt.Sprintf(`{"status": "pending", "url": "%s"}\n`, url)))
		}

		c.Writer.Write([]byte(`{"status": "finished"}\n`))
		c.Writer.Flush()
		return false
	})
}

func GenerateImageAsync(c *gin.Context) (*types.HandlerResponse, error) {
	data := types.GenerateData{}
	if err := c.BindJSON(&data); err != nil {
		return types.NewErrorResponse("failed to parse request body: %w", err)
	}

	go func() {
		imageChan := make(chan []byte)
		go generateImage(data, imageChan)

		for {
			image, ok := <-imageChan
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
				*data.WebhookUrl,
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

func readFileContent(file *multipart.FileHeader) ([]byte, error) {
	content, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer content.Close()

	return io.ReadAll(content)
}

func generateImage(data types.GenerateData, responseChan chan []byte) error {
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
	defer close(responseChan)

	lengthBuf := make([]byte, 4)
	for {
		_, err := io.ReadFull(response.Body, lengthBuf)
		if err != nil {
			if err == io.EOF {
				fmt.Println("EOF reached")
				break // End of stream
			}
			fmt.Printf("Error reading length prefix: %v\n", err)
			break
		}

		contentLength := binary.BigEndian.Uint32(lengthBuf)
		if contentLength == 0 {
			fmt.Println("Received a chunk with length 0, skipping")
			continue
		}

		contentBytes := make([]byte, contentLength)
		_, err = io.ReadFull(response.Body, contentBytes)
		if err != nil {
			if err == io.ErrUnexpectedEOF {
				fmt.Printf("Error: expected %d bytes but got fewer\n", contentLength)
			}

			if err == io.EOF {
				fmt.Println("EOF reached while reading chunk", err)
				break // End of stream
			}

			fmt.Printf("Error reading chunk data: %v\n", err)
			break
		}

		responseChan <- contentBytes
	}

	return nil
}
