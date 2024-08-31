package api

import (
	"bytes"
	"cozy-creator/gen-server/internal/config"
	"cozy-creator/gen-server/internal/services"
	"cozy-creator/gen-server/internal/types"
	"cozy-creator/gen-server/internal/utils"
	"cozy-creator/gen-server/internal/worker"
	"cozy-creator/gen-server/tools"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image/jpeg"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"golang.org/x/image/bmp"
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

	if _, ok := data["format"]; !ok {
		data["format"] = "png"
	}

	requestId := uuid.NewString()
	data["request_id"] = requestId
	format := data["format"].(string)

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

func generateImage(data map[string]any) error {
	cfg := config.GetConfig()
	serverAddress := fmt.Sprintf("127.0.0.1:%d", cfg.TcpPort)
	timeout := time.Duration(cfg.TcpTimeout) * time.Second

	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal json data: %w", err)
	}

	client, err := services.NewTCPClient(serverAddress, timeout)
	if err != nil {
		return err
	}

	defer func() {
		if err := client.Close(); err != nil {
			fmt.Printf("Failed to close connection: %v\n", err)
		}
	}()

	client.Send(string(jsonData))
	requestId := data["request_id"].(string)
	format := data["format"].(string)

	for {
		sizeBytes, err := client.ReceiveFullBytes(4)
		// Receive the size of the incoming data (4 bytes for size)
		if err != nil {
			if err == io.EOF {
				fmt.Println("EOF reached")
				break // End of stream
			}
			fmt.Printf("Error reading size header: %v\n", err)
			break
		}

		contentsize := binary.BigEndian.Uint32(sizeBytes)
		if contentsize == 0 {
			fmt.Println("Received a chunk with size 0, skipping")
			continue
		}

		// Receive the actual data based on the size
		response, err := (client.ReceiveFullBytes(int(contentsize)))
		if err != nil {
			if errors.Is(err, io.ErrUnexpectedEOF) {
				fmt.Println("Unexpected EOF reached while reading data")
				break
			}
			if errors.Is(err, io.EOF) {
				fmt.Println("EOF reached while reading data")
				break
			}

			fmt.Println("error receiving data: %w", err)
			break
		}

		imageBytes, err := convertImageFormat(response, format)
		if err != nil {
			fmt.Println("Error converting image format: %w", err)
			continue
		}

		mapChan.Send(requestId, imageBytes)
	}

	mapChan.Delete(requestId)

	return nil
}

func convertImageFormat(bmpBytes []byte, format string) ([]byte, error) {
	img, err := bmp.Decode(bytes.NewReader(bmpBytes))
	if err != nil {
		return nil, err
	}

	var output bytes.Buffer
	switch format {
	case "png":
		err = png.Encode(&output, img)
	case "jpg":
	case "jpeg":
		options := &jpeg.Options{Quality: 90}
		err = jpeg.Encode(&output, img, options)
	default:
		return nil, err
	}

	if err != nil {
		return nil, err
	}

	return output.Bytes(), nil
}
