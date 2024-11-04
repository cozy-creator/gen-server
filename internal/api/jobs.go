package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/google/uuid"

	"github.com/gin-gonic/gin"
	"github.com/gin-gonic/gin/binding"
)

type Job struct {
	ID          string                 `json:"id"`
	Status      string                 `json:"status"`
	Input       map[string]interface{} `json:"input"`
	CreatedAt   time.Time              `json:"created_at"`
	CompletedAt *time.Time             `json:"completed_at"`
}

type JobOutput struct {
	Urls []string `json:"urls"`
}

type JobWithOutput struct {
	ID          string                 `json:"id"`
	Status      string                 `json:"status"`
	Input       map[string]interface{} `json:"input"`
	Output      JobOutput              `json:"output"`
	CreatedAt   time.Time              `json:"created_at"`
	CompletedAt *time.Time             `json:"completed_at"`
}

func GenerateImageSync(c *gin.Context) {
	data := &types.GenerateParams{}
	if err := c.BindJSON(data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)
	if _, err := generation.NewRequest(data, app.MQ()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.Stream(func(w io.Writer) bool {
		c.Header("Content-Type", "application/json")
		output, err := generation.GenerateImageSync(app, data)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"message": "unable to complete image generation"})
			return false
		}

		for url := range output {
			urlc, _ := json.Marshal(url)
			c.Writer.Write(urlc)
		}

		c.Writer.Flush()
		return false
	})
}

func SubmitRequest(c *gin.Context) {
	var params = types.GenerateParams{}
	if err := c.ShouldBindWith(&params, binding.MsgPack); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	// if err := c.BindJSON(&params); err != nil {
	// 	c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
	// 	return
	// }

	fmt.Println("submit params: ", params)

	if params.WebhookUrl == "" {
		c.JSON(http.StatusBadRequest, gin.H{"message": "webhook_url is required"})
		return
	}

	app := c.MustGet("app").(*app.App)
	if _, err := generation.NewRequest(&params, app.MQ()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	input, err := json.Marshal(params)
	if err != nil {
		fmt.Println("Error marshaling params: ", err)
	}

	if params.ID == "" {
		params.ID = uuid.NewString()
	}

	createData := db.CreateJobParams{
		Input:  input,
		ID:     uuid.MustParse(params.ID),
		Status: db.JobStatusEnumINQUEUE,
	}

	if _, err := app.JobsRepo.Create(app.Context(), createData); err != nil {
		fmt.Println("Error creating job: ", err)
	}

	go generation.GenerateImageAsync(app, &params)
	c.JSON(http.StatusOK, types.GenerationResponse{
		Input:  &params,
		ID:     params.ID,
		Status: generation.StatusInQueue,
	})
}

func GetJob(c *gin.Context) {
	app := c.MustGet("app").(*app.App)

	id := uuid.MustParse(c.Param("id"))
	job, err := app.JobsRepo.GetWithImages(app.Context(), id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	var input map[string]interface{}
	if err := json.Unmarshal(job.Input, &input); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	var images []map[string]interface{}
	if err := json.Unmarshal(job.Images.([]byte), &images); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	var output JobOutput
	for _, image := range images {
		output.Urls = append(output.Urls, image["url"].(string))
	}

	jsonJob := JobWithOutput{
		Input:       input,
		Output:      output,
		ID:          job.ID.String(),
		Status:      string(job.Status),
		CreatedAt:   job.CreatedAt.Time,
		CompletedAt: &job.CompletedAt.Time,
	}

	c.JSON(http.StatusOK, gin.H{
		"data": jsonJob,
	})
}

func GetJobStatus(c *gin.Context) {
	id := c.Param("id")
	app := c.MustGet("app").(*app.App)

	job, err := app.JobsRepo.Get(app.Context(), id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": job.Status,
	})
}

func StreamJob(c *gin.Context) {
	app := c.MustGet("app").(*app.App)
	fmt.Println("Streaming job: ", c.Param("id"))

	id := uuid.MustParse(c.Param("id"))
	// job, err := app.JobsRepo.Get(app.Context(), id)
	// if err != nil {
	// 	c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
	// 	return
	// }

	// if job == nil {
	// 	c.JSON(http.StatusNotFound, gin.H{"message": "job not found"})
	// 	return
	// }

	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.WriteHeader(http.StatusOK)
	c.Writer.Flush()

	for {
		select {
		case <-c.Request.Context().Done():
			return
		default:
			topic := config.DefaultStreamsTopic + "/" + id.String()
			message, err := app.MQ().Receive(app.Context(), topic)
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) || errors.Is(err, mq.ErrQueueClosed) {
					break
				}

				continue
			}

			messageData, err := app.MQ().GetMessageData(message)
			if err != nil {
				continue
			}
			if bytes.Equal(messageData, []byte("END")) {
				if err := app.MQ().CloseTopic(topic); err != nil {
					fmt.Println("Error closing topic:", err)
				}

				if _, err := fmt.Fprintf(c.Writer, "data: {\"type\":\"message\", \"data\":\"%s\"}\n\n", "END"); err != nil {
					fmt.Println("Error writing to stream:", err)
				}
				break
			}

			if _, err = fmt.Fprintf(c.Writer, "data: %s\n\n", string(messageData)); err != nil {
				continue
			}
			c.Writer.Flush()
		}
	}
}

func SubmitAndStreamRequest(c *gin.Context) {
	var params = types.GenerateParams{}
	if err := c.ShouldBindWith(&params, binding.MsgPack); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "failed to parse request body"})
		return
	}

	app := c.MustGet("app").(*app.App)
	if _, err := generation.NewRequest(&params, app.MQ()); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		return
	}

	input, err := json.Marshal(params)
	if err != nil {
		fmt.Println("Error marshaling params: ", err)
	}

	if params.ID == "" {
		params.ID = uuid.NewString()
	}

	createData := db.CreateJobParams{
		Input:  input,
		ID:     uuid.MustParse(params.ID),
		Status: db.JobStatusEnumINQUEUE,
	}

	if _, err := app.JobsRepo.Create(app.Context(), createData); err != nil {
		fmt.Println("Error creating job: ", err)
	}

	go generation.GenerateImageAsync(app, &params)
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.WriteHeader(http.StatusOK)
	c.Writer.Flush()

	for {
		select {
		case <-c.Request.Context().Done():
			return
		default:
			topic := config.DefaultStreamsTopic + "/" + params.ID
			message, err := app.MQ().Receive(app.Context(), topic)
			if err != nil {
				if errors.Is(err, mq.ErrTopicClosed) || errors.Is(err, mq.ErrQueueClosed) {
					break
				}

				continue
			}

			messageData, err := app.MQ().GetMessageData(message)
			if err != nil {
				continue
			}
			if bytes.Equal(messageData, []byte("END")) {
				break
			}

			if _, err = fmt.Fprintf(c.Writer, "data: %s\n\n", string(messageData)); err != nil {
				continue
			}
			c.Writer.Flush()
		}
	}
}
