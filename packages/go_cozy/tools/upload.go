package tools

import (
	"cozy-creator/go-cozy/internal/services"
	"cozy-creator/go-cozy/internal/worker"
	"fmt"
)

func handleResponse(response chan string) {
	url := <-response

	if url == "" {
		fmt.Println("Failed to upload file")
	} else {
		fmt.Println("File uploaded successfully:", url)
		// send back url to the gpu worker
	}
}

var Requests = make(chan services.FileMeta)

func StartUploadWorker() {
	worker := worker.GetUploadWorker()

	for {
		select {
		case request, ok := <-Requests:
			if !ok {
				break
			}
			response := make(chan string)
			worker.Upload(request, response)
			handleResponse(response)
		}
	}

}
