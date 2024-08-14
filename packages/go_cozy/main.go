package main

import (
	"cozy-creator/go-cozy/cmd"
	"cozy-creator/go-cozy/internal/services"
	"cozy-creator/go-cozy/internal/worker"
	"fmt"
)

func main() {
	cli := cmd.NewCLI()

	if err := cli.Run(); err != nil {
		fmt.Println(err)
		return
	}

	uploader, err := services.GetUploader()
	if err != nil {
		fmt.Println(err)
		return
	}

	uploadWorker := worker.NewUploadWorker(uploader, 10)
	worker.SetUploadWorker(uploadWorker)
}
