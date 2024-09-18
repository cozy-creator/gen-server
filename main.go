package main

import (
	"fmt"
	"os"

	cmd "github.com/cozy-creator/gen-server/cmd/cozy"
)

func main() {
	// rootCmd := cmd.GetRootCmd()
	// if err := rootCmd.Execute(); err != nil {
	// 	fmt.Println(err)
	// 	os.Exit(1)
	// }

	if err := config.InitConfig(); err != nil {
		fmt.Println(err)
		return
	}

	handler, err := filehandler.GetFileHandler()
	if err != nil {
		fmt.Println(err)
		return
	}

	logger.InitLogger(config.GetConfig())

	go worker.InitializeUploadWorker(handler, 10)
	go worker.StartGeneration(context.TODO())

	workflow, err := loadWorkflowFromFile("generate_images.json")
	// workflow, err := loadWorkflowFromFile("load_save_flow.json")
	if err != nil {
		fmt.Println(err)
		return
	}

	exec := executor.NewWorkflowExecutor(&workflow)
	err = exec.Execute(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
}

func loadWorkflowFromFile(filename string) (executor.Workflow, error) {
	flow, err := os.ReadFile(filename)
	if err != nil {
		return executor.Workflow{}, err
	}

	var workflow executor.Workflow
	err = json.Unmarshal(flow, &workflow)
	if err != nil {
		return executor.Workflow{}, err
	}

	return workflow, nil
}

// func loadAndSaveImageWorkflow() {
// 	filenames := []string{"7510d7fc15d58b5afeb380993d780fa535235617a73469a3174a98105dbad3da.png"}
// 	targetSize := image.Point{1024, 1024}

// 	images, err := imagenode.LoadImage(filenames, 0, targetSize)
// 	if err != nil {
// 		fmt.Println(err)
// 		return
// 	}

// 	_, err = imagenode.SaveImage(images, "png", false)
// 	if err != nil {
// 		fmt.Println(err)
// 		return
// 	}
// }

// func simpleGenerateImageWorkflow() {
// 	prompt := "A beautiful anime girl with long blue hair, wearing a white dress, standing in a field of flowers"
// 	numImages := 2
// 	randomSeed := 42
// 	outputFormat := "png"

// 	// TODO: add load model node

// 	// generate image node
// 	fmt.Println("generating images...\n")
// 	output, err := generationnode.GenerateImage("playground2.5", prompt, numImages, randomSeed, outputFormat)
// 	if err != nil {
// 		fmt.Println(err)
// 		return
// 	}

// 	images := []image.Image{}
// 	for {
// 		data, ok := <-output
// 		if !ok {
// 			break
// 		}

// 		fmt.Println("decoding image...\n")
// 		img, err := imagenode.DecodeImage(data, outputFormat)
// 		if err != nil {
// 			fmt.Println(err)
// 			return
// 		}

// 		images = append(images, img)
// 	}
// 	// decode image output node

// 	// save image node
// 	fmt.Println("saving images...\n")
// 	urls, err := imagenode.SaveImage(images, outputFormat, false)
// 	if err != nil {
// 		fmt.Println(err)
// 		return
// 	}

// 	for i, url := range urls {
// 		fmt.Printf("image saved %d to %s\n\n", i, url)
// 	}
// }
