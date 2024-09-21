package scripts

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

func getAPIKey() string {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey != "" {
		return apiKey
	}

	fmt.Println("No API key found in environment variable OPENAI_API_KEY.")
	fmt.Print("Please enter your OpenAI API key manually: ")
	reader := bufio.NewReader(os.Stdin)
	apiKey, _ = reader.ReadString('\n')
	return strings.TrimSpace(apiKey)
}

func run() {
	apiKey := getAPIKey()

	if apiKey == "" {
		fmt.Println("API key cannot be empty")
		return
	}

	client := openai.NewClient(apiKey)
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter your message: ")
	userMessage, _ := reader.ReadString('\n')

	req := openai.ChatCompletionRequest{
		Model: openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: userMessage},
		},
		Stream: true, // Enable streaming
	}

	stream, err := client.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		fmt.Printf("ChatCompletion error: %v\n", err)
		return
	}
	defer stream.Close()

	for {
		response, err := stream.Recv()
		if err != nil {
			if err == context.Canceled || err == io.EOF {
				break
			}
			fmt.Printf("Stream error: %v\n", err)
			return
		}

		if len(response.Choices) > 0 {
			content := response.Choices[0].Delta.Content
			fmt.Print(content)
		}
	}
}
