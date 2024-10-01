package scripts

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

type OpenAINodeInputs struct {
	Message string `json:"message"`
}

type OpenAINodeOutput struct {
	Response string `json:"response"`
}


func CallOpenAINode(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	// Get API key from environment
	apiKey := getAPIKeyFromEnv()
	if apiKey == "" {
		return nil, fmt.Errorf("API key is missing")
	}

	// Extract the user message from the inputs
	message, err := getMessage(inputs)
	if err != nil {
		return nil, err
	}

	// Initialize the OpenAI client
	client := openai.NewClient(apiKey)

	// Prepare the request for chat completion
	req := openai.ChatCompletionRequest{
		Model: openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: message},
		},
		Stream: true, // Enable streaming for real-time responses
	}

	// Create a streaming chat completion
	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("stream creation error: %v", err)
	}
	defer stream.Close()

	// Process the streaming response
	var responseBuilder strings.Builder
	for {
		response, err := stream.Recv()
		if err != nil {
			if err == context.Canceled || err == io.EOF {
				break // Stream finished
			}
			return nil, fmt.Errorf("stream error: %v", err)
		}

		// Append the content to the response builder
		if len(response.Choices) > 0 {
			content := response.Choices[0].Delta.Content
			responseBuilder.WriteString(content)
			fmt.Print(content) // Print the content in real-time
		}
	}

	// Extract the full response and return it as output
	output := map[string]interface{}{
		"response": responseBuilder.String(),
	}
	return output, nil
}

// getAPIKeyFromEnv retrieves the OpenAI API key from the environment variables
func getAPIKeyFromEnv() string {
	apiKey := os.Getenv("OPENAI_API_KEY")
	return apiKey
}

// getMessage extracts the message from the inputs map
func getMessage(inputs map[string]interface{}) (string, error) {
	message, ok := inputs["message"].(string)
	if !ok {
		return "", fmt.Errorf("message input is invalid")
	}
	return message, nil
}