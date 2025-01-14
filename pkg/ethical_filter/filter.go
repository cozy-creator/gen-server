package ethical_filter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html/template"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type SafetyFilter struct {
    client *openai.Client
}

// TO DO: perhaps in the future support multiple providers
func NewSafetyFilter(apiKey string) (*SafetyFilter, error) {
	if (apiKey == "") {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

    return &SafetyFilter{
        client: openai.NewClient(option.WithAPIKey(apiKey)),
    }, nil
}

type PromptFilterResponse struct {
	Accepted bool   `json:"accepted"`
	Reason   string `json:"reason"`
}

const SEED int64 = 420

func (f *SafetyFilter) InvokeChatGPT(ctx context.Context, positivePrompt, negativePrompt string) (*ChatGPTFilterResponse, error) {
	tmpl, err := template.New("systemPrompt").Parse(SystemPrompt)
	if err != nil {
		return nil, err
	}

	// For now, our system prompt template does not need any data
	var tmplBuffer bytes.Buffer
	if err := tmpl.Execute(&tmplBuffer, struct{}{}); err != nil {
		return nil, err
	}

	completion, err := f.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(tmplBuffer.String()),
			openai.UserMessage(fmt.Sprintf("Positive prompt: %s", positivePrompt)),
			openai.UserMessage(fmt.Sprintf("Negative prompt: %s", negativePrompt)),
		}),
		ResponseFormat: openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
			openai.ResponseFormatJSONObjectParam{
				Type: openai.F(openai.ResponseFormatJSONObjectTypeJSONObject),
			},
		),
		Seed:        openai.F(SEED),
		Model:       openai.F(openai.ChatModelGPT4oMini),
		Temperature: openai.F(0.2),
	})

	if err != nil {
		return nil, fmt.Errorf("request to ChatGPT failed: %w", err)
	}

	if len(completion.Choices) == 0 || len(completion.Choices[0].Message.Content) == 0 {
		return nil, fmt.Errorf("could not filter or validate prompt")
	}

	var res ChatGPTFilterResponse
	if err := json.Unmarshal([]byte(completion.Choices[0].Message.Content), &res); err != nil {
		return nil, fmt.Errorf("could not parse response: %w", err)
	}

	return &res, nil
}

func (f * SafetyFilter) EvaluatePrompt(ctx context.Context, positivePrompt, negativePrompt string) (*PromptFilterResponse, error) {
	res, err := f.InvokeChatGPT(ctx, positivePrompt, negativePrompt)
	if err != nil {
		return nil, err
	}

	response, err := f.EvaluateResponse(res)
	if err != nil {
		return nil, err
	}

	return &response, nil
}

func (f *SafetyFilter) EvaluateResponse(res *ChatGPTFilterResponse) (PromptFilterResponse, error) {
	if res.SexualizeChild || (res.Child && (res.Sexual || res.Nudity)) {
		return PromptFilterResponse{
			Accepted: false,
			Reason: "contains child sexual content",
		}, nil
	} else if res.Child && (res.Violence || res.Disturbing) {
		return PromptFilterResponse{
			Accepted: false,
			Reason: "contains children and violent or disturbing content",
		}, nil
		// len(res.LiveActionCharacters) > 0
	} else if (res.Sexual || res.Nudity) && len(res.Celebrities) > 0 {
		return PromptFilterResponse{
			Accepted: false,
			Reason: "contains non-consensual sexual or nude content of a real person",
		}, nil
	}

	return PromptFilterResponse{
		Accepted: true,
	}, nil
}

// func isNakedInPrompt(positivePrompt string) bool {
// 	terms := []string{"naked", "nude", "nudity", "porno", "sperm"}
// 	for _, term := range terms {
// 		if strings.Contains(strings.ToLower(positivePrompt), term) {
// 			return true
// 		}
// 	}
// 	return false
// }
