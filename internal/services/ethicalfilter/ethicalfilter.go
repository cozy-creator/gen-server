package ethicalfilter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"strings"
	"time"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type ChatGPTFilterResponse struct {
	SexualizeChild bool                          `json:"sexualize_child"`
	Child          bool                          `json:"child"`
	Nudity         bool                          `json:"nudity"`
	Sexual         bool                          `json:"sexual"`
	Violence       bool                          `json:"violence"`
	Disturbing     bool                          `json:"disturbing"`
	RequestedText  bool                          `json:"requested_text"`
	Persons        []ChatGPTFilterResponsePerson `json:"persons"`
	Styles         []string                      `json:"styles"`
}

type PromptTemplateData struct {
	Styles []string
}

type ChatGPTFilterResponsePerson struct {
	Name       string `json:"name"`
	RealPerson bool   `json:"real_person"`
}

type PromptFilterResponse struct {
	Type   PromptFilterResponseType `json:"status"`
	Reason string                   `json:"reason"`
}

type PromptFilterResponseType string

const (
	PromptFilterResponseTypeApproved PromptFilterResponseType = "approved"
	PromptFilterResponseTypeRejected PromptFilterResponseType = "rejected"
)

var styleExpertise []string

func init() {
	styleExpertise = []string{
		"photo",
		"hyper-real",
		"anime",
		"cartoon",
		"3d-game",
		"pixar",
		"classical-art",
		"physical-art",
		"videogame",
		"NSFW",
	}
}

func InvokeChatGPT(ctx context.Context, cfg *config.Config, positivePrompt, negativePrompt string) (*ChatGPTFilterResponse, error) {
	client := openai.NewClient(option.WithAPIKey(cfg.OpenAI.APIKey))
	tmpl, err := template.New("promptFilter").Parse(GetPromptFilterTemplate())
	if err != nil {
		return nil, err
	}

	var tmplBuffer bytes.Buffer
	if err := tmpl.Execute(&tmplBuffer, PromptTemplateData{Styles: styleExpertise}); err != nil {
		return nil, err
	}

	completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
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
		Seed:        openai.F(int64(time.Now().Unix())),
		Model:       openai.F(openai.ChatModelGPT4o),
		Temperature: openai.F(0.2),
	})

	if err != nil {
		return nil, err
	}

	if len(completion.Choices) == 0 || len(completion.Choices[0].Message.Content) == 0 {
		return nil, fmt.Errorf("could not filter or validate prompt")
	}

	var response ChatGPTFilterResponse
	if err := json.Unmarshal([]byte(completion.Choices[0].Message.Content), &response); err != nil {
		return nil, fmt.Errorf("could not parse response: %w", err)
	}

	return &response, nil
}

func FilterPrompt(ctx context.Context, cfg *config.Config, positivePrompt, negativePrompt string) (*PromptFilterResponse, error) {
	response, err := InvokeChatGPT(ctx, cfg, positivePrompt, negativePrompt)
	if err != nil {
		return nil, err
	}

	fmt.Println("cresponse")
	fmt.Println(response)

	// filter off unknown styles
	response.Styles = normalizeStyles(response.Styles)

	if isCPInPrompt(response, positivePrompt) {
		return &PromptFilterResponse{
			Type:   PromptFilterResponseTypeRejected,
			Reason: "contains child naked or sexual content",
		}, nil
	} else if response.Child && (response.Violence || response.Disturbing) {
		return &PromptFilterResponse{
			Type:   PromptFilterResponseTypeRejected,
			Reason: "contains child violence or disturbing content",
		}, nil
	} else if (response.Sexual || response.Nudity) && hasRealPerson(response.Persons) {
		return &PromptFilterResponse{
			Type:   PromptFilterResponseTypeRejected,
			Reason: "contains real-person sexual or nude content",
		}, nil
	}

	return &PromptFilterResponse{
			Type: PromptFilterResponseTypeApproved,
		},
		nil
}

func normalizeStyles(styles []string) []string {
	normalizedStyles := []string{}
	for _, style := range styles {
		lStyle := strings.ToLower(style)
		if isKnownStyle(lStyle) {
			normalizedStyles = append(normalizedStyles, lStyle)
		}
	}
	return normalizedStyles
}

func isKnownStyle(style string) bool {
	for _, knownStyle := range styleExpertise {
		if strings.ToLower(style) == knownStyle {
			return true
		}
	}
	return false
}

func isCPInPrompt(response *ChatGPTFilterResponse, positivePrompt string) bool {
	isNakedInPrompt := isNakedInPrompt(positivePrompt)
	return response.SexualizeChild || (response.Child && (response.Sexual || response.Nudity)) || (response.Child && isNakedInPrompt)
}

func hasRealPerson(persons []ChatGPTFilterResponsePerson) bool {
	for _, person := range persons {
		if person.RealPerson {
			return true
		}
	}
	return false
}

func isNakedInPrompt(positivePrompt string) bool {
	terms := []string{"naked", "nude", "nudity", "porno", "sperm"}
	for _, term := range terms {
		if strings.Contains(strings.ToLower(positivePrompt), term) {
			return true
		}
	}
	return false
}
