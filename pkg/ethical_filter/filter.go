package ethical_filter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"strings"

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

type StyleCatalog []struct {
	Name        string
	Description string
}

type PromptFilterResponse struct {
	Type   PromptFilterResponseType `json:"status"`
	Reason string                   `json:"reason"`
}

type PromptFilterResponseType string

const SEED int64 = 420

const (
	PromptFilterResponseTypeApproved PromptFilterResponseType = "approved"
	PromptFilterResponseTypeRejected PromptFilterResponseType = "rejected"
)

var styleCatalog = StyleCatalog{
	{Name: "abstract-art", Description: "Classical European"},
	{Name: "anime", Description: "Japanese animation style"},
	{Name: "cartoon", Description: "Western animation style art, simple"},
	{Name: "comic-book", Description: "Cartoonish, cel-shaded with with shading"},
	{Name: "digital-art", Description: "Almost photo-realistic looking, but clearly drawn with a digital brush"},
	{Name: "gothic", Description: "Dark, dramatic style with medieval influences"},
	{Name: "hyperrealism", Description: "Photo-realistic, but too detailed and perfect to be real"},
	{Name: "line-art", Description: "Hand-drawn black and white, sketch"},
	{Name: "oil-painting", Description: "18th century hand-painted brushstrokes"},
	{Name: "photo-realistic", Description: "Indistinguishable from a real photograph"},
	{Name: "pixar-3d", Description: "Cartoonish 3d models, like Monsters Inc."},
	{Name: "pop-art", Description: "Bold colors and hip-hop themed"},
	{Name: "retro game", Description: "Pixel-art art, 80's and 90's video game style"},
	{Name: "ukiyo-e", Description: "Traditional Japanese woodblock print style"},
	{Name: "videogame", Description: "Realistic 3d models, like Grand Theft Auto"},
	{Name: "watercolor", Description: "Soft, flowing style with transparent color washes"},
}

var knownStylesMap = func() map[string]struct{} {
    m := make(map[string]struct{})
    for _, style := range styleCatalog {
        m[strings.ToLower(style.Name)] = struct{}{}
    }
    return m
}()

func (f *SafetyFilter) InvokeChatGPT(ctx context.Context, positivePrompt, negativePrompt string) (*ChatGPTFilterResponse, error) {
	tmpl, err := template.New("systemPrompt").Parse(SystemPrompt)
	if err != nil {
		return nil, err
	}

	var tmplBuffer bytes.Buffer
	if err := tmpl.Execute(&tmplBuffer, styleCatalog); err != nil {
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

	// Filter out unknown styles
	res.Styles = normalizeStyles(res.Styles)

	if res.SexualizeChild || (res.Child && (res.Sexual || res.Nudity)) {
		return &PromptFilterResponse{
			Type:   PromptFilterResponseTypeRejected,
			Reason: "contains child sexual content",
		}, nil
	} else if res.Child && (res.Violence || res.Disturbing) {
		return &PromptFilterResponse{
			Type:   PromptFilterResponseTypeRejected,
			Reason: "contains children and violent or disturbing content",
		}, nil
	} else if (res.Sexual || res.Nudity) && len(res.Celebrities) > 0 || len(res.LiveActionCharacters) > 0 {
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
    normalizedStyles := make([]string, 0)
    for _, style := range styles {
        if _, exists := knownStylesMap[strings.ToLower(style)]; exists {
            normalizedStyles = append(normalizedStyles, strings.ToLower(style))
        }
    }
    return normalizedStyles
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