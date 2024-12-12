package scripts

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const replicateBaseURL = "https://api.replicate.com/v1"

type ReplicateAI struct {
	APIKey string
}

type ReplicateGeneration struct {
	ID     string      `json:"id"`
	Status string      `json:"status"`
	Output interface{} `json:"output"`
	Error  string      `json:"error,omitempty"`
	URLs   struct {
		Get    string `json:"get"`
		Cancel string `json:"cancel"`
	} `json:"urls"`
}

type ReplicateGenerationRequest struct {
	Input          map[string]interface{} `json:"input"`
	Webhook        string                 `json:"webhook,omitempty"`
	WebhookEvents  []string               `json:"webhook_events_filter,omitempty"`
}

func NewReplicateAI(apiKey string) *ReplicateAI {
	if apiKey == "" {
		apiKey = os.Getenv("REPLICATE_API_KEY")
	}
	return &ReplicateAI{APIKey: apiKey}
}

func (r *ReplicateAI) doRequest(method, endpoint string, data interface{}) ([]byte, error) {
	url := replicateBaseURL + endpoint

	var requestBody io.Reader
	if data != nil {
		jsonData, err := json.Marshal(data)
		if err != nil {
			return nil, fmt.Errorf("error marshaling request: %w", err)
		}
		requestBody = bytes.NewBuffer(jsonData)
	}

	req, err := http.NewRequest(method, url, requestBody)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", r.APIKey))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response body: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	return body, nil
}


func (r *ReplicateAI) CreateRecraft(prompt string, style string, size string) (*ReplicateGeneration, error) {
    fmt.Printf("[Replicate] Creating generation with prompt: '%s', style: '%s', size: '%s'\n",
        prompt, style, size)

    req := ReplicateGenerationRequest{
        Input: map[string]interface{}{
            "prompt": prompt,
            "style": style,
            "size": size,
        },
    }

	body, err := r.doRequest("POST", "/models/recraft-ai/recraft-v3-svg/predictions", req)
	if err != nil {
		fmt.Printf("[Replicate] Error creating generation: %v\n", err)
		return nil, err
	}

	var gen ReplicateGeneration
	if err := json.Unmarshal(body, &gen); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	fmt.Printf("[Replicate] Generation created successfully with ID: %s\n", gen.ID)
	return &gen, nil
}

func (r *ReplicateAI) GetGeneration(getURL string) (*ReplicateGeneration, error) {
	req, err := http.NewRequest("GET", getURL, nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", r.APIKey))

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response body: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	var gen ReplicateGeneration
	if err := json.Unmarshal(body, &gen); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &gen, nil
}

// PollGeneration polls until generation is complete or fails
func (r *ReplicateAI) PollGeneration(getURL string) (*ReplicateGeneration, error) {
	lastStatus := ""
	startTime := time.Now()

	for i := 0; i < 30; i++ { // 30 attempts, 2s each = 1min timeout make shift while loop 🥲
		gen, err := r.GetGeneration(getURL)
		if err != nil {
			return nil, err
		}

		// Log status changes
		if gen.Status != lastStatus {
			fmt.Printf("[Replicate] Status changed to: %s (elapsed: %s)\n", 
				gen.Status, 
				time.Since(startTime).Round(time.Second))
			lastStatus = gen.Status
		}

		switch gen.Status {
		case "succeeded":
			fmt.Printf("[Replicate] Generation completed successfully in %s\n", 
				time.Since(startTime).Round(time.Second))
			return gen, nil
		case "failed":
			fmt.Printf("[Replicate] Generation failed after %s: %s\n", 
				time.Since(startTime).Round(time.Second), 
				gen.Error)
			return nil, fmt.Errorf("generation failed: %s", gen.Error)
		case "canceled":
			fmt.Printf("[Replicate] Generation was canceled after %s\n", 
				time.Since(startTime).Round(time.Second))
			return nil, fmt.Errorf("generation was canceled")
		default:
			time.Sleep(2 * time.Second)
			continue
		}
	}

	return nil, fmt.Errorf("generation timed out")
}