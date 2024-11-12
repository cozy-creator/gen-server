package scripts

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

const baseURL = "https://api.lumalabs.ai/dream-machine/v1"

type LumaAI struct {
	AuthToken string
}

type Generation struct {
	ID     string `json:"id"`
	State  string `json:"state"`
	Assets struct {
		Video string `json:"video"`
	} `json:"assets"`
}

type GenerationRequest struct {
	Prompt      string                 `json:"prompt"`
	AspectRatio string                 `json:"aspect_ratio,omitempty"`
	Loop        bool                   `json:"loop,omitempty"`
	Keyframes   map[string]interface{} `json:"keyframes,omitempty"`
}

type Keyframe struct {
	Type string `json:"type"`
	URL  string `json:"url,omitempty"`
	ID   string `json:"id,omitempty"`
}

func NewLumaAI(authToken string) *LumaAI {
	if authToken == "" {
		authToken = os.Getenv("LUMA_AI_API_KEY")
	}
	return &LumaAI{AuthToken: authToken}
}

func (l *LumaAI) doRequest(method, endpoint string, data interface{}) ([]byte, error) {
	url := baseURL + endpoint

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

	req.Header.Set("Authorization", "Bearer "+l.AuthToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	client := &http.Client{}
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

func (l *LumaAI) CreateGeneration(req GenerationRequest) (*Generation, error) {
	body, err := l.doRequest("POST", "/generations", req)
	if err != nil {
		return nil, err
	}

	var gen Generation
	if err := json.Unmarshal(body, &gen); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &gen, nil
}

func (l *LumaAI) GetGeneration(id string) (*Generation, error) {
	body, err := l.doRequest("GET", fmt.Sprintf("/generations/%s", id), nil)
	if err != nil {
		return nil, err
	}

	var gen Generation
	if err := json.Unmarshal(body, &gen); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &gen, nil
}

func (l *LumaAI) ListGenerations(limit, offset int) ([]Generation, error) {
	body, err := l.doRequest("GET", fmt.Sprintf("/generations?limit=%d&offset=%d", limit, offset), nil)
	if err != nil {
		return nil, err
	}

	var gens []Generation
	if err := json.Unmarshal(body, &gens); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return gens, nil
}

func (l *LumaAI) DeleteGeneration(id string) error {
	_, err := l.doRequest("DELETE", fmt.Sprintf("/generations/%s", id), nil)
	return err
}

func (l *LumaAI) ListCameraMotions() ([]string, error) {
	body, err := l.doRequest("GET", "/generations/camera_motion/list", nil)
	if err != nil {
		return nil, err
	}

	var motions []string
	if err := json.Unmarshal(body, &motions); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return motions, nil
}

func (l *LumaAI) DownloadVideo(url, filename string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("error downloading video: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("error downloading video (status %d)", resp.StatusCode)
	}

	out, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating file: %w", err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("error writing file: %w", err)
	}

	return nil
}

func (l *LumaAI) ExtendVideo(prompt string, generationID string, isReverse bool) (*Generation, error) {
	keyframes := make(map[string]interface{})
	frameKey := "frame0"
	if isReverse {
		frameKey = "frame1"
	}
	keyframes[frameKey] = Keyframe{
		Type: "generation",
		ID:   generationID,
	}

	req := GenerationRequest{
		Prompt:    prompt,
		Keyframes: keyframes,
	}

	body, err := l.doRequest("POST", "/generations", req)
	if err != nil {
		return nil, err
	}

	var gen Generation
	if err := json.Unmarshal(body, &gen); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &gen, nil
}

func (l *LumaAI) ExtendVideoWithEndFrame(prompt, generationID, imageURL string) (*Generation, error) {
	keyframes := map[string]interface{}{
		"frame0": Keyframe{
			Type: "generation",
			ID:   generationID,
		},
		"frame1": Keyframe{
			Type: "image",
			URL:  imageURL,
		},
	}

	req := GenerationRequest{
		Prompt:    prompt,
		Keyframes: keyframes,
	}

	body, err := l.doRequest("POST", "/generations", req)
	if err != nil {
		return nil, err
	}

	var gen Generation
	if err := json.Unmarshal(body, &gen); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &gen, nil
}

func (l *LumaAI) InterpolateBetweenVideos(prompt, startGenerationID, endGenerationID string) (*Generation, error) {
	keyframes := map[string]interface{}{
		"frame0": Keyframe{
			Type: "generation",
			ID:   startGenerationID,
		},
		"frame1": Keyframe{
			Type: "generation",
			ID:   endGenerationID,
		},
	}

	req := GenerationRequest{
		Prompt:    prompt,
		Keyframes: keyframes,
	}

	body, err := l.doRequest("POST", "/generations", req)
	if err != nil {
		return nil, err
	}

	var gen Generation
	if err := json.Unmarshal(body, &gen); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &gen, nil
}

func (l *LumaAI) ImageToVideo(prompt, imageURL string, aspectRatio string, loop bool) (*Generation, error) {
	keyframes := map[string]interface{}{
		"frame0": Keyframe{
			Type: "image",
			URL:  imageURL,
		},
	}

	req := GenerationRequest{
		Prompt:      prompt,
		Keyframes:   keyframes,
		AspectRatio: aspectRatio,
		Loop:        loop,
	}

	body, err := l.doRequest("POST", "/generations", req)
	if err != nil {
		return nil, err
	}

	var gen Generation
	if err := json.Unmarshal(body, &gen); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &gen, nil
}
