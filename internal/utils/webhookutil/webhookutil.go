package webhookutil

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type WebhookData struct {
	ID     string `json:"id"`
	Status string `json:"status,omitempty"`
}

func contains(arr []int, value int) bool {
	for _, v := range arr {
		if v == value {
			return true
		}
	}
	return false
}

func Invoke[T any](ctx context.Context, url string, data T) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	httpClient := &http.Client{Timeout: 2 * time.Minute}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	succsessStatuses := []int{http.StatusOK, http.StatusCreated, http.StatusAccepted}
	if !contains(succsessStatuses, resp.StatusCode) {
		return fmt.Errorf("Webhook returned non-200 status: %d", resp.StatusCode)
	}

	return nil
}

func InvokeWithRetries[T any](ctx context.Context, url string, data T, maxAttempts int) error {
	var err error
	backOff := time.Second
	for i := 0; i < maxAttempts; i++ {
		err = Invoke(ctx, url, data)
		if err != nil {
			time.Sleep(backOff)
			backOff *= 2
			continue
		}

		return nil
	}

	return err
}
