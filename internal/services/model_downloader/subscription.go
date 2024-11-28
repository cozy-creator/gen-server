package model_downloader

import (
	// "context"
	"sync"
	"errors"
)

type DownloadStatus string

const (
	StatusDownloading DownloadStatus = "downloading"
	StatusReady       DownloadStatus = "ready"
	StatusFailed      DownloadStatus = "failed"
)

type ModelRequest struct {
	ModelID string
	Result  chan error
}

type SubscriptionManager struct {
	mu              sync.RWMutex
	modelStatus     map[string]DownloadStatus
	pendingRequests map[string][]chan error
}

func NewSubscriptionManager() *SubscriptionManager {
	return &SubscriptionManager{
		modelStatus:     make(map[string]DownloadStatus),
		pendingRequests: make(map[string][]chan error),
	}
}

var ErrModelDownloadFailed = errors.New("model download failed")


func (sm *SubscriptionManager) SetModelStatus(modelID string, status DownloadStatus) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.modelStatus[modelID] = status

	// If model is ready or failed, notify all pending requests
	if status == StatusReady || status == StatusFailed {
		if channels, ok := sm.pendingRequests[modelID]; ok {
			var err error
			if status == StatusFailed {
				err = ErrModelDownloadFailed
			}
			for _, ch := range channels {
				ch <- err
				close(ch)
			}
			delete(sm.pendingRequests, modelID)
		}
	}
}

func (sm *SubscriptionManager) GetModelStatus(modelID string) DownloadStatus {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.modelStatus[modelID]
}

func (sm *SubscriptionManager) Subscribe(modelID string) <-chan error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	resultChan := make(chan error, 1)

	// If model is already ready, return immediately
	if sm.modelStatus[modelID] == StatusReady {
		resultChan <- nil
		close(resultChan)
		return resultChan
	}

	// If model failed, return error immediately
	if sm.modelStatus[modelID] == StatusFailed {
		resultChan <- ErrModelDownloadFailed
		close(resultChan)
		return resultChan
	}

	// Add to pending requests
	sm.pendingRequests[modelID] = append(sm.pendingRequests[modelID], resultChan)
	return resultChan
}