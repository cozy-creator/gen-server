package types

import "context"

type ModelState string

const (
    ModelStateReady       ModelState = "ready"
    ModelStateDownloading ModelState = "downloading"
    ModelStateNotFound    ModelState = "not_found"
)

type ModelDownloader interface {
    GetModelState(modelID string) ModelState
    WaitForModelReady(ctx context.Context, modelID string) error
    DownloadMultipleLoRAs(urls []string) ([]string, error)
}