package model_downloader

import (
	"context"
	"fmt"
	"sync"

	"github.com/cozy-creator/gen-server/internal/app"
    "github.com/cozy-creator/hf-hub/hub"
    "go.uber.org/zap"
)


type ModelDownloaderManager struct {
	app 		*app.App
	hubClient 	*hub.Client
	logger 		*zap.Logger
	ctx 		context.Context
	civitaiAPIKey string
}

func NewModelDownloaderManager(app *app.App) (*ModelDownloaderManager, error) {
	hubClient := hub.DefaultClient()

	// TODO: add civitai api key
	// if app.Config().CivitaiAPIKey != "" {
	// 	hubClient.WithAuth(app.Config().CivitaiAPIKey)
	// }

	return &ModelDownloaderManager{
		app: 		app,
		hubClient: 	hubClient,
		logger: 	app.Logger.Named("model_downloader"),
		ctx: 		app.Context(),
		// civitaiClient: app.Config().CivitaiAPIKey,
	}, nil
}

func (m *ModelDownloaderManager) InitializeModels() error {
	pipelineDefs := m.app.Config().PipelineDefs
	if len(pipelineDefs) == 0 {
		m.logger.Info("No models configured in pipeline definitions")
		return nil
	}

	var wg sync.WaitGroup
	errorChan := make(chan error, len(pipelineDefs))

	for modelID := range pipelineDefs {
		wg.Add(1)
		go func(modelID string) {
			defer wg.Done()

			downloaded, err := m.IsDownloaded(modelID)
			if err != nil {
				errorChan <- fmt.Errorf("failed to check if model %s is downloaded: %w", modelID, err)
				return
			}

			if !downloaded {
				m.logger.Info("Downloading model", zap.String("model_id", modelID))
				if err := m.Download(modelID); err != nil {
					errorChan <- fmt.Errorf("failed to download model %s: %w", modelID, err)
				}
			} else {
				m.logger.Info("Model already downloaded", zap.String("model_id", modelID))
			}
		}(modelID)
	}

	// wait for all goroutines to finish
	wg.Wait()
	close(errorChan)

	// check for any errors
	for err := range errorChan {
		if err != nil {
			return fmt.Errorf("error during model initialization: %w", err)
		}
	}

	return nil
}

func (m *ModelDownloaderManager) Download(modelID string) error {
	modelConfig, ok := m.app.Config().PipelineDefs[modelID]
	if !ok {
		return fmt.Errorf("model %s not found in config", modelID)
	}

	source, err := ParseModelSource(modelConfig.Source)
	if err != nil {
		return fmt.Errorf("failed to parse model source: %w", err)
	}

	if err := m.downloadFromSource(modelID, source); err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}

	// download components
	for name, comp := range modelConfig.Components {
		compSource, err := ParseModelSource(comp.Source)
		if err != nil {
			return fmt.Errorf("failed to parse component source for %s: %w", name, err)
		}

		if err := m.downloadFromSource(fmt.Sprintf("%s_%s", modelID, name), compSource); err != nil {
			return fmt.Errorf("failed to download component %s: %w", name, err)
		}
	}

	return nil
}


func (m *ModelDownloaderManager) IsDownloaded(modelID string) (bool, error) {
	modelConfig, ok := m.app.Config().PipelineDefs[modelID]
	if !ok {
		return false, fmt.Errorf("model %s not found in config", modelID)
	}

	// check main model
	source, err := ParseModelSource(modelConfig.Source)
	if err != nil {
		return false, fmt.Errorf("failed to parse model source: %w", err)
	}

	mainDownloaded, err := m.isSourceDownloaded(modelID, source)
	if err != nil || !mainDownloaded {
		return false, err
	}

	// check components
	for name, comp := range modelConfig.Components {
		compSource, err := ParseModelSource(comp.Source)
		if err != nil {
			return false, fmt.Errorf("invalid component source for %s: %w", name, err)
		}

		downloaded, err := m.isSourceDownloaded(fmt.Sprintf("%s_%s", modelID, name), compSource)
		if err != nil || !downloaded {
			return false, err
		}
	}

	return true, nil
}

