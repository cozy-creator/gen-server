package model_downloader

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/cozy-creator/gen-server/internal/app"
    "github.com/cozy-creator/hf-hub/hub"
    "go.uber.org/zap"
	"github.com/vbauerster/mpb/v7"
	"github.com/cozy-creator/gen-server/internal/config"
)


type ModelDownloaderManager struct {
	app 		*app.App
	hubClient 	*hub.Client
	logger 		*zap.Logger
	ctx 		context.Context
	civitaiAPIKey string
	progress		*mpb.Progress
	progressMu		sync.Mutex
	subscriptionManager *SubscriptionManager
}

func NewModelDownloaderManager(app *app.App) (*ModelDownloaderManager, error) {
	hubClient := hub.DefaultClient()

	civitaiAPIKey := ""
	if app.Config().Civitai != nil {
		civitaiAPIKey = app.Config().Civitai.APIKey
	}

	progress := mpb.New(
		mpb.WithWidth(60),
		mpb.WithRefreshRate(180*time.Millisecond),
	)

	return &ModelDownloaderManager{
		app: 		app,
		hubClient: 	hubClient,
		logger: 	app.Logger.Named("model_downloader"),
		ctx: 		app.Context(),
		civitaiAPIKey: civitaiAPIKey,
		progress:		progress,
		subscriptionManager: NewSubscriptionManager(),
	}, nil
}

func (m *ModelDownloaderManager) InitializeModels() error {
	ctx := m.ctx
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

			select {
			case <-ctx.Done():
				return
			default:
				downloaded, err := m.IsDownloaded(modelID)
				if err != nil {
					errorChan <- fmt.Errorf("failed to check if model %s is downloaded: %w", modelID, err)
					m.subscriptionManager.SetModelStatus(modelID, StatusFailed)
					return
				}

			if !downloaded {
				m.subscriptionManager.SetModelStatus(modelID, StatusDownloading)
				m.logger.Info("Downloading model", zap.String("model_id", modelID))
				if err := m.Download(modelID); err != nil {
					errorChan <- fmt.Errorf("failed to download model %s: %w", modelID, err)
					m.subscriptionManager.SetModelStatus(modelID, StatusFailed)
				   }
				} else {
					m.logger.Info("Model already downloaded", zap.String("model_id", modelID))
					m.subscriptionManager.SetModelStatus(modelID, StatusReady)
				}
			}
		}(modelID)
	}

	// wait for all goroutines to finish
	done := make(chan struct{})
    go func() {
        wg.Wait()
        close(done)
        close(errorChan)
    }()

    select {
    case <-ctx.Done():
        return context.Canceled
    case <-done:
        // Check for any errors
        for err := range errorChan {
            if err != nil {
                return fmt.Errorf("error during model initialization: %w", err)
            }
        }
        return nil
    }
}

func (m *ModelDownloaderManager) WaitForModel(modelID string) error {
    resultChan := m.subscriptionManager.Subscribe(modelID)
    return <-resultChan
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

	// download main model
	if err := m.downloadFromSource(modelID, source); err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}

	// download components
	if len(modelConfig.Components) > 0 {
		var wg sync.WaitGroup
		errChan := make(chan error, len(modelConfig.Components))

		for name, comp := range modelConfig.Components {
			wg.Add(1)
			go func(name string, comp *config.ComponentDefs) {
				defer wg.Done()
			
				compSource, err := ParseModelSource(comp.Source)
				if err != nil {
					errChan <- fmt.Errorf("failed to parse component source for %s: %w", name, err)
					return
				}

				if err := m.downloadFromSource(fmt.Sprintf("%s___%s", modelID, name), compSource); err != nil {
					errChan <- fmt.Errorf("failed to download component %s: %w", name, err)
				}
			}(name, comp)
		}

		// wait for all components to download
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
			close(errChan)
		}()

		// wait for all components to download or context to be cancelled
		select {
		case <-m.ctx.Done():
			return m.ctx.Err()
		case <-done:
			// check for any errors
			for err := range errChan {
				if err != nil {
					return err
				}
			}
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

