package model_downloader

import (
	"context"
	"fmt"
	// "os"
	"sync"
	"time"
	"path/filepath"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/services/lora_cache"
	"github.com/cozy-creator/gen-server/internal/types"
	"github.com/cozy-creator/hf-hub/hub"
	"github.com/vbauerster/mpb/v7"
	"go.uber.org/zap"
)


type ModelDownloaderManager struct {
	app 		*app.App
	hubClient 	*hub.Client
	logger 		*zap.Logger
	ctx 		context.Context
	civitaiAPIKey string
	// progress		*mpb.Progress
	modelStates     map[string]types.ModelState
	modelReadyChans map[string][]chan struct{}
	stateMu         sync.RWMutex
	loraCache		*lora_cache.LoRACache
}

func NewModelDownloaderManager(app *app.App) (*ModelDownloaderManager, error) {
	hubClient := hub.DefaultClient()
	

	civitaiAPIKey := ""
	if app.Config().Civitai != nil {
		civitaiAPIKey = app.Config().Civitai.APIKey
	}

	progress := mpb.New(
		mpb.WithWidth(60),
		mpb.WithRefreshRate(2*time.Second),
		// mpb.WithOutput(os.Stderr),
	)

	hubClient.Progress = progress

	// initialize lora cache
	loraCache, err := lora_cache.NewLoRACache(lora_cache.Config{
		EvictionThreshold: 97,
		BaseDir: filepath.Join(app.Config().ModelsDir, "loras"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize LoRA cache: %w", err)
	}

	return &ModelDownloaderManager{
		app: 		app,
		hubClient: 	hubClient,
		logger: 	app.Logger.Named("model_downloader"),
		ctx: 		app.Context(),
		civitaiAPIKey: civitaiAPIKey,
		// progress:		progress,
		modelStates:     make(map[string]types.ModelState),
		modelReadyChans: make(map[string][]chan struct{}),
		loraCache:		loraCache,
	}, nil
}

func (m *ModelDownloaderManager) GetModelState(modelID string) types.ModelState {
	m.stateMu.RLock()
	defer m.stateMu.RUnlock()
	
	state, exists := m.modelStates[modelID]
	if !exists {
		return types.ModelStateNotFound
	}
	return state
}

func (m *ModelDownloaderManager) SetModelState(modelID string, state types.ModelState) {
	m.stateMu.Lock()
	defer m.stateMu.Unlock()

	m.modelStates[modelID] = state

	// if model becomes ready, notify all waiting goroutines
	if state == types.ModelStateReady {
		if chans, exists := m.modelReadyChans[modelID]; exists {
			for _, ch := range chans {
				close(ch)
			}
			delete(m.modelReadyChans, modelID)
		}
	}
}

func (m *ModelDownloaderManager) WaitForModelReady(ctx context.Context, modelID string) error {
	readyChan := make(chan struct{})

	m.stateMu.Lock()
	state, exists := m.modelStates[modelID]
	if !exists || state == types.ModelStateNotFound {
		m.stateMu.Unlock()
		return fmt.Errorf("model %s not found", modelID)
	}

	// if model is already ready, return immediately
	if state == types.ModelStateReady {
		m.stateMu.Unlock()
		return nil
	}

	// Model is downloading, register a listener
	if m.modelReadyChans == nil {
		m.modelReadyChans = make(map[string][]chan struct{})
	}
	m.modelReadyChans[modelID] = append(m.modelReadyChans[modelID], readyChan)
	m.stateMu.Unlock()

	// wait for model to be ready or context to be cancelled
	select {
	case <-readyChan:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *ModelDownloaderManager) InitializeModels() error {
	// ctx := m.ctx
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if len(m.app.Config().EnabledModels) == 0 {
        m.logger.Info("No models enabled for generation")
        return nil
    }

	pipelineDefs := m.app.Config().PipelineDefs

	maxConcurrentModels := 3 // Limit number of concurrent model downloads
	modelSemaphore := make(chan struct{}, maxConcurrentModels)

	var wg sync.WaitGroup
	errorChan := make(chan error, len(pipelineDefs))

	done := make(chan struct{})

	for modelID := range pipelineDefs {
		modelSemaphore <- struct{}{}
		wg.Add(1)
		go func(modelID string) {
			defer wg.Done()
			defer func() { <-modelSemaphore }()

			select {
			case <-ctx.Done():
				return
			default:
				downloaded, err := m.IsDownloaded(modelID)
				if err != nil {
					errorChan <- fmt.Errorf("failed to check if model %s is downloaded: %w", modelID, err)
					return
				}
				
				if downloaded {
                    m.logger.Info("Model already downloaded", zap.String("model_id", modelID))
                    m.SetModelState(modelID, types.ModelStateReady)
                    return
                }

                m.logger.Info("Downloading model", zap.String("model_id", modelID))
                m.SetModelState(modelID, types.ModelStateDownloading)
                
                if err := m.Download(modelID); err != nil {
                    m.SetModelState(modelID, types.ModelStateNotFound)
                    errorChan <- fmt.Errorf("failed to download model %s: %w", modelID, err)
                    return
                }
                
                m.SetModelState(modelID, types.ModelStateReady)
			}
		}(modelID)
	}

	// wait for all goroutines to finish
	// done := make(chan struct{})
    go func() {
        wg.Wait()
        close(done)
        close(errorChan)
    }()

    select {
    case <-ctx.Done():
		// clean up model states for any downloading models
		m.logger.Warn("Context cancelled, waiting for in-progress downloads to complete...")
        for modelID := range pipelineDefs {
            if m.GetModelState(modelID) == types.ModelStateDownloading {
                m.SetModelState(modelID, types.ModelStateNotFound)
            }
        }
        return context.Canceled
    case <-done:
        // Check for any errors
        for err := range errorChan {
            if err != nil {
                return fmt.Errorf("error during model initialization: %w", err)
            }
        }

		// m.progress.Wait()

        return nil
    }
}

func (m *ModelDownloaderManager) Download(modelID string) error {
	m.SetModelState(modelID, types.ModelStateDownloading)
    defer func() {
        // Set state based on error
        if err := recover(); err != nil {
            m.SetModelState(modelID, types.ModelStateNotFound)
            panic(err) // re-panic after updating state
        }
    }()

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

	fmt.Printf("modelConfig: %v\n", modelConfig)

	// download components
	if len(modelConfig.Components) > 0 {
		maxConcurrentComponents := 5 // Limit component concurrency
		componentSemaphore := make(chan struct{}, maxConcurrentComponents)
		var wg sync.WaitGroup
		errChan := make(chan error, len(modelConfig.Components))

		for name, comp := range modelConfig.Components {
			// check if component has source else skip
			if comp.Source == "" {
				continue
			}

			componentSemaphore <- struct{}{}
			wg.Add(1)
			go func(name string, comp *config.ComponentDef) {
				defer wg.Done()
				defer func() { <-componentSemaphore }()

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

	// if download succeeds, set state to ready
	m.SetModelState(modelID, types.ModelStateReady)

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
		// check if component has source else skip
		if comp.Source == "" {
			continue
		}

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

