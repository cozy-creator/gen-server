package model_downloader

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	// "sync"
	"time"

	"github.com/cozy-creator/hf-hub/hub"
	"github.com/cozy-creator/hf-hub/hub/pipeline"
    "go.uber.org/zap"
	"github.com/vbauerster/mpb/v7"
)


func (m *ModelDownloaderManager) downloadFromSource(modelID string, source *ModelSource) error {
    var downloader hub.DownloadSource

    progress := mpb.New(
		mpb.WithWidth(60),
		mpb.WithRefreshRate(2*time.Second),
	)
    defer progress.Wait()
    
    switch source.Type {
    case SourceTypeHuggingface:
        return m.downloadHuggingFace(modelID, source.Location)
    case SourceTypeCivitai:
        downloader = hub.NewCivitaiSource(source.Location, m.civitaiAPIKey)
    case SourceTypeDirect:
        downloader = hub.NewDirectURLSource(source.Location)
    case SourceTypeFile:
        return m.verifyLocalFile(source.Location)
    default:
        return fmt.Errorf("unsupported source type: %s", source.Type)
    }

    fileInfo, err := downloader.GetFileInfo()
    if err != nil {
        return fmt.Errorf("failed to get file info: %w", err)
    }

    destDir := m.getCachePath(modelID, source)
    if err := os.MkdirAll(destDir, 0755); err != nil {
        return fmt.Errorf("failed to create cache directory: %w", err) 
    }

    destPath := filepath.Join(destDir, fileInfo.Filename)
    if fileInfo.Filename == "" {
        destPath = filepath.Join(destDir, modelID + ".safetensors")
        m.logger.Warn("Could not get filename. Using modelID as fallback.",
            zap.String("fallback_name", filepath.Base(destPath)),
        )
    }

    return downloader.Download(destPath, progress)
}

// ignore any component that doesn't have source

func (m *ModelDownloaderManager) downloadHuggingFace(modelID, repoID string) error {
	m.logger.Info("Downloading from HuggingFace",
		zap.String("model_id", modelID),
		zap.String("repo_id", repoID),
	)

	// Check if this is a component download
    if strings.Contains(modelID, "___") {
        // This is a component - use snapshot download
        parts := strings.Split(repoID, "/")
		if len(parts) > 2 {
            // Has subfolder - use pattern matching
            baseRepo := strings.Join(parts[:2], "/")
			subFolder := strings.Join(parts[2:], "/")

		params := &hub.DownloadParams{
			Repo: &hub.Repo{
				Id: baseRepo,
				Type: hub.ModelRepoType,
			},
                AllowPatterns: []string{fmt.Sprintf("%s/*", subFolder)},
		}

            m.logger.Info("Downloading component subfolder",
			zap.String("repo", baseRepo),
                zap.String("subfolder", subFolder),
		)

		_, err := m.hubClient.Download(params)
		if err != nil {
                return fmt.Errorf("failed to download component subfolder: %w", err)
            }
        } else {
            // Download entire repo
            params := &hub.DownloadParams{
                Repo: &hub.Repo{
                    Id: repoID,
                    Type: hub.ModelRepoType,
                },
            }
            
            m.logger.Info("Downloading full component repo",
                zap.String("repo", repoID),
            )

            _, err := m.hubClient.Download(params)
            if err != nil {
                return fmt.Errorf("failed to download component repo: %w", err)
		}
        }
		return nil
	}

    // Get model config 
    modelConfig, ok := m.app.Config().PipelineDefs[modelID]
    if !ok {
        return fmt.Errorf("model config not found for %s", modelID)
    }


	variants := []string{
		"bf16",
		"fp8",
		"fp16",
		"",
	}

	downloader := pipeline.NewDiffusionPipelineDownloader(m.hubClient)

	var lastErr error
	for _, variant := range variants {
        components := make(map[string]*hub.ComponentDef)
		for name, comp := range modelConfig.Components {
            if name == "scheduler" {
                continue
            }
			components[name] = &hub.ComponentDef{
				Source: comp.Source,
			}
		}
		_, err := downloader.Download(repoID, variant, nil, components)
		if err == nil {
			m.logger.Info("Successfully downloaded model",
				zap.String("model_id", modelID),
				zap.String("variant", variant),
			)
			return nil
		}
		lastErr = err
	}


	return fmt.Errorf("failed to download model from HuggingFace: %w", lastErr)
}


func (m *ModelDownloaderManager) verifyLocalFile(path string) error {
	if err := m.verifyFile(path); err != nil {
		return fmt.Errorf("failed to verify local file: %w", err)
	}

	return nil
}

// func fileExists(path string) bool {
//     _, err := os.Stat(path)
//     return err == nil
// }


func (m *ModelDownloaderManager) DownloadLoRA(url string) (string, error) {
    // check if lora is already cached
    // metadata, err := m.loraCache.Get(url)
    // if err != nil {
    //     return "", fmt.Errorf("failed to check LoRA cache: %w", err)
    // }

    // if metadata != nil && fileExists(metadata.FilePath) {
    //     return metadata.FilePath, nil
    // }

    progress := mpb.New(
		mpb.WithWidth(60),
		mpb.WithRefreshRate(2*time.Second),
	)
    defer progress.Wait()

    destPath, err := m.loraCache.GetLoRAPath(url)
    if err != nil {
        return "", fmt.Errorf("failed to get LoRA path: %w", err)
    }

    // Check if file already exists
    if _, err := os.Stat(destPath); err == nil {
        return destPath, nil
    }

    // create directory if it doesn't exist
    if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
        return "", fmt.Errorf("failed to create directory: %w", err)
    }

    source := hub.NewDirectURLSource(url)


    if err := source.Download(destPath, progress); err != nil {
        os.Remove(destPath)
        return "", fmt.Errorf("failed to download LoRA: %w", err)
    }

    fmt.Println("Adding LoRA to cache", url, "->", destPath)

    // add to cache
    // if err := m.loraCache.Add(url, destPath); err != nil {
    //     os.Remove(destPath) // clean up the downloaded file if it fails to add to cache
    //     return "", fmt.Errorf("failed to add LoRA to cache: %w", err)
    // }

    fmt.Println("Downloaded LoRA", url, "to", destPath)

    return destPath, nil
}

func (m *ModelDownloaderManager) DownloadMultipleLoRAs(urls []string) ([]string, error) {
    paths := make([]string, len(urls))
    
    for i, url := range urls {
        path, err := m.DownloadLoRA(url)
        if err != nil {
            // Clean up any previously downloaded files
            for j := 0; j < i; j++ {
                if paths[j] != "" {
                    os.Remove(paths[j])
                }
            }
            return nil, fmt.Errorf("failed to download LoRA %s: %w", url, err)
        }
        paths[i] = path
    }

    return paths, nil
}