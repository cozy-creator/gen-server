package model_downloader

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/cozy-creator/hf-hub/hub"
	"github.com/cozy-creator/hf-hub/hub/pipeline"
    "github.com/vbauerster/mpb/v7"
    "github.com/vbauerster/mpb/v7/decor"
    "go.uber.org/zap"
	"github.com/cenkalti/backoff/v4"
)


func (m *ModelDownloaderManager) downloadFromSource(modelID string, source *ModelSource) error {
	switch source.Type {
	case SourceTypeHuggingface:
		return m.downloadHuggingFace(modelID, source.Location)
	case SourceTypeCivitai:
		return m.downloadCivitai(modelID, source)
	case SourceTypeDirect:
		return m.downloadDirect(modelID, source)
	case SourceTypeFile:
		return m.verifyLocalFile(source.Location)
	default:
		return fmt.Errorf("unsupported source type: %s", source.Type)
	}
}

func (m *ModelDownloaderManager) downloadHuggingFace(modelID, repoID string) error {
	m.logger.Info("Downloading from HuggingFace",
		zap.String("model_id", modelID),
		zap.String("repo_id", repoID),
	)

	// Check if this is a component download
    if strings.Contains(modelID, "_") {
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

	variants := []string{
		"bf16",
		"fp8",
		"fp16",
		"",
	}

	downloader := pipeline.NewDiffusionPipelineDownloader(m.hubClient)

	var lastErr error
	for _, variant := range variants {
		_, err := downloader.Download(repoID, variant, nil)
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

func (m *ModelDownloaderManager) downloadCivitai(modelID string, source *ModelSource) error {
	m.logger.Info("Downloading from Civitai",
		zap.String("model_id", modelID),
		zap.String("url", source.Location),
	)

	// validate Civitai URL
	if !strings.Contains(source.Location, "civitai.com/api/download/models/") {
		return fmt.Errorf("invalid Civitai URL format. Expected format: https://civitai.com/api/download/models/<model_number>")
	}

	// get filename from redirect
	filename, err := m.getCivitaiFilename(source.Location)
	if err != nil {
		return fmt.Errorf("failed to get Civitai filename: %w", err)
	}

	destDir := m.getCachePath(modelID, source)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	var destPath string
	if filename != "" {
		destPath = filepath.Join(destDir, filename)
	} else {
		// fallback to using modelID if we could not get the filename
		destPath = filepath.Join(destDir, modelID + ".safetensors")
		m.logger.Warn("Could not get filename from Civitai URL. Using modelID as fallback.",
			zap.String("fallback_name", filepath.Base(destPath)),
		)
	}

	return m.downloadWithProgress(source, destPath)
}


func (m *ModelDownloaderManager) downloadDirect(modelID string, source *ModelSource) error {
	m.logger.Info("Downloading from direct URL",
		zap.String("model_id", modelID),
		zap.String("url", source.Location),
	)

	destPath := m.getCachePath(modelID, &ModelSource{Type: SourceTypeDirect})
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	return m.downloadWithProgress(source, destPath)
}


func (m *ModelDownloaderManager) verifyLocalFile(path string) error {
	if err := m.verifyFile(path); err != nil {
		return fmt.Errorf("failed to verify local file: %w", err)
	}

	return nil
}


func (m *ModelDownloaderManager) getCivitaiFilename(urlStr string) (string, error) {
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
		Timeout: 30 * time.Second,
	}

	req, err := http.NewRequest("GET", urlStr, nil)
	if err != nil {
		return "", err
	}

	if m.civitaiAPIKey != "" {
		req.Header.Set("Authorization", "Bearer " + m.civitaiAPIKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusFound && resp.StatusCode != http.StatusMovedPermanently && resp.StatusCode != http.StatusTemporaryRedirect {
		return "", fmt.Errorf("expected redirect response, got status %d", resp.StatusCode)
	}

	location := resp.Header.Get("Location")
	if location == "" {
		return "", fmt.Errorf("no redirect location found")
	}

	redirectURL, err := url.Parse(location)
	if err != nil {
		return "", fmt.Errorf("failed to parse redirect location: %w", err)
	}

	// try to get filename from content disposition in query params
	queryParams := redirectURL.Query()
	if contentDisp := queryParams.Get("response-content-disposition"); contentDisp != "" {
		re := regexp.MustCompile(`filename="([^"]+)`)
		if matches := re.FindStringSubmatch(contentDisp); len(matches) > 1 {
			return matches[1], nil
		}
	}

	// fallback to path
	if path := redirectURL.Path; path != "" {
		return filepath.Base(path), nil
	}

	return "", nil
}


func (m *ModelDownloaderManager) downloadWithProgress(source *ModelSource, destPath string) error {
	// create temporary file
	tmpPath := destPath + ".tmp"
	
	b := backoff.NewExponentialBackOff()
	b.MaxElapsedTime = 5 * time.Minute
	b.InitialInterval = 1 * time.Second
	b.MaxInterval = 30 * time.Second

	// retry with backoff
	return backoff.Retry(func() error {
		return m.downloadWithResume(source.Location, destPath, tmpPath)
	}, b)
}


func (m *ModelDownloaderManager) downloadWithResume(url, destPath, tmpPath string) error {

	// check for partial download
	var initialSize int64 = 0
	if info, err := os.Stat(tmpPath); err == nil {
		initialSize = info.Size()
	}

	// use append mode if resuming, else create new
	flag := os.O_CREATE | os.O_WRONLY
	if initialSize > 0 {
		flag |= os.O_APPEND
	}


	out, err := os.OpenFile(tmpPath, flag, 0644)
	if err != nil {
		return err
	}
	defer func() {
		out.Sync() // Ensure all writes are flushed
		out.Close()
	}()

	client := &http.Client{
		Timeout: 0, // No total timeout
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout: 60 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout: 	60 * time.Second,
			ResponseHeaderTimeout: 	60 * time.Second,
			IdleConnTimeout: 		60 * time.Second,
		},
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	if m.civitaiAPIKey != "" {
		req.Header.Set("Authorization", "Bearer " + m.civitaiAPIKey)
	}

	if initialSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", initialSize))
	}

	
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// handle resume status
	var totalSize int64
	if initialSize > 0 {
		if resp.StatusCode == http.StatusPartialContent {
			totalSize = initialSize + resp.ContentLength
		} else if resp.StatusCode == http.StatusOK {
			// server does not support partial content (i.e. resume)
			m.logger.Warn("Server doesn't support resume, starting download from beginning")
			initialSize = 0
			out.Seek(0, 0)
			out.Truncate(0)
			totalSize = resp.ContentLength
		} else {
			return fmt.Errorf("resume failed with status %d", resp.StatusCode)
		}
	} else {
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("download failed with status %d", resp.StatusCode)
		}
		totalSize = resp.ContentLength
	}

	m.progressMu.Lock()

	bar := m.progress.AddBar(totalSize,
		mpb.PrependDecorators(
			decor.Name(filepath.Base(destPath), decor.WC{W: 40, C: decor.DidentRight}),
			decor.CountersKibiByte("% .2f / % .2f"),
		),
		mpb.AppendDecorators(
			decor.EwmaETA(decor.ET_STYLE_GO, 90),
			decor.Name(" ] "),
			decor.EwmaSpeed(decor.UnitKiB, "% .2f", 60),
		),
	)

	m.progressMu.Unlock()

	// set initial progress
	if initialSize > 0 {
		bar.SetCurrent(initialSize)
	}

	downloadedSize := initialSize
	lastUpdate := time.Now()
	stallTimer := time.Duration(0)

	reader := bar.ProxyReader(resp.Body)
	defer reader.Close()

	buf := make([]byte, 32*1024) // 32KB buffer


	for {
		n, err := reader.Read(buf)
		if n > 0 {
			// write chunk
			if _, werr := out.Write(buf[:n]); werr != nil {
				return fmt.Errorf("write failed: %w", werr)
			}

			// update progress
			downloadedSize += int64(n)

			// check for stalls
			now := time.Now()
			if now.Sub(lastUpdate) > 30*time.Second {
				stallTimer += now.Sub(lastUpdate)
				if stallTimer > 2*time.Minute {
					return fmt.Errorf("download stalled for too long")
				}
			} else {
				stallTimer = 0
				lastUpdate = now
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read failed: %w", err)
		}
	}

	// verify size
	if totalSize > 0 && downloadedSize != totalSize {
		return fmt.Errorf("download size mismatch: expected %d, got %d", totalSize, downloadedSize)
	}

	// Ensure all writes are complete
	if err := out.Sync(); err != nil {
		return fmt.Errorf("failed to sync file: %w", err)
	}

	// Close the file explicitly before rename
	out.Close()

	// move file to final destination
	if err := os.Rename(tmpPath, destPath); err != nil {
		fmt.Println("failed to move file:", err)
		return fmt.Errorf("failed to move file: %w", err)
	}

	// Use zap to log the model has been downloaded
	m.logger.Info("Model downloaded",
		zap.String("model_id", filepath.Base(destPath)),
	)

	return nil
}
