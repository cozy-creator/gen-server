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
		return m.downloadCivitai(modelID, source.Location)
	case SourceTypeDirect:
		return m.downloadDirect(modelID, source.Location)
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

	downloader := pipeline.NewDiffusionPipelineDownloader(m.hubClient)
	_, err := downloader.Download(repoID, "", nil)
	if err != nil {
		return fmt.Errorf("failed to download model from HuggingFace: %w", err)
	}

	return nil
}

func (m *ModelDownloaderManager) downloadCivitai(modelID, urlStr string) error {
	m.logger.Info("Downloading from Civitai",
		zap.String("model_id", modelID),
		zap.String("url", urlStr),
	)

	// validate Civitai URL
	if !strings.Contains(urlStr, "civitai.com/api/download/models/") {
		return fmt.Errorf("invalid Civitai URL format. Expected format: https://civitai.com/api/download/models/<model_number>")
	}

	// get filename from redirect
	filename, err := m.getCivitaiFilename(urlStr)
	if err != nil {
		return fmt.Errorf("failed to get Civitai filename: %w", err)
	}

	fmt.Println("Civitai filename:", filename)

	destDir := m.getCachePath(modelID, &ModelSource{Type: SourceTypeCivitai})
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

	return m.downloadWithProgress(urlStr, destPath)
}


func (m *ModelDownloaderManager) downloadDirect(modelID, url string) error {
	m.logger.Info("Downloading from direct URL",
		zap.String("model_id", modelID),
		zap.String("url", url),
	)

	destPath := m.getCachePath(modelID, &ModelSource{Type: SourceTypeDirect})
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	return m.downloadWithProgress(url, destPath)
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


func (m *ModelDownloaderManager) downloadWithProgress(url, destPath string) error {
	// create temporary file
	tmpPath := destPath + ".tmp"
	
	b := backoff.NewExponentialBackOff()
	b.MaxElapsedTime = 5 * time.Minute
	b.InitialInterval = 1 * time.Second
	b.MaxInterval = 30 * time.Second

	// retry with backoff
	return backoff.Retry(func() error {
		return m.downloadWithResume(url, destPath, tmpPath)
	}, b)
}


func (m *ModelDownloaderManager) downloadWithResume(url, destPath, tmpPath string) error {
	// check for partial download
	var initialSize int64 = 0
	if info, err := os.Stat(tmpPath); err == nil {
		initialSize = info.Size()
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	if m.civitaiAPIKey != "" {
		req.Header.Set("Authorization", "Bearer " + m.civitaiAPIKey)
	}

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
		} else {
			return fmt.Errorf("resume failed with status %d", resp.StatusCode)
		}
	} else {
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("download failed with status %d", resp.StatusCode)
		}
		totalSize = resp.ContentLength
	}

	// open file in appropriate mode
	flag := os.O_CREATE | os.O_WRONLY
	if initialSize > 0 {
		flag |= os.O_APPEND
	}

	f, err := os.OpenFile(tmpPath, flag, 0644)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	// setup progress bar
	progress := mpb.New(
		mpb.WithWidth(60),
		mpb.WithRefreshRate(180*time.Millisecond),
	)

	bar := progress.AddBar(totalSize,
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

	// set initial progress
	if initialSize > 0 {
		bar.SetCurrent(initialSize)
	}

	downloadedSize := initialSize
	lastUpdate := time.Now()
	stallTimer := time.Duration(0)

	reader := bar.ProxyReader(resp.Body)
	buf := make([]byte, 32*1024) // 32KB buffer

	for {
		n, err := reader.Read(buf)
		if n > 0 {
			// write chunk
			if _, werr := f.Write(buf[:n]); werr != nil {
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

	// verify file
	if err := m.verifyFile(tmpPath); err != nil {
		return fmt.Errorf("failed to verify file: %w", err)
	}

	// move file to final destination
	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("failed to move file: %w", err)
	}

	return nil
}




