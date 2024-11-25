package model_downloader

import (
	"path/filepath"
	"strings"
	"os"
	"fmt"
	"crypto/sha256"
	"encoding/hex"

	"github.com/cozy-creator/hf-hub/hub"
)

func repoFolderName(repoID string, repoType string) string {
	// converts "username/repo" to "models--username--repo" (for models. same goes for datasets and spaces)
	repoParts := strings.Split(repoID, "/")
	parts := append([]string{repoType + "s"}, repoParts...)
	return strings.Join(parts, "--")
}

func (m *ModelDownloaderManager) getCachePath(modelID string, source *ModelSource) string {
	if source.Type == SourceTypeHuggingface {
		return filepath.Join(m.hubClient.CacheDir, repoFolderName(source.Location, "model"))
	}

	h := sha256.New()
	h.Write([]byte(source.Location))
	urlHash := hex.EncodeToString(h.Sum(nil))[:8]

	safeID := strings.ReplaceAll(modelID, "/", "-")
	return filepath.Join(m.app.Config().ModelsDir, fmt.Sprintf("%s--%s", safeID, urlHash))

}

func isRepoDownloaded(hubClient *hub.Client, repoID string) (bool, error) {
    // Check if repo exists in cache
    storageFolder := filepath.Join(hubClient.CacheDir, repoFolderName(repoID, "model"))
    if _, err := os.Stat(storageFolder); err != nil {
        return false, nil
    }

    // Check for main branch reference
    refPath := filepath.Join(storageFolder, "refs", "main")
    commitHash, err := os.ReadFile(refPath)
    if err != nil {
        return false, nil
    }

    // Check snapshot folder exists
    snapshotPath := filepath.Join(storageFolder, "snapshots", string(commitHash))
    if _, err := os.Stat(snapshotPath); err != nil {
        return false, nil
    }

    return true, nil
}


func (m *ModelDownloaderManager) isSourceDownloaded(modelID string, source *ModelSource) (bool, error) {
    switch source.Type {
    case SourceTypeHuggingface:
        downloaded, err := isRepoDownloaded(m.hubClient, source.Location)
        if err != nil {
            return false, fmt.Errorf("repo not downloaded: %w", err)
        }
        return downloaded, nil

    case SourceTypeFile:
        if !m.isFileValid(source.Location) {
            return false, fmt.Errorf("local file invalid or missing: %s", source.Location)
        }
        return true, nil

    case SourceTypeCivitai, SourceTypeDirect:
        baseDir := m.getCachePath(modelID, source)
        return m.isAnyValidModelInDir(baseDir), nil

    default:
        return false, fmt.Errorf("unsupported source type: %s", source.Type)
    }
}

//  check if any valid model file exists in a directory
func (m *ModelDownloaderManager) isAnyValidModelInDir(dir string) bool {
    entries, err := os.ReadDir(dir)
    if err != nil {
        return false
    }

    for _, entry := range entries {
        if entry.IsDir() {
            continue
        }
        path := filepath.Join(dir, entry.Name())
        if m.isFileValid(path) {
            return true
        }
    }
    return false
}

func (m *ModelDownloaderManager) isFileValid(path string) bool {
    return m.verifyFile(path) == nil
}

func (m *ModelDownloaderManager) verifyFile(path string) error {
    if _, err := os.Stat(path); err != nil {
        return fmt.Errorf("file does not exist: %w", err)
    }

    // Check file size
    info, err := os.Stat(path)
    if err != nil {
        return fmt.Errorf("failed to get file info: %w", err)
    }
    if info.Size() < 1024*1024 { // 1MB minimum
        return fmt.Errorf("file too small: %d bytes", info.Size())
    }

    // Check extension
    ext := strings.ToLower(filepath.Ext(path))
    validExts := map[string]bool{
        ".safetensors": true,
        ".ckpt":       true,
        ".pt":         true,
        ".bin":        true,
    }
    if !validExts[ext] {
        return fmt.Errorf("invalid file extension: %s", ext)
    }

    // Try to open and read some data to verify integrity
    f, err := os.Open(path)
    if err != nil {
        return fmt.Errorf("failed to open file: %w", err)
    }
    defer f.Close()

    // Read first and last 1MB
    buf := make([]byte, 1024*1024)
    if _, err := f.Read(buf); err != nil {
        return fmt.Errorf("failed to read file start: %w", err)
    }
    if _, err := f.Seek(-1024*1024, 2); err != nil {
        return fmt.Errorf("failed to seek file end: %w", err)
    }
    if _, err := f.Read(buf); err != nil {
        return fmt.Errorf("failed to read file end: %w", err)
    }

    return nil
}