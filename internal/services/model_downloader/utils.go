package model_downloader

import (
	"path/filepath"
	"strings"
	"os"
	"fmt"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

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

func isRepoDownloaded(hubClient *hub.Client, repoID string, component string) (bool, error) {
    // Check if repo exists in cache
    storageFolder := filepath.Join(hubClient.CacheDir, repoFolderName(repoID, "model"))
    if !pathExists(storageFolder) {
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
    if !pathExists(snapshotPath) {
        return false, nil
    }

    if component != "" {
        componentPath := filepath.Join(snapshotPath, component)
        if !pathExists(componentPath) {
            return false, nil
        }

        if !checkFolderCompletenessWithVariants(componentPath) {
            return false, nil
        }

        return true, nil
    }

    // check model_index.json exists
    modelIndexPath := filepath.Join(snapshotPath, "model_index.json")
    if pathExists(modelIndexPath) {
        fmt.Println("model_index.json found for ", repoID)
        data, err := os.ReadFile(modelIndexPath)
        if err != nil {
            return false, nil
        }

        var modelIndex map[string]interface{}
        if err := json.Unmarshal(data, &modelIndex); err != nil {
            return false, nil
        }

        // get required folders
        requiredFolders := make(map[string]bool)
        for k, v := range modelIndex {
            if list, ok := v.([]interface{}); ok {
                if len(list) == 2 && list[0] != nil && list[1] != nil {
                    requiredFolders[k] = true
                }
            }
        }

        // ignore folders
        ignoredFolders := map[string]bool{
            "_class_name":         true,
            "_diffusers_version":  true,
            "scheduler":           true,
            "feature_extractor":   true,
            "tokenizer":           true,
            "tokenizer_2":         true,
            "tokenizer_3":         true,
            "safety_checker":      true,
        }

        for ignored := range ignoredFolders {
            delete(requiredFolders, ignored)
        }

        for folder := range requiredFolders {
            folderPath := filepath.Join(snapshotPath, folder)
            if !checkFolderCompletenessWithVariants(folderPath) {
                return false, nil
            }
        }

        return true, nil
    } else {
        fmt.Println("no model_index.json found")
        // For repos without model_index.json, check the blob folder
        blobFolder := filepath.Join(storageFolder, "blobs")
        if pathExists(blobFolder) {
            var hasIncomplete bool
            err := filepath.Walk(blobFolder, func(path string, info os.FileInfo, err error) error {
                if err != nil {
                    return err
                }
                if !info.IsDir() && strings.HasSuffix(info.Name(), ".incomplete") {
                    hasIncomplete = true
                    return fmt.Errorf("found incomplete file")
                }
                return nil
            })

            if err == nil && !hasIncomplete {
                return true, nil
            }
        }
    }

    return false, nil
}

func checkFolderCompletenessWithVariants(folderPath string) bool {
    variants := []string{"bf16", "fp16", "fp8", ""}
    for _, variant := range variants {
        if checkFolderCompleteness(folderPath, variant) {
            return true
        }
    }
    return false
}

func checkFolderCompleteness(folderPath string, variant string) bool {
    if !pathExists(folderPath) {
        return false
    }

    err := filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }

        if info.IsDir() {
            return nil
        }

        // Check for incomplete files
        if strings.HasSuffix(info.Name(), ".incomplete") {
            return fmt.Errorf("incomplete file found")
        }

        // Check for variant-specific files
        if variant != "" {
            if strings.HasSuffix(info.Name(), fmt.Sprintf("%s.safetensors", variant)) ||
                strings.HasSuffix(info.Name(), fmt.Sprintf("%s.bin", variant)) {
                return fmt.Errorf("found complete file")
            }
        } else {
            // Check for non-variant files
            if strings.HasSuffix(info.Name(), ".safetensors") ||
                strings.HasSuffix(info.Name(), ".bin") ||
                strings.HasSuffix(info.Name(), ".ckpt") {
                return fmt.Errorf("found complete file")
            }
        }

        return nil
    })

    return err != nil && err.Error() == "found complete file"
}



func (m *ModelDownloaderManager) isSourceDownloaded(modelID string, source *ModelSource) (bool, error) {
    switch source.Type {
    case SourceTypeHuggingface:
        var repoID, component string

        parts := strings.Split(source.Location, "/")
        if len(parts) > 2 {
            repoID = strings.Join(parts[:2], "/")
            component = parts[2]
        } else {
            repoID = source.Location
            component = ""
        }

        downloaded, err := isRepoDownloaded(m.hubClient, repoID, component)
        if err != nil {
            return false, fmt.Errorf("repo not downloaded: %w", err)
        }
        fmt.Println("repo downloaded: ", downloaded, "for ", source.Location)
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

func pathExists(path string) bool {
    _, err := os.Stat(path)
    return err == nil
}