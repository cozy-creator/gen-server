package utils

import (
	"cozy-creator/go-cozy/internal/config"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func ExpandPath(path string) (string, error) {
	if strings.HasPrefix(path, "~") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get user home directory: %w", err)
		}
		// Replace "~" with the home directory path
		path = filepath.Join(homeDir, path[1:])
	}
	return path, nil
}

func GetCozyHomeDir() (string, error) {
	return ExpandPath(config.GetConfig().HomeDir)
}

func GetCozyAssetsDir() (string, error) {
	if config.GetConfig().AssetsDir == "" {
		homeDir, err := GetCozyHomeDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(homeDir, "assets"), nil
	}

	return ExpandPath(config.GetConfig().AssetsDir)
}

func GetCozyModelsDir() (string, error) {
	if config.GetConfig().ModelsDir == "" {
		homeDir, err := GetCozyHomeDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(homeDir, "models"), nil
	}

	return ExpandPath(config.GetConfig().ModelsDir)
}

func GetTempDir() (string, error) {
	homeDir, err := GetCozyHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(homeDir, "temp"), nil
}
