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

func GetWorkspacePath() (string, error) {
	return ExpandPath(config.GetConfig().WorkspacePath)
}

func GetAssetsPath() (string, error) {
	if config.GetConfig().AssetsPath == "" {
		workspacePath, err := GetWorkspacePath()
		if err != nil {
			return "", err
		}

		fmt.Println("workspacePath", workspacePath)
		return filepath.Join(workspacePath, "assets"), nil
	}

	fmt.Println("config.GetConfig().AssetsPath", config.GetConfig().AssetsPath)
	return ExpandPath(config.GetConfig().AssetsPath)
}

func GetModelsPath() (string, error) {
	if config.GetConfig().ModelsPath == "" {
		workspacePath, err := GetWorkspacePath()
		if err != nil {
			return "", err
		}
		return filepath.Join(workspacePath, "models"), nil
	}

	return ExpandPath(config.GetConfig().ModelsPath)
}

func GetTempPath() (string, error) {
	workspacePath, err := GetWorkspacePath()
	if err != nil {
		return "", err
	}

	return filepath.Join(workspacePath, "temp"), nil
}
