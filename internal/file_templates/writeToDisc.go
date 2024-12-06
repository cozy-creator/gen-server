package templates

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/viper"
)

func CreateCozyHomeDirs(cozyHome string) error {
	subdirs := []string{"assets", "models", "temp"}
	if err := os.MkdirAll(cozyHome, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create cozyhome directory: %w", err)
	}

	for _, subdir := range subdirs {
		dir := filepath.Join(cozyHome, subdir)
		if err := os.MkdirAll(dir, os.ModePerm); err != nil {
			return fmt.Errorf("failed to create %s directory: %w", subdir, err)
		}
	}

	return nil
}

func WriteExampleTemplates() error {
	cozyHome := viper.GetString("cozy_home")
	exampleEnvFilePath := filepath.Join(cozyHome, ".env.example")
	exampleConfigFilePath := filepath.Join(cozyHome, "config.example.yaml")

	// Check to see if exampleEnvFilePath does not exist, and if it doesn't then create it
	if _, err := os.Stat(exampleEnvFilePath); os.IsNotExist(err) {
		err = WriteEnv(exampleEnvFilePath)
		if err != nil {
			return fmt.Errorf("failed to create .env file: %w", err)
		}
	}

	// Check to see if exampleConfigFilePath does not exist, and if it doesn't then create it
	if _, err := os.Stat(exampleConfigFilePath); os.IsNotExist(err) {
		err = WriteConfig(exampleConfigFilePath)
		if err != nil {
			return fmt.Errorf("failed to create config.yaml file: %w", err)
		}
	}

	return nil
}
