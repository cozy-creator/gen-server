package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/gen-server/internal/templates"

	"github.com/joho/godotenv"
	"github.com/spf13/viper"
)

const (
	FilesystemLocal = "local"
	FilesystemS3    = "s3"
)

const cozyPrefix = "COZY"

type Config struct {
	Port          int       `mapstructure:"port"`
	Host          string    `mapstructure:"host"`
	TcpPort       int       `mapstructure:"tcp_port"`
	CozyHome      string    `mapstructure:"cozy_home"`
	TcpTimeout    int       `mapstructure:"tcp_timeout"`
	Environment   string    `mapstructure:"environment"`
	AssetsDir     string    `mapstructure:"assets_dir"`
	ModelsDir     string    `mapstructure:"models_dir"`
	TempDir       string    `mapstructure:"temp_dir"`
	AuxModelsDirs []string  `mapstructure:"aux_models_dirs"`
	Filesystem    string    `mapstructure:"filesystem_type"`
	S3            *S3Config `mapstructure:"s3"`
}

type S3Config struct {
	Folder    string `mapstructure:"folder"`
	Region    string `mapstructure:"region_name"`
	Bucket    string `mapstructure:"bucket_name"`
	AccessKey string `mapstructure:"access_key"`
	SecretKey string `mapstructure:"secret_key"`
	PublicUrl string `mapstructure:"public_url"`
}

var config *Config

func InitConfig() error {
	cozyHome, err := getCozyHome()
	if err != nil {
		return err
	}

	assetsDir, err := getAssetsDir(cozyHome)
	if err != nil {
		return err
	}

	modelsDir, err := getModelsDir(cozyHome)
	if err != nil {
		return err
	}

	tempDir, err := getTempDir(cozyHome)
	if err != nil {
		return err
	}

	// Set the cozy home directory, assets directory, models directory, and temp directory in viper
	viper.Set("cozy_home", cozyHome)
	viper.Set("assets_dir", assetsDir)
	viper.Set("models_dir", modelsDir)
	viper.Set("temp_dir", tempDir)

	envFile := viper.GetString("env_file")
	configFile := viper.GetString("config_file")

	envFile = filepath.Join(cozyHome, ".env")
	configFile = filepath.Join(cozyHome, "config.yaml")

	if _, err := os.Stat(envFile); err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("failed to stat .env file: %w", err)
		}

		if err := templates.WriteEnv(envFile); err != nil {
			return fmt.Errorf("failed to create .env file: %w", err)
		}

		if err := godotenv.Load(envFile); err != nil {
			return fmt.Errorf("failed to load env file: %w", err)
		}
	}

	if _, err := os.Stat(configFile); err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("failed to stat config.yaml file: %w", err)
		}

		if err := templates.WriteConfig(configFile); err != nil {
			return fmt.Errorf("failed to create config.yaml file: %w", err)
		}
	}

	if envFile != "" {
		if err := godotenv.Load(envFile); err != nil {
			return fmt.Errorf("failed to load env file: %w", err)
		}
	}

	viper.AutomaticEnv()
	viper.SetEnvPrefix(cozyPrefix)
	viper.SetEnvKeyReplacer(strings.NewReplacer(`.`, `_`))
	if configFile != "" {
		viper.SetConfigFile(configFile)
	} else {
		viper.SetConfigType("yaml")
		viper.SetConfigName("config")
		viper.AddConfigPath(cozyHome)
	}

	if err := LoadConfig(false); err != nil {
		if errors.As(err, &viper.ConfigFileNotFoundError{}) {
			fmt.Println("No config file found. Using default config.")
		} else {
			return err
		}
	}

	return nil
}

func LoadConfig(reload bool) error {
	if config != nil && !reload {
		return fmt.Errorf("config already loaded")
	}

	if err := viper.ReadInConfig(); err != nil {
		return fmt.Errorf("error reading config: %w", err)
	}

	config = &Config{}
	err := viper.Unmarshal(config)
	if err != nil {
		return fmt.Errorf("error unmarshalling config: %w", err)
	}

	return nil
}

func GetConfig() *Config {
	if config == nil {
		panic("config not loaded")
	}

	return config
}

// Returns the cozy home directory path.
// It attempts to retrieve the cozy home directory from the following sources in order:
// 1. The `cozy_home` flag from viper.
// 2. The `COZY_HOME` environment variable.
// 3. The default cozy home directory.
func getCozyHome() (string, error) {
	cozyHome := viper.GetString("cozy_home")
	if cozyHome == "" {
		cozyHome = os.Getenv("COZY_HOME")
		if cozyHome == "" {
			cozyHome = DefaultCozyHome
		}
	}

	cozyHome, err := pathutil.ExpandPath(cozyHome)
	if err != nil {
		return "", fmt.Errorf("failed to expand cozy home path: %w", err)
	}

	return cozyHome, nil
}

func getAssetsDir(cozyHome string) (string, error) {
	if cozyHome == "" {
		return "", ErrCozyHomeNotSet
	}

	assetsDir := viper.GetString("assets_dir")
	if assetsDir == "" {
		assetsDir = filepath.Join(cozyHome, "assets")
	}

	assetsDir, err := pathutil.ExpandPath(assetsDir)
	if err != nil {
		return "", ErrCozyHomeExpandFailed
	}

	return assetsDir, nil
}

func getModelsDir(cozyHome string) (string, error) {
	if cozyHome == "" {
		return "", ErrCozyHomeNotSet
	}

	modelsDir := viper.GetString("models_dir")
	if modelsDir == "" {
		modelsDir = filepath.Join(cozyHome, "models")
	}

	modelsDir, err := pathutil.ExpandPath(modelsDir)
	if err != nil {
		return "", ErrCozyHomeExpandFailed
	}

	viper.Set("models_dir", modelsDir)
	return modelsDir, nil
}

func getTempDir(cozyHome string) (string, error) {
	if cozyHome == "" {
		return "", ErrCozyHomeNotSet
	}

	tempDir := viper.GetString("temp_dir")
	if tempDir == "" {
		tempDir = filepath.Join(cozyHome, "temp")
	}

	tempDir, err := pathutil.ExpandPath(tempDir)
	if err != nil {
		return "", fmt.Errorf("failed to expand cozy home path: %w", err)
	}

	return tempDir, nil
}

func createCozyHomeDirs(cozyHome string) error {
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
