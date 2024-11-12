package config

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/gen-server/internal/templates"
	"github.com/cozy-creator/gen-server/internal/utils/pathutil"

	"github.com/joho/godotenv"
	"github.com/spf13/viper"
)

const (
	FilesystemLocal = "local"
	FilesystemS3    = "s3"
)

const cozyPrefix = "COZY"

type Config struct {
	Port          int                   `mapstructure:"port"`
	Host          string                `mapstructure:"host"`
	TcpPort       int                   `mapstructure:"tcp_port"`
	CozyHome      string                `mapstructure:"cozy_home"`
	TcpTimeout    int                   `mapstructure:"tcp_timeout"`
	Environment   string                `mapstructure:"environment"`
	AssetsDir     string                `mapstructure:"assets_dir"`
	ModelsDir     string                `mapstructure:"models_dir"`
	TempDir       string                `mapstructure:"temp_dir"`
	AuxModelsDirs []string              `mapstructure:"aux_models_dirs"`
	Filesystem    string                `mapstructure:"filesystem_type"`
	MQType        string                `mapstructure:"mq_type"`
	S3            *S3Config             `mapstructure:"s3"`
	Pulsar        *PulsarConfig         `mapstructure:"pulsar"`
	DB            *DBConfig             `mapstructure:"db"`
	LumaAI        *LumaAIConfig         `mapstructure:"luma_ai"`
	Replicate     *ReplicateConfig      `mapstructure:"replicate"`
	EnabledModels []EnabledModelsConfig `mapstructure:"enabled_models"`
	WarmupModels  string                `mapstructure:"warmup_models"`
}

type S3Config struct {
	Folder      string `mapstructure:"folder"`
	Region      string `mapstructure:"region_name"`
	Bucket      string `mapstructure:"bucket_name"`
	AccessKey   string `mapstructure:"access_key"`
	SecretKey   string `mapstructure:"secret_key"`
	PublicUrl   string `mapstructure:"vanity_url"`
	EndpointUrl string `mapstructure:"endpoint_url"`
}

type EnabledModelsConfig struct {
	Type       string                    `mapstructure:"type"`
	Source     string                    `mapstructure:"source"`
	Components map[string]ModelComponent `mapstructure:"components"`
}

type ModelComponent struct {
	Source string `mapstructure:"source"`
}

type PulsarConfig struct {
	URL                    string `mapstructure:"url"`
	OperationTimeout       int    `mapstructure:"operation_timeout"`
	ConnectionTimeout      int    `mapstructure:"connection_timeout"`
	MaxConcurrentConsumers int    `mapstructure:"max_concurrent_consumers"`
}

type DBConfig struct {
	DSN string `mapstructure:"dsn"`
}

type LumaAIConfig struct {
	APIKey string `mapstructure:"api_key"`
}

type ReplicateConfig struct {
	APIKey string `mapstructure:"api_key"`
}

var config *Config

func InitConfig() error {
	cozyHome, err := getCozyHome()
	if err != nil {
		return err
	}

	if err = createCozyHomeDirs(cozyHome); err != nil {
		return err
	}

	// Set the cozy home directory, assets directory, models directory, and temp directory in viper
	viper.Set("cozy_home", cozyHome)
	viper.Set("temp_dir", path.Join(cozyHome, "temp"))
	viper.Set("assets_dir", path.Join(cozyHome, "assets"))
	viper.Set("models_dir", path.Join(cozyHome, "models"))

	envFile := viper.GetString("env_file")
	if envFile == "" {
		envFile = filepath.Join(cozyHome, ".env")
	}

	if _, err := os.Stat(envFile); err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("failed to stat .env file: %w", err)
		}

		if err := templates.WriteEnv(filepath.Join(cozyHome, ".env.example")); err != nil {
			return fmt.Errorf("failed to create .env file: %w", err)
		}
	} else {
		if err := godotenv.Load(envFile); err != nil {
			return fmt.Errorf("failed to load env file: %w", err)
		}
	}

	configFilePath := viper.GetString("config_file")
	if configFilePath == "" {
		configFilePath = filepath.Join(cozyHome, "config.yaml")
	}

	// Check if we should write config.example.yaml file
	_, err = os.Stat(configFilePath)
	if err != nil && os.IsNotExist(err) {
		configExample := filepath.Join(cozyHome, "config.example.yaml")

		// Check if config.example.yaml exists
		_, exampleErr := os.Stat(configExample)
		if os.IsNotExist(exampleErr) {
			// Create config.example.yaml
			if err := templates.WriteConfig(configExample); err != nil {
				return fmt.Errorf("failed to create config.example.yaml file: %w", err)
			}
		} else if exampleErr != nil {
			return fmt.Errorf("failed to stat config.example.yaml file: %w", exampleErr)
		}
	}

	viper.AutomaticEnv()
	viper.SetEnvPrefix(cozyPrefix)
	viper.SetEnvKeyReplacer(strings.NewReplacer(`.`, `_`))
	if configFilePath != "" {
		viper.SetConfigFile(configFilePath)
	} else {
		viper.SetConfigType("yaml")
		viper.SetConfigName("config")
		viper.AddConfigPath(cozyHome)
	}

	if err := LoadConfig(); err != nil {
		if errors.As(err, &viper.ConfigFileNotFoundError{}) {
			fmt.Println("No config file found. Using default config.")
		} else {
			return err
		}
	}

	return nil
}

func LoadConfig() error {
	if err := viper.ReadInConfig(); err != nil {
		return fmt.Errorf("error reading config: %w", err)
	}

	config = &Config{}
	if err := viper.Unmarshal(config); err != nil {
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

func parseWarmupModels() ([]string, error) {
	warmupModels := viper.GetString("warmup_models")
	fmt.Println("Warmup Models", warmupModels)
	if warmupModels == "" {
		return nil, nil
	}

	return strings.Split(warmupModels, ","), nil
}
