package config

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"

	"github.com/cozy-creator/gen-server/internal/templates"
	"github.com/cozy-creator/gen-server/internal/utils/pathutil"

	"github.com/joho/godotenv"
	"github.com/spf13/viper"
)

const (
	FilesystemLocal = "local"
	FilesystemS3    = "s3"
)

// TO DO: it's confusing the mix of dashes - and underscores _ ???
type Config struct {
	Port           int                     `mapstructure:"port"`
	Host           string                  `mapstructure:"host"`
	TcpPort        int                     `mapstructure:"tcp-port"`
	CozyHome       string                  `mapstructure:"cozy-home"`
	Environment    string                  `mapstructure:"environment"`
	DisableAuth    bool                    `mapstructure:"disable-auth"`
	AssetsDir      string                  `mapstructure:"assets_dir"`
	ModelsDir      string                  `mapstructure:"models_dir"`
	TempDir        string                  `mapstructure:"temp_dir"`
	AuxModelsDirs  []string                `mapstructure:"aux_models_dirs"`
	FilesystemType string                  `mapstructure:"filesystem-type"`
	PublicDir      string                  `mapstructure:"public_dir"`
	S3             *S3Config               `mapstructure:"s3"`
	Pulsar         *PulsarConfig           `mapstructure:"pulsar"`
	DB             *DBConfig               `mapstructure:"db"`
	LumaAI         *LumaAIConfig           `mapstructure:"luma_ai"`
	Replicate      *ReplicateConfig        `mapstructure:"replicate"`
	PipelineDefs   map[string]*PipelineDefs `mapstructure:"pipeline_defs"`
	WarmupModels   []string                `mapstructure:"warmup_models"`
}

type S3Config struct {
	Folder      string `mapstructure:"folder"`
	Region      string `mapstructure:"region_name"`
	Bucket      string `mapstructure:"bucket_name"`
	AccessKey   string `mapstructure:"access_key"`
	SecretKey   string `mapstructure:"secret_key"`
	VanityUrl   string `mapstructure:"vanity_url"`
	EndpointUrl string `mapstructure:"endpoint_url"`
}

type PipelineDefs struct {
	ClassName  string                    	`mapstructure:"class_name"`
	Source     string                    	`mapstructure:"source"`
	Components map[string]*PipelineDefs		`mapstructure:"components"`
}

type PulsarConfig struct {
	URL                    string `mapstructure:"url"`
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

func LoadEnvAndConfigFiles() error {
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

	envFilePath := viper.GetString("env_file")
	configFilePath := viper.GetString("config_file")

	// Consider writing example .env and config.yaml files
	writeExampleTemplates(envFilePath, configFilePath, cozyHome)

	// Set default file paths if not provided
	defaultEnvFilePath, _ := getDefaultFilePaths(cozyHome)

	// Load .env file into the environment. Does not override any environment variables already set.
	if envFilePath != "" {
		if err := godotenv.Load(envFilePath); err != nil {
			fmt.Printf("failed to load env file at %s: %v", envFilePath, err)
		}
	} else {
		godotenv.Load(defaultEnvFilePath)
	}

	fmt.Println("config file path:", configFilePath)
	if configFilePath != "" {
		viper.SetConfigFile(configFilePath)
	} else {
		viper.SetConfigType("yaml")
		viper.SetConfigName("config")
		viper.AddConfigPath(cozyHome)
	}

	if err := loadConfig(); err != nil {
		if errors.As(err, &viper.ConfigFileNotFoundError{}) {
			fmt.Println("No config file found. Using default config.")
		} else {
			return err
		}
	}

	// configJSON, _ := json.MarshalIndent(config, "", "  ")
	// fmt.Printf("config that got loaded:\n%s\n", string(configJSON))

	return nil
}

func loadConfig() error {
	if err := viper.ReadInConfig(); err != nil {
		return fmt.Errorf("error reading config: %w", err)
	}

	// Config struct is no longer nil
	config = &Config{}

	// Copy Viper's internal config-state into our local struct
	if err := viper.Unmarshal(&config); err != nil {
		return fmt.Errorf("error unmarshalling config: %w", err)
	}

	return nil
}

func IsLoaded() bool {
	return config != nil
}

func MustGetConfig() *Config {
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
// TO DO: shouldn't viper take care of these env and defaults for us? So we don't do this manually?
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

func writeExampleTemplates(envFilePath string, configFilePath string, cozyHome string) error {
	defaultEnvFilePath, defaultConfigFilePath := getDefaultFilePaths(cozyHome)

	exampleEnvFilePath := filepath.Join(cozyHome, ".env.example")
	exampleConfigFilePath := filepath.Join(cozyHome, "config.example.yaml")

	// Consider writing example .env and config.yaml files
	if _, err := os.Stat(defaultEnvFilePath); envFilePath == "" && os.IsNotExist(err) {
		if _, err := os.Stat(exampleEnvFilePath); os.IsNotExist(err) {
			err = templates.WriteEnv(exampleEnvFilePath)
			if err != nil {
				return fmt.Errorf("failed to create .env file: %w", err)
			}
		}
	}

	if _, err := os.Stat(defaultConfigFilePath); configFilePath == "" && os.IsNotExist(err) {
		if _, err := os.Stat(exampleConfigFilePath); os.IsNotExist(err) {
			err = templates.WriteConfig(exampleConfigFilePath)
			if err != nil {
				return fmt.Errorf("failed to create config.yaml file: %w", err)
			}
		}
	}

	return nil
}

func getDefaultFilePaths(cozyHome string) (string, string) {
	return filepath.Join(cozyHome, ".env"), filepath.Join(cozyHome, "config.yaml")
}
