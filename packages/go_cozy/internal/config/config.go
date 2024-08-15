package config

import (
	"fmt"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/viper"
)

const (
	FilesystemLocal = "local"
	FilesystemS3    = "s3"
)

type Config struct {
	Port          int       `mapstructure:"port"`
	Host          string    `mapstructure:"host"`
	Environment   string    `mapstructure:"environment"`
	AssetsPath    string    `mapstructure:"assets_path"`
	ModelsPath    string    `mapstructure:"models_path"`
	WorkspacePath string    `mapstructure:"workspace_path"`
	Filesystem    string    `mapstructure:"filesystem_type"`
	S3            *S3Config `mapstructure:"s3"`
}

type S3Config struct {
	Folder      string `mapstructure:"folder"`
	Region      string `mapstructure:"region"`
	Bucket      string `mapstructure:"bucket"`
	AccessKey   string `mapstructure:"access_key_id"`
	SecretKey   string `mapstructure:"secret_access_key"`
	EndpointUrl string `mapstructure:"endpoint_url"`
}

var config *Config

func InitConfig() error {
	envFile := viper.GetString("env-file")
	configFile := viper.GetString("config-file")
	workspacePath := viper.GetString("workspace-path")

	if envFile != "" {
		if err := godotenv.Load(envFile); err != nil {
			return fmt.Errorf("failed to load env file: %w", err)
		}
	}

	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer(`.`, `_`))
	if configFile != "" {
		viper.SetConfigFile(configFile)
	} else {
		viper.SetConfigType("yaml")
		viper.SetConfigName("config")
		viper.AddConfigPath(workspacePath)
	}

	if err := LoadConfig(false); err != nil {
		return err
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
