package config

import (
	"cozy-creator/gen-server/internal/templates"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/viper"
)

const (
	FilesystemLocal = "local"
	FilesystemS3    = "s3"
)

const cozyPrefix = "COZY"

type Config struct {
	Port           int       `mapstructure:"port"`
	Host           string    `mapstructure:"host"`
	TcpPort        int       `mapstructure:"tcp_port"`
	TcpTimeout     int       `mapstructure:"tcp_timeout"`
	Environment    string    `mapstructure:"environment"`
	AssetsPath     string    `mapstructure:"assets_path"`
	ModelsPath     string    `mapstructure:"models_path"`
	WorkspacePath  string    `mapstructure:"workspace_path"`
	AuxModelsPaths []string  `mapstructure:"aux_models_paths"`
	Filesystem     string    `mapstructure:"filesystem_type"`
	S3             *S3Config `mapstructure:"s3"`
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
	envFile := viper.GetString("env_file")
	configFile := viper.GetString("config_file")
	workspacePath := viper.GetString("workspace_path")

	if workspacePath != "" {
		if err := os.MkdirAll(workspacePath, os.ModePerm); err != nil {
			return fmt.Errorf("failed to create workspace directory: %w", err)
		}

		envFile = filepath.Join(workspacePath, ".env")
		configFile = filepath.Join(workspacePath, "config.yaml")

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
		viper.AddConfigPath(workspacePath)
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
