package config

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"

	"github.com/cozy-creator/gen-server/internal/templates"

	"github.com/joho/godotenv"
	"github.com/spf13/viper"
)

const (
	FilesystemLocal = "local"
	FilesystemS3    = "s3"
)

type Config struct {
	Port           int                      `mapstructure:"port"`
	Host           string                   `mapstructure:"host"`
	CozyHome       string                   `mapstructure:"cozy_home"`
	Environment    string                   `mapstructure:"environment"`
	AssetsDir      string                   `mapstructure:"assets_dir"`
	ModelsDir      string                   `mapstructure:"models_dir"`
	TempDir        string                   `mapstructure:"temp_dir"`
	AuxModelsDirs  []string                 `mapstructure:"aux-models-dirs"`
	FilesystemType string                   `mapstructure:"filesystem-type"`
	PublicDir      string                   `mapstructure:"public_dir"`
	DisableAuth    bool                     `mapstructure:"disable-auth"`
	S3             *S3Config                `mapstructure:"s3"`
	Pulsar         *PulsarConfig            `mapstructure:"pulsar"`
	DB             *DBConfig                `mapstructure:"db"`
	LumaAI         *LumaAIConfig            `mapstructure:"luma_ai"`
	Replicate      *ReplicateConfig         `mapstructure:"replicate"`
	Civitai        *CivitaiConfig           `mapstructure:"civitai"`
	PipelineDefs   map[string]*PipelineDefs // unmarshalled manually from config.yaml
	WarmupModels   []string                 `mapstructure:"warmup-models"`
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
	ClassName  string                    `mapstructure:"class_name" json:"class_name"`
	Source     string                    `mapstructure:"source" json:"source"`
	Components map[string]*ComponentDefs `mapstructure:"components" json:"components"`
}

type ComponentDefs struct {
	ClassName string `mapstructure:"class_name" json:"class_name"`
	Source    string `mapstructure:"source" json:"source"`
}

type PulsarConfig struct {
	URL                    string `mapstructure:"url"`
	MaxConcurrentConsumers int    `mapstructure:"max-concurrent-consumers"`
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

type CivitaiConfig struct {
	APIKey string `mapstructure:"api_key"`
}

var config *Config

func LoadEnvAndConfigFiles() error {
	// Load .env file into the environment.
	// This does not override any environment variables already set.
	envFilePath := viper.GetString("env_file")
	if envFilePath != "" {
		if err := godotenv.Load(envFilePath); err != nil {
			fmt.Printf("failed to load env file at %s: %v", envFilePath, err)
		}
	} else {
		envFiles := []string{".cozy.env", ".env.local", ".env"}
		for _, file := range envFiles {
			if _, err := os.Stat(file); err == nil {
				godotenv.Load(file)
				break
			}
		}
	}

	// Set the assets directory, models directory, and temp directory in viper
	cozyHome := viper.GetString("cozy_home")
	if err := createCozyHomeDirs(cozyHome); err != nil {
		return err
	}
	viper.Set("temp_dir", path.Join(cozyHome, "temp"))
	viper.Set("assets_dir", path.Join(cozyHome, "assets"))
	viper.Set("models_dir", path.Join(cozyHome, "models"))

	configFilePath := viper.GetString("config_file")
	viper.SetConfigFile(configFilePath)

	if err := readAndUnmarshalConfig(); err != nil {
		return err
	}

	writeExampleTemplates()

	return nil
}

func readAndUnmarshalConfig() error {
	if err := viper.ReadInConfig(); err != nil {
		if errors.As(err, &viper.ConfigFileNotFoundError{}) {
			fmt.Println("No config file found. Using default config.")
		} else {
			// return err
		}
	}

	// Copy Viper's internal config-state into our local struct
	config = &Config{}
	if err := viper.Unmarshal(&config); err != nil {
		return fmt.Errorf("error unmarshalling config: %w", err)
	}

	// Handle PipelineDefs separately
	if raw := viper.Get("pipeline_defs"); raw != nil {
		if rawMap, ok := raw.(map[string]interface{}); ok {
			pipelineDefs := make(map[string]*PipelineDefs)
			for key, val := range rawMap {
				if def := unmarshalPipelineDef(val); def != nil {
					pipelineDefs[key] = def
				}
			}
			config.PipelineDefs = pipelineDefs
		}
	}

	return nil
}

func unmarshalPipelineDef(raw interface{}) *PipelineDefs {
	modelMap, ok := raw.(map[string]interface{})
	if !ok {
		return nil
	}

	def := &PipelineDefs{}

	// Extract class_name
	if className, ok := modelMap["class_name"].(string); ok {
		def.ClassName = className
	}

	// Extract source
	if source, ok := modelMap["source"].(string); ok {
		def.Source = source
	}

	// Extract components
	if componentsRaw, ok := modelMap["components"].(map[string]interface{}); ok {
		def.Components = make(map[string]*ComponentDefs)
		for compKey, compVal := range componentsRaw {
			if compMap, ok := compVal.(map[string]interface{}); ok {
				compDef := &ComponentDefs{}
				if className, ok := compMap["class_name"].(string); ok {
					compDef.ClassName = className
				}
				if source, ok := compMap["source"].(string); ok {
					compDef.Source = source
				}
				def.Components[compKey] = compDef
			}
		}
	}

	return def
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

func writeExampleTemplates() error {
	cozyHome := viper.GetString("cozy_home")
	exampleEnvFilePath := filepath.Join(cozyHome, ".env.example")
	exampleConfigFilePath := filepath.Join(cozyHome, "config.example.yaml")

	// Check to see if exampleEnvFilePath does not exist, and if it doesn't then create it
	if _, err := os.Stat(exampleEnvFilePath); os.IsNotExist(err) {
		err = templates.WriteEnv(exampleEnvFilePath)
		if err != nil {
			return fmt.Errorf("failed to create .env file: %w", err)
		}
	}

	// Check to see if exampleConfigFilePath does not exist, and if it doesn't then create it
	if _, err := os.Stat(exampleConfigFilePath); os.IsNotExist(err) {
		err = templates.WriteConfig(exampleConfigFilePath)
		if err != nil {
			return fmt.Errorf("failed to create config.yaml file: %w", err)
		}
	}

	return nil
}
